import json
import time
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel

from .exceptions import APIError, InferenceGatewayError, ParsingError
from .schemas import GatewayConfig, InferenceResult


class HuggingFaceGateway:
    """Domain-agnostic client for Hugging Face Router (OpenAI-compatible)."""

    def __init__(self, config: GatewayConfig):
        if not config.api_token.get_secret_value():
            raise ValueError("API token cannot be empty.")
        
        self.config = config
        self._headers = {
            "Authorization": f"Bearer {config.api_token.get_secret_value()}",
            "Content-Type": "application/json",
            "User-Agent": "hf-inference-gateway/0.1.0"
        }
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            http2=False
        )

    def execute_inference(
        self,
        message: str,
        context: Dict[str, Any],
        system_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        start_time = time.time()

        try:
            payload = self._build_payload(message, context, system_prompt)
            raw_response = self._send_request(payload)
            parsed_payload = self._parse_response(raw_response, response_schema)

            latency = (time.time() - start_time) * 1000

            return InferenceResult(
                payload=parsed_payload,
                raw_text=raw_response,
                latency_ms=latency,
                model_id=self.config.model_id,
            )
        except Exception as e:
            if isinstance(e, (APIError, InferenceGatewayError, ParsingError)):
                raise
            raise InferenceGatewayError(f"Unexpected error during inference: {e}") from e

    def _build_payload(
        self,
        message: str,
        context: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        final_system_content = system_prompt
        if context:
            context_json = json.dumps(context, indent=2, ensure_ascii=False)
            final_system_content += f"\n\n### CURRENT CONTEXT ###\n{context_json}"

        return {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": final_system_content},
                {"role": "user", "content": message}
            ],
            "temperature": 0.1
        }

    def _send_request(self, payload: Dict[str, Any]) -> str:
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers=self._headers
                )
                response.raise_for_status()
                
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code
                
                if status_code == 429 or status_code >= 500:
                    if attempt < self.config.max_retries:
                        wait_time = min(2 ** attempt, 30)
                        time.sleep(wait_time)
                        continue
                raise APIError(f"API request failed with status {status_code}: {e.response.text}") from e
                
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                    continue
                raise APIError(f"Network error occurred: {e}") from e
                
        raise last_exception if last_exception else InferenceGatewayError("Max retries exceeded")

    def _parse_response(self, raw_text: str, schema: Optional[Type[BaseModel]]) -> Dict[str, Any]:
        cleaned_text = raw_text.strip()
        
        import re
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = cleaned_text[first_brace:last_brace + 1]
            else:
                json_str = cleaned_text

        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParsingError(f"Failed to parse model output as JSON: {e}") from e

        if schema:
            try:
                validated_model = schema.model_validate(parsed_data)
                return validated_model.model_dump()
            except Exception as e:
                raise ParsingError(f"Response validation failed against schema: {e}") from e

        return parsed_data