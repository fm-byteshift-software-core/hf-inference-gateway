import time
import json
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel

from .exceptions import APIError, InferenceGatewayError
from .schemas import GatewayConfig, InferenceResult


class HuggingFaceGateway:
    """Domain-agnostic client for the Hugging Face Inference API."""

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
        """
        Executes the inference request against the configured model.
        """
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
            # Re-raise domain exceptions directly, wrap others
            if isinstance(e, (APIError, InferenceGatewayError)):
                raise
            raise InferenceGatewayError(f"Unexpected error during inference: {e}") from e

    def _send_request(self, payload: Dict[str, Any]) -> str:
        """
        Sends the HTTP POST request with retry logic and exponential backoff.
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(
                    f"/models/{self.config.model_id}",
                    json=payload,
                    headers=self._headers
                )
                response.raise_for_status()
                return response.text
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code
                
                # Do not retry on 4xx errors except rate limits (429)
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

    def _build_payload(
        self,
        message: str,
        context: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Constructs the standardized payload for the inference API.

        Strategy:
        1. Injects arbitrary 'context' as a structured JSON block within the system prompt.
           This ensures the LLM has access to domain state (order details, etc.) 
           without modifying the core logic of the module.
        2. Formats the request as a conversation list (ChatML), which is the 
           de-facto standard for Hugging Face's conversational endpoints.
        """
        # Inject context into system prompt if provided
        final_system_content = system_prompt
        if context:
            # JSON format is preferred over plain text for LLMs to parse context reliably
            context_json = json.dumps(context, indent=2, ensure_ascii=False)
            final_system_content += f"\n\n### CURRENT CONTEXT ###\n{context_json}"

        # Return the standard HF Inference payload structure
        return {
            "inputs": [
                {"role": "system", "content": final_system_content},
                {"role": "user", "content": message}
            ]
        }

    def _parse_response(self, raw_text: str, schema: Optional[Type[BaseModel]]) -> Dict[str, Any]:
        """Parses the raw text response into JSON and validates against schema."""
        raise NotImplementedError("Response parsing logic not yet implemented.")