import time
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError


class InferenceGatewayError(Exception):
    """Base exception for all gateway errors."""

    pass


class ConfigurationError(InferenceGatewayError):
    """Raised when the gateway configuration is invalid."""

    pass


class APIError(InferenceGatewayError):
    """Raised when the Hugging Face API returns a non-200 status code."""

    pass


class ParsingError(InferenceGatewayError):
    """Raised when the model response cannot be parsed as JSON."""

    pass


class GatewayConfig(BaseModel):
    """
    Configuration for the Hugging Face Inference Gateway.
    """

    api_token: SecretStr
    model_id: str
    timeout: float = Field(default=30.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    base_url: str = "https://api-inference.huggingface.co"


class InferenceResult(BaseModel):
    """
    Standardized result object returned after a successful inference call.
    """

    payload: Dict[str, Any]
    raw_text: str
    latency_ms: float
    model_id: str


class HuggingFaceGateway:
    """
    Domain-agnostic client for the Hugging Face Inference API.
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._client = httpx.Client(base_url=config.base_url, timeout=config.timeout)

    def execute_inference(
        self,
        message: str,
        context: Dict[str, Any],
        system_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        """
        Executes the inference request against the configured model.

        Args:
            message: The user input or query.
            context: Arbitrary dictionary of state/context variables.
            system_prompt: Instructions defining the model's behavior and output format.
            response_schema: Optional Pydantic model to validate the JSON response.

        Returns:
            InferenceResult object containing the parsed payload and metadata.
        """
        start_time = time.time()

        try:
            # 1. Construct the prompt payload (Logic to be implemented in detail later)
            payload = self._build_payload(message, context, system_prompt)

            # 2. Send request to API (Logic to be implemented in detail later)
            raw_response = self._send_request(payload)

            # 3. Parse and Validate (Logic to be implemented in detail later)
            parsed_payload = self._parse_response(raw_response, response_schema)

            latency = (time.time() - start_time) * 1000

            return InferenceResult(
                payload=parsed_payload,
                raw_text=raw_response,
                latency_ms=latency,
                model_id=self.config.model_id,
            )

        except httpx.HTTPStatusError as e:
            raise APIError(f"API request failed with status {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise APIError(f"Network error occurred: {e}") from e
        except Exception as e:
            raise InferenceGatewayError(f"Unexpected error during inference: {e}") from e

    def _build_payload(self, message: str, context: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """Constructs the JSON payload for the HF API."""
        # Implementation detail: Combine system prompt, context, and message
        # into the format expected by the specific model.
        raise NotImplementedError("Payload building logic not yet implemented.")

    def _send_request(self, payload: Dict[str, Any]) -> str:
        """Sends the HTTP POST request and returns raw text."""
        # Implementation detail: Handle retries and authorization header.
        raise NotImplementedError("HTTP request logic not yet implemented.")

    def _parse_response(self, raw_text: str, schema: Optional[Type[BaseModel]]) -> Dict[str, Any]:
        """Parses the raw text response into JSON and validates against schema."""
        # Implementation detail: Extract JSON from markdown blocks if necessary.
        raise NotImplementedError("Response parsing logic not yet implemented.")