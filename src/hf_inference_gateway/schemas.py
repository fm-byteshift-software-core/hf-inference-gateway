"""Pydantic schemas for configuration and inference results."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, SecretStr

class GatewayConfig(BaseModel):
    """Configuration for the Hugging Face Inference Gateway."""
    api_token: SecretStr
    model_id: str
    timeout: float = Field(default=30.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    base_url: str = "https://router.huggingface.co/v1"

class InferenceResult(BaseModel):
    """Standardized result object returned after a successful inference call."""
    payload: Dict[str, Any]
    raw_text: str
    latency_ms: float
    model_id: str