"""
hf-inference-gateway: Domain-agnostic client for Hugging Face Inference API.

This module provides a reusable, framework-agnostic gateway for interacting
with the Hugging Face Inference API, supporting configurable models,
retry logic, and structured response validation.
"""

from .exceptions import (
    APIError,
    ConfigurationError,
    InferenceGatewayError,
    ParsingError,
)
from .client import HuggingFaceGateway
from .schemas import GatewayConfig, InferenceResult

__version__ = "0.1.0"
__all__ = [
    "HuggingFaceGateway",
    "GatewayConfig",
    "InferenceResult",
    "InferenceGatewayError",
    "ConfigurationError",
    "APIError",
    "ParsingError",
    "__version__",
]