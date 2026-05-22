"""Custom exceptions for the Hugging Face Inference Gateway."""

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