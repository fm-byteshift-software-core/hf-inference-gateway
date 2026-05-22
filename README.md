# hf-inference-gateway

Foundational, domain-agnostic client for the Hugging Face Inference API.

## Overview

`hf-inference-gateway` provides a lightweight, framework-agnostic interface for interacting with Hugging Face's Router endpoint (OpenAI-compatible) and standard inference APIs. It abstracts transport, retry logic, timeout handling, and strict JSON validation while remaining completely independent of any specific business domain.

Designed for rapid prototyping and scalable architecture, this module can be integrated into any application requiring structured LLM responses without coupling to external business logic.

## Features

- Domain-agnostic design: Zero hardcoded business rules or vertical-specific terminology
- Configurable model routing: Support any Hugging Face model via dynamic `model_id` injection
- OpenAI-compatible format: Native support for `/chat/completions` workflows
- Automatic retry logic: Exponential backoff with configurable limits and smart error filtering
- Strict response validation: Optional Pydantic schema enforcement on model outputs
- Timeout management: Configurable request deadlines with predictable failure modes
- Connection pooling: Persistent HTTP client for efficient request handling

## Installation

Stable releases are published to PyPI. Development versions can be installed directly from the repository.

```bash
# Install from PyPI (once published)
pip install hf-inference-gateway

# Install from source for development
pip install git+https://github.com/fm-byteshift-software-core/hf-inference-gateway.git
```

Requires Python 3.10 or higher.

## Quick Start

```python
import os
from hf_inference_gateway import HuggingFaceGateway, GatewayConfig

# Initialize configuration
config = GatewayConfig(
    api_token=os.getenv("HF_API_TOKEN"),
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    base_url="https://router.huggingface.co/v1",
    timeout=30.0,
    max_retries=2
)

# Instantiate gateway
gateway = HuggingFaceGateway(config)

# Execute inference with arbitrary context
result = gateway.execute_inference(
    message="Where is my order?",
    context={
        "status": "in_transit",
        "eta_minutes": 12,
        "attempt_count": 1
    },
    system_prompt="You are a support assistant. Return a JSON object with intent, sentiment, and response fields.",
    response_schema=None  # Optional Pydantic model for validation
)

print(result.payload)
print(result.latency_ms)
```

## Configuration

| Parameter     | Type    | Default                            | Description                                    |
| ------------- | ------- | ---------------------------------- | ---------------------------------------------- |
| `api_token`   | `str`   | Required                           | Hugging Face API authentication token          |
| `model_id`    | `str`   | Required                           | Hugging Face model identifier                  |
| `timeout`     | `float` | `30.0`                             | Maximum request duration in seconds            |
| `max_retries` | `int`   | `3`                                | Number of retry attempts on transient failures |
| `base_url`    | `str`   | `https://router.huggingface.co/v1` | Inference API endpoint (OpenAI-compatible)     |

## Error Handling

The module raises specific exceptions to enable predictable fallback strategies:

- `InferenceGatewayError`: Base exception for all gateway-related failures
- `ConfigurationError`: Raised when initialization parameters are invalid
- `APIError`: Raised on HTTP failures (status codes, network errors, rate limits)
- `ParsingError`: Raised when the model response cannot be parsed as valid JSON

## Development

```bash
# Clone repository
git clone https://github.com/fm-byteshift-software-core/hf-inference-gateway.git
cd hf-inference-gateway

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

## Maintained By

This project is developed and maintained by **FM ByteShift Software**

**Fernando Magalhães**  
CEO – FM ByteShift Software  
[contact@fmbyteshiftsoftware.com](mailto:contact@fmbyteshiftsoftware.com)  
[fmbyteshiftsoftware.com](https://fmbyteshiftsoftware.com)
