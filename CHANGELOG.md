# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-22

### Added

- Domain-agnostic gateway client for Hugging Face Router (OpenAI-compatible endpoint)
- Configurable model routing via `model_id` parameter
- Automatic retry logic with exponential backoff for transient failures (429, 5xx)
- Structured payload construction with JSON context injection
- Robust response parsing with markdown/code block extraction and bracket fallback
- Optional Pydantic schema validation for model outputs
- Comprehensive exception hierarchy (`APIError`, `ParsingError`, `ConfigurationError`)
- Persistent HTTP client with connection pooling and configurable timeouts
- Unit tests for parsing logic and schema validation
- Full documentation and quick start examples
