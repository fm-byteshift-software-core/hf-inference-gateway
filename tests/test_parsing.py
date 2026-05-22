import pytest
from pydantic import BaseModel, Field

from hf_inference_gateway.client import HuggingFaceGateway
from hf_inference_gateway.exceptions import ParsingError
from hf_inference_gateway.schemas import GatewayConfig


class MockResponseSchema(BaseModel):
    intent: str
    sentiment: str = Field(default="NEUTRAL")


class TestParsingLogic:
    @pytest.fixture
    def gateway(self):
        config = GatewayConfig(
            api_token="test_token_placeholder",
            model_id="test-model",
            timeout=5.0,
            max_retries=0
        )
        return HuggingFaceGateway(config)

    def test_parse_clean_json(self, gateway):
        raw = '{"intent": "DELIVERY_ETA", "sentiment": "NEUTRAL"}'
        result = gateway._parse_response(raw, None)
        assert result["intent"] == "DELIVERY_ETA"
        assert result["sentiment"] == "NEUTRAL"

    def test_parse_markdown_wrapped_json(self, gateway):
        raw = '''
        ```json
        {
            "intent": "REFUND_STATUS",
            "sentiment": "FRUSTRATED"
        }
        ```
        '''
        result = gateway._parse_response(raw, None)
        assert result["intent"] == "REFUND_STATUS"
        assert result["sentiment"] == "FRUSTRATED"

    def test_parse_conversational_fallback(self, gateway):
        raw = (
            'Sure, here is the analysis:\n'
            '{"intent": "GENERAL_SUPPORT", "sentiment": "NEUTRAL"}\n'
            'Let me know if you need anything else.'
        )
        result = gateway._parse_response(raw, None)
        assert result["intent"] == "GENERAL_SUPPORT"

    def test_validate_against_schema_success(self, gateway):
        raw = '{"intent": "ORDER_STATUS"}'
        result = gateway._parse_response(raw, MockResponseSchema)
        assert isinstance(result, dict)
        assert result["intent"] == "ORDER_STATUS"
        assert result["sentiment"] == "NEUTRAL"

    def test_validate_against_schema_failure(self, gateway):
        raw = '{"invalid_field": "test"}'
        with pytest.raises(ParsingError, match="Response validation failed against schema"):
            gateway._parse_response(raw, MockResponseSchema)

    def test_invalid_json_raises_parsing_error(self, gateway):
        raw = 'This is not valid json at all'
        with pytest.raises(ParsingError, match="Failed to parse model output as JSON"):
            gateway._parse_response(raw, None)