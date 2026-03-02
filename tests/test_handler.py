"""Unit tests for Lambda handler — mocks ONNX and tokenizer."""

import sys
from unittest.mock import MagicMock

# Mock all heavy dependencies BEFORE handler is imported anywhere.
# handler.py imports these at module level, so they must exist in sys.modules.
sys.modules["prompt_utils"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["numpy"] = MagicMock()

import handler  # noqa: E402


def test_app_creates_successfully():
    """Verify FastAPI app initializes without errors."""
    assert handler.app is not None
    assert handler.app.title == "NJ Housing Price Predictor"


def test_predict_endpoint_exists():
    """Verify /predict route is registered."""
    routes = [r.path for r in handler.app.routes]
    assert "/predict" in routes
