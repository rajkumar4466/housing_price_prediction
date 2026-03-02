"""Unit tests for Lambda handler — mocks ONNX and tokenizer."""

from unittest.mock import MagicMock, patch
import sys


def _mock_dependencies():
    """Mock heavy dependencies so handler can import without model artifacts."""
    mock_ort = MagicMock()
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    return mock_ort, mock_tokenizer_cls, mock_tokenizer


@patch.dict(sys.modules, {"prompt_utils": MagicMock()})
@patch("onnxruntime.InferenceSession")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_app_creates_successfully(mock_tokenizer_from, mock_session):
    """Verify FastAPI app initializes without errors."""
    mock_tokenizer_from.return_value = MagicMock(eos_token_id=0)
    mock_session.return_value = MagicMock()

    from importlib import reload
    import handler  # noqa: F811
    reload(handler)

    assert handler.app is not None
    assert handler.app.title == "NJ Housing Price Predictor"
