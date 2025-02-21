# test_summary_generator.py
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.models.model_manager import ModelManager
from src.models.summary import SummaryGenerator


@pytest.fixture
def mock_model_manager():
    mock_mgr = Mock(spec=ModelManager)
    mock_mgr.default_client = Mock()
    mock_mgr.default_client.invoke = Mock(return_value=Mock(content="Mocked summary"))
    mock_mgr.default_client.ainvoke = AsyncMock(return_value=Mock(content="Mocked summary"))
    return mock_mgr


@pytest.fixture
def summary_generator(mock_model_manager):
    generator = SummaryGenerator(mock_model_manager, chunk_size=50, timeout=5)
    yield generator
    mock_model_manager.default_client.invoke.reset_mock()
    mock_model_manager.default_client.ainvoke.reset_mock()


def test_summary_generation(summary_generator):
    text = "Short test text."
    result = summary_generator.generate_summary(text, "brief")
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.invoke.called


def test_detailed_summary_generation(summary_generator):
    text = "Detailed test text."
    result = summary_generator.generate_summary(text, "detailed")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "detailed" in summary_generator.model_manager.default_client.invoke.call_args[0][0][0].content.lower()  # noqa


def test_bullet_point_summary_generation(summary_generator):
    text = "Bullet test text."
    result = summary_generator.generate_summary(text, "bullet")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "bullet" in summary_generator.model_manager.default_client.invoke.call_args[0][0][0].content.lower()  # noqa


@pytest.mark.asyncio
async def test_handling_very_long_input(summary_generator):
    long_text = "This is a very long text " * 1000
    assert len(long_text) > summary_generator.chunk_size
    # Directly await the async path since generate_summary uses asyncio.run()
    chunks = summary_generator._chunk_text(long_text)
    result = " ".join(await summary_generator._summarize_chunks(chunks, "brief"))
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.ainvoke.called


def test_handling_very_short_input(summary_generator):
    short_text = "Short"
    result = summary_generator.generate_summary(short_text, "brief")
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.invoke.called


def test_handling_special_characters(summary_generator):
    special_text = "Text with !@#$%^&*() chars"  # Keep under 50 chars for sync path
    assert len(special_text) <= summary_generator.chunk_size
    result = summary_generator.generate_summary(special_text, "brief")
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.invoke.called


def test_handling_multiple_languages(summary_generator):
    multi_lang_text = "Hello, Hola, 你好!"  # Keep under 50 chars
    result = summary_generator.generate_summary(multi_lang_text, "brief")
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.invoke.called


def test_handling_technical_content(summary_generator):
    tech_text = "Python 3.9 type hints"  # Keep under 50 chars
    result = summary_generator.generate_summary(tech_text, "detailed")
    assert isinstance(result, str)
    assert len(result) > 0
    assert summary_generator.model_manager.default_client.invoke.called


def test_invalid_summary_type(summary_generator):
    with pytest.raises(ValueError) as exc_info:
        summary_generator.generate_summary("Test text", "invalid_type")
    assert "Invalid summary type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_handling(summary_generator):
    text = "Test text " * 10
    summary_generator.model_manager.default_client.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError)
    result = await summary_generator._summarize_chunk(text, "brief")
    assert "[Partial Result - Timeout" in result
    # Adjust expectation: with width=100, full text might fit
    assert len(result) > len("[Partial Result - Timeout after 5s]: ")  # Check some content is included
