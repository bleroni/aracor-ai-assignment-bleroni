import io
import pytest

from contextlib import redirect_stdout
from importlib import reload
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any, Sequence

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig

import src.models.cohere_config as cohere_config

# Import the configuration modules so we can test API key loading.
import src.models.openai_config as openai_config
from src.models.cohere_config import cohere_model
from src.models.model_manager import ModelManager
from src.models.openai_config import openai_model


# Dummy response class that mimics the expected behavior.
class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyClient(BaseChatModel):
    # Define fields with Pydantic-compatible type hints and Field for validation
    name: str = Field(..., description="Name of the model")
    response_content: str = Field(default="Default response", description="Response content")
    raise_exception: bool = Field(default=False, description="Whether to raise an exception")

    # Pydantic v2 model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow arbitrary types if needed by BaseChatModel
        extra="forbid",  # Disallow extra fields to enforce strict validation
    )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        if self.raise_exception:
            raise Exception("Simulated API error")

        response_message = AIMessage(content=self.response_content)
        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "dummy_client"

    def invoke(
        self,
        input: Any,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> BaseMessage:
        if self.raise_exception:
            raise Exception("Simulated API error")
        return AIMessage(content=self.response_content)


def test_openai_model_initialization():
    """
    Test that the ModelManager initializes with the OpenAI model as the default client.
    """
    manager = ModelManager()
    # Check that client1 is the OpenAI model and that it is set as the default.
    assert manager.client1 == openai_model
    assert manager.default_client == openai_model


def test_cohere_model_initialization():
    """
    Test that switching to the Cohere model works as expected.
    """
    manager = ModelManager()
    # Switch to the Cohere model
    manager.switch_client("cohere")
    # Verify that client2 is the Cohere model and that the default has been updated.
    assert manager.client2 == cohere_model
    assert manager.default_client == cohere_model


def test_api_key_validation(monkeypatch):
    """
    Test that the API keys are correctly set by load_dotenv().
    We set dummy API keys using monkeypatch and then reload the config modules.
    """
    # Set dummy API keys.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_openai_key")
    monkeypatch.setenv("COHERE_API_KEY", "dummy_cohere_key")

    # Reload the modules so that they pick up the new environment variables.
    reload(openai_config)
    reload(cohere_config)

    # Verify that the keys are correctly set.
    assert openai_config.openai_api_key == "dummy_openai_key"
    assert cohere_config.cohere_api_key == "dummy_cohere_key"


def test_model_switching():
    """
    Test that ModelManager switches models correctly and that call_model()
    prints the expected dummy responses.
    """
    from src.models.model_manager import ModelManager

    # Create dummy clients for testing.
    dummy_openai = DummyClient(name="openai", response_content="OpenAI response")
    dummy_cohere = DummyClient(name="cohere", response_content="Cohere response")

    # Instantiate the ModelManager and override the real clients.
    manager = ModelManager()
    manager.client1 = dummy_openai
    manager.client2 = dummy_cohere
    manager.default_client = dummy_openai

    # Verify that the default client is the dummy OpenAI client.
    assert manager.default_client == dummy_openai

    # Capture the output of call_model() and check it includes the OpenAI response.
    f = io.StringIO()
    with redirect_stdout(f):
        manager.call_model()
    output = f.getvalue()
    assert "OpenAI response" in output

    # Switch to the Cohere client and verify the switch.
    manager.switch_client("cohere")
    assert manager.default_client == dummy_cohere

    # Capture the output after switching and check for the Cohere response.
    f = io.StringIO()
    with redirect_stdout(f):
        manager.call_model()
    output = f.getvalue()
    assert "Cohere response" in output

    # Confirm that an invalid client name raises a ValueError.
    with pytest.raises(ValueError):
        manager.switch_client("invalid")


def test_call_model_error_response(capsys):
    mm = ModelManager()
    # Override the default client with one that simulates an error response
    mm.default_client = DummyClient(name="dummy_model", response_content="Error: Invalid API key")
    mm.call_model()
    captured = capsys.readouterr().out
    assert "Response from dummy_model: Error: Invalid API key" in captured
