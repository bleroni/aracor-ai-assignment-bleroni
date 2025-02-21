import io
from contextlib import redirect_stdout
from importlib import reload

import pytest

import src.models.cohere_config as cohere_config

# Import the configuration modules so we can test API key loading.
import src.models.openai_config as openai_config
from src.models.cohere_config import cohere_model
from src.models.model_manager import ModelManager
from src.models.openai_config import openai_model


# Dummy client class that mimics the expected behavior.
class DummyResponse:
    def __init__(self, content):
        self.content = content


class DummyClient:
    def __init__(self, model_name, response_content=None, raise_exception=False):
        self.model_name = model_name
        self.response_content = response_content
        self.raise_exception = raise_exception

    def invoke(self, prompt):
        if self.raise_exception:
            raise Exception("Simulated API error")
        return DummyResponse(self.response_content)


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
    dummy_openai = DummyClient("openai", "OpenAI response")
    dummy_cohere = DummyClient("cohere", "Cohere response")

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
    mm.default_client = DummyClient("dummy_model", response_content="Error: Invalid API key")
    mm.call_model()
    captured = capsys.readouterr().out
    assert "Response from dummy_model: Error: Invalid API key" in captured


def test_call_model_response_content_error():
    # Define a response class that raises an error when accessing its content.
    class ErrorResponse:
        @property
        def content(self):
            raise Exception("Simulated error in response content")

    # Dummy client that returns an ErrorResponse instance.
    class DummyClientWithErrorResponse:
        def __init__(self, model_name):
            self.model_name = model_name

        def invoke(self, prompt):
            return ErrorResponse()

    mm = ModelManager()
    mm.default_client = DummyClientWithErrorResponse("dummy_model")
    with pytest.raises(Exception) as excinfo:
        mm.call_model()
    assert "Simulated error in response content" in str(excinfo.value)
