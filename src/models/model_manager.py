"""Module for managing model clients.

This module provides the ModelManager class, which allows you to switch between
different model clients (OpenAI and Anthropic) and invoke the default client.
"""

# from langchain.chat_models.base import BaseChatModel

from src.models.anthropic_config import anthropic_model

# from src.models.cohere_config import cohere_model
from src.models.openai_config import openai_model


class ModelManager:
    """Manages different language model clients and their invocation.

    Attributes:
        client1: Instance of the OpenAI model client.
        client2: Instance of the Anthropic model client.
        default_client: The current default client used for invoking models.
    """

    def __init__(self):
        """Initializes the ModelManager with default model clients."""
        self.client1 = openai_model
        self.client2 = anthropic_model
        self.default_client = self.client1  # Initial default

    def switch_client(self, new_default):
        """Switches the default model client.

        Args:
            new_default (str): The name of the client to switch to.

        Raises:
            ValueError: If an unknown client name is provided.
        """
        if new_default == "openai":
            self.default_client = self.client1
        elif new_default == "anthropic":
            self.default_client = self.client2
        else:
            raise ValueError("Unknown client")

    def call_model(self):
        """Calls the default model's invoke method and prints its response.

        Sends a query to the default client and prints response and model name.
        """
        resp = self.default_client.invoke("what's your name").content
        print(f"Response from {self.default_client.name}: {resp}")


if __name__ == "__main__":
    model_manager = ModelManager()
    model_manager.call_model()  # Default is OpenAI

    # Switch to Anthropic and call the model again
    model_manager.switch_client("anthropic")
    model_manager.call_model()
