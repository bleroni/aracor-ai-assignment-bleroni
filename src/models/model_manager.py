from src.models.openai_config import openai_model
from src.models.cohere_config import cohere_model


class ModelManager:
    def __init__(self):
        self.client1 = openai_model
        self.client2 = cohere_model
        self.default_client = self.client1  # Initial default

    def switch_client(self, new_default):
        if new_default == "openai":
            self.default_client = self.client1
        elif new_default == "cohere":
            self.default_client = self.client2
        else:
            raise ValueError("Unknown client")

    def call_model(self):
        resp = self.default_client.invoke("what's your name").content
        print(f"Response from {self.default_client.model_name}: {resp}")


if __name__ == "__main__":
    model_manager = ModelManager()
    model_manager.call_model()  # Default is OpenAI

    # Switch to Cohere and call the model again
    model_manager.switch_client("cohere")
    model_manager.call_model()
