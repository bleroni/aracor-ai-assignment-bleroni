from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

cohere_model = init_chat_model(
    "command",             # the Cohere model name
    model_provider="cohere",
    temperature=0,
    cohere_api_key=cohere_api_key
)

# print("Cohere model: " + cohere_model.invoke("what's your name").content + "\n")
print("Model name: " + cohere_model.model_name)
