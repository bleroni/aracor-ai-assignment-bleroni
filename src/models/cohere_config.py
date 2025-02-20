from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

cohere_model = init_chat_model(
    "command",             # the Cohere model name
    model_provider="cohere",
    temperature=0,
    cohere_api_key=cohere_api_key
)

# system_prompt = SystemMessage(content="You are a helpful assistant named Doctor Sebastian.")

# message_history = [
#     system_prompt,
#     HumanMessage(content="Hello, what's your name?")
# ]

# response = cohere_model.invoke(message_history)
# print(response.content)
