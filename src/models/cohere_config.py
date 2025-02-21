"""
Module for configuring and initializing the Cohere model.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.utils.rate_limiting import rate_limiter

# from langchain.schema import SystemMessage, HumanMessage


load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable is not set")

cohere_model = init_chat_model(
    "command",  # the Cohere model name
    model_provider="cohere",
    temperature=0,
    cohere_api_key=cohere_api_key,
    rate_limiter=rate_limiter,
    max_retries=5,
)

# system_prompt = SystemMessage(content="You are a helpful assistant named Doctor Sebastian.")

# message_history = [
#     system_prompt,
#     HumanMessage(content="Hello, what's your name?")
# ]

# response = cohere_model.invoke(message_history)
# print(response.content)
