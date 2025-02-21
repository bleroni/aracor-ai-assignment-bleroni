"""
Module for configuring and initializing the OpenAI model.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.utils.rate_limiting import rate_limiter

# from langchain.schema import SystemMessage, HumanMessage


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai_model = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    name="gpt-4o",
    temperature=0,
    openai_api_key=openai_api_key,
    rate_limiter=rate_limiter,
    max_retries=5,
)

# system_prompt = SystemMessage(content="You are a helpful assistant named Doctor Green.")

# # Build the conversation history with previous messages
# message_history = [
#     system_prompt,
#     HumanMessage(content="Hello, what's your name?")
# ]

# # Call the model with the history
# response = openai_model.invoke(message_history)
# print(response.content)

# print("GPT-4o: " + gpt_4o_model.invoke("what's your name").content + "\n")
# print("Model name: " + gpt_4o_model.model_name)
