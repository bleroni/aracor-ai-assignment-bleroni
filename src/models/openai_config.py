"""
Module for configuring and initializing the OpenAI model.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.utils.rate_limiting import rate_limiter

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
