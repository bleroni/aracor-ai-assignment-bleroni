"""
Module for configuring and initializing the Anthropic model.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.utils.rate_limiting import rate_limiter

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

anthropic_model = init_chat_model(
    "claude-3-5-sonnet-20240620",
    model_provider="anthropic",
    name="anthropic-claude",
    temperature=0,
    anthropic_api_key=anthropic_api_key,
    rate_limiter=rate_limiter,
    max_retries=5,
)
