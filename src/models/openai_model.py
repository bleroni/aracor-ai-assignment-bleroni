from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

gpt_4o_model = init_chat_model(
  "gpt-4o",
  model_provider="openai",
  temperature=0,
  openai_api_key=openai_api_key
)

print("GPT-4o: " + gpt_4o_model.invoke("what's your name").content + "\n")
