import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    #GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/complete")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    VECTOR_STORE_PATH = "./data/vector_store"
    MODEL_NAME = os.getenv("MODEL_NAME")  # Update with your Groq model identifier if needed
    LORA_OUTPUT_DIR = "./data/lora_model"
    RLHF_FEEDBACK_LOG = "./logs/rlhf_feedback_logs.json"

config = Config()
