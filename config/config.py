import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 🔑 API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    NEWS_API_KEY = os.getenv("NEWSAPI_API_KEY")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    GOLDAPI_API_KEY = os.getenv("GOLDAPI_API_KEY")
    OIL_API_KEY = os.getenv("OIL_API_KEY")
    DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")

    # 🤗 Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN")

    # 📍 Model and Vector Paths
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
    LORA_OUTPUT_DIR = os.getenv("LORA_OUTPUT_DIR", "./data/lora_model")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    RLHF_FEEDBACK_LOG = os.getenv("RLHF_FEEDBACK_LOG", "./logs/rlhf_feedback_logs.json")

config = Config()

