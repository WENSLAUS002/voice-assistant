import os

class Config:
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = "banking_db"
    MONGO_COLLECTION_FAQ = "Banking"
    
    # MySQL Configuration
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_DB = os.getenv("MYSQL_DB", "banking_db")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "wenslaus001")
    if not MYSQL_PASSWORD:
        raise ValueError("MYSQL_PASSWORD is not set!")

    # OpenAI GPT Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
    
    # NLP Model Paths
    BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH") or "bert-base-uncased"
    SPACY_MODEL = os.getenv("SPACY_MODEL") or "en_core_web_sm"

    # Rasa Configuration
    RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005/webhooks/rest/webhook")
    
    # Other Configurations
    DEBUG = os.getenv("DEBUG", "True") == "True"
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
