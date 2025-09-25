import torch
import logging
import pandas as pd
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from transformers import BertModel
import torch.nn as nn

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizers
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Initialize SBERT model
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

# Configure Logging
logging.basicConfig(
    filename="logs/system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message):
    """Log messages for debugging and tracking."""
    logging.info(message)

def preprocess_text(text):
    """Cleans and preprocesses input text."""
    text = text.strip().lower()  # Convert to lowercase
    text = text.replace("\n", " ")  # Remove line breaks
    return text

def load_dataset(file_path="banking_dataset.csv"):
    """Loads dataset for training or inference."""
    try:
        df = pd.read_csv(file_path)
        log_message(f"Successfully loaded dataset from {file_path}")
        return df
    except Exception as e:
        log_message(f"Error loading dataset: {e}")
        return None

def tokenize_text(text, model_type="bert"):
    """Tokenizes input text based on the specified model."""
    if model_type == "bert":
        return bert_tokenizer(text, return_tensors="pt").to(DEVICE)
    elif model_type == "gpt":
        return gpt_tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    elif model_type == "t5":
        return t5_tokenizer("summarize: " + text, return_tensors="pt").to(DEVICE)
    else:
        log_message(f"Unsupported model type: {model_type}")
        return None

def get_sentence_embedding(text):
    """Returns sentence embedding using SBERT."""
    return sbert_model.encode(text)

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    vec1 = torch.tensor(vec1).to(DEVICE)
    vec2 = torch.tensor(vec2).to(DEVICE)
    similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
    return similarity.item()

class BERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

class SBERTFAQ:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class GPTChatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate_response(self, text):
        """Generates a response using GPT model."""
        print("Preprocessed Text:", preprocess_text(text))  # text is now a parameter
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class T5Summarizer:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=100):
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    sample_text = "I lost my credit card. What should I do?"

    print("Preprocessed Text:", preprocess_text(sample_text))
    print("BERT Tokenized:", tokenize_text(sample_text, "bert"))
    print("GPT Tokenized:", tokenize_text(sample_text, "gpt"))
    print("T5 Tokenized:", tokenize_text(sample_text, "t5"))
  
    embedding = get_sentence_embedding(sample_text)
    print("SBERT Embedding Shape:", embedding.shape)
