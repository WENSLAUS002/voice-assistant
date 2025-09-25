import json
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from rasa.shared.nlu.interpreter import Interpreter  # ✅ Correct import for Rasa 3.6+
from db import fetch_banking_info
import openai

# Load NLP Models
spacy_nlp = spacy.load("en_core_web_sm")
gpt_model = "gpt-3.5-turbo"  # OpenAI GPT model
bert_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # ✅ Correct for intent classification
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
t5_model = pipeline("text2text-generation", model="t5-small")
rasa_interpreter = Interpreter.load("./models")  # ✅ Adjust path as needed

# OpenAI API Key
openai.api_key = "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"  # ⚠️ Use environment variable in production!

def detect_intent(user_query):
    """Detects intent using multiple NLP models and selects the most confident result."""
    intents = ["check_balance", "view_transactions", "transfer_money", "report_fraud", "general_inquiry"]
    
    # Using BERT
    bert_result = bert_model(user_query, candidate_labels=intents)
    bert_intent = bert_result["labels"][0]
    bert_confidence = bert_result["scores"][0]
    
    # Using SBERT
    intent_embeddings = sbert_model.encode(intents, convert_to_tensor=True)
    query_embedding = sbert_model.encode(user_query, convert_to_tensor=True)
    sbert_scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)
    sbert_intent = intents[sbert_scores.argmax().item()]
    sbert_confidence = sbert_scores.max().item()
    
    # Using Rasa
    rasa_result = rasa_interpreter.parse(user_query)
    rasa_intent = rasa_result["intent"]["name"]
    rasa_confidence = rasa_result["intent"]["confidence"]
    
    # Selecting best intent based on confidence scores
    results = [(bert_intent, bert_confidence), (sbert_intent, sbert_confidence), (rasa_intent, rasa_confidence)]
    best_intent, best_confidence = max(results, key=lambda x: x[1])
    
    return best_intent, best_confidence

def generate_response(user_id, intent, query):
    """Generates response based on detected intent using GPT, T5, and predefined rules."""
    responses = {
        "check_balance": lambda user_id: f"Your account balance is ${fetch_banking_info(user_id, 'balance')}.",
        "view_transactions": lambda user_id: f"Your last transaction was ${fetch_banking_info(user_id, 'last_transaction')}.",
        "transfer_money": lambda _: "Please specify the amount and recipient.",
        "report_fraud": lambda _: "Your fraud report request has been noted. A representative will contact you shortly.",
        "general_inquiry": lambda _: "How can I assist you with your banking needs?"
    }
    
    # Use predefined responses if intent is known
    if intent in responses:
        return responses[intent](user_id)
    
    # If no predefined response, use GPT or T5 to generate one
    try:
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[{"role": "user", "content": query}]
        )
        return gpt_response["choices"][0]["message"]["content"]
    except Exception:
        t5_response = t5_model(query)[0]["generated_text"]
        return t5_response

def process_nlp(user_id, query):
    """Processes user input and returns the appropriate response."""
    intent, confidence = detect_intent(query)
    response = generate_response(user_id, intent, query)
    
    return {
        "intent": intent,
        "confidence": confidence,
        "response": response
    }

def main():
    user_id = 123
    query = "What is my account balance?"
    result = process_nlp(user_id, query)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
