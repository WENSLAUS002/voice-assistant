import spacy
import json
import torch
from openai import OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# OpenAI Client Initialization (use env var in prod)
client = OpenAI(api_key="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")

# Load SBERT for FAQ matching
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Use zero-shot classification instead of BERT fine-tuning
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intent list for classification
intents = [
    "balance_inquiry",
    "loan_application",
    "interest_rate_query",
    "account_opening",
    "transaction_issue"
]

# 🔹 Rasa NLU Simulation
def rasa_nlu_process(user_input):
    response = {
        "intent": {"name": "balance_inquiry", "confidence": 0.95},
        "entities": [{"entity": "account_type", "value": "savings"}],
        "text": "You want to check your savings account balance."
    }
    return response

# 🔹 spaCy Processing
def spacy_nlu_process(user_input):
    doc = nlp(user_input)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"text": user_input, "entities": entities}

# 🔹 OpenAI GPT Processing (migrated to new SDK)
def gpt_nlu_process(user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return {"text": response.choices[0].message.content}

# 🔹 Zero-Shot BERT Intent Classification
def bert_nlu_process(user_input):
    result = zero_shot_classifier(user_input, candidate_labels=intents)
    intent = result["labels"][0]
    confidence = result["scores"][0]
    return {"intent": intent, "confidence": confidence}

# 🔹 SBERT Similarity-Based FAQ Matching
FAQ_DB = {
    "How can I check my account balance?": "You can check your balance using our mobile app or by calling customer support.",
    "What are the interest rates?": "Our current interest rates are 3.5% for savings and 5% for fixed deposits.",
    "How to apply for a loan?": "You can apply for a loan online via our portal or visit a branch."
}

def sbert_nlu_process(user_input):
    user_embedding = sbert_model.encode(user_input, convert_to_tensor=True)
    best_match = None
    highest_score = 0.0

    for question, answer in FAQ_DB.items():
        question_embedding = sbert_model.encode(question, convert_to_tensor=True)
        score = util.pytorch_cos_sim(user_embedding, question_embedding).item()
        if score > highest_score:
            highest_score = score
            best_match = answer

    return {"faq_match": best_match, "confidence": highest_score} if highest_score > 0.5 else {"faq_match": None}

# 🔹 Main NLU Function
def process_nlu(user_input, model="bert"):
    if model == "rasa":
        return rasa_nlu_process(user_input)
    elif model == "spacy":
        return spacy_nlu_process(user_input)
    elif model == "gpt":
        return gpt_nlu_process(user_input)
    elif model == "bert":
        return bert_nlu_process(user_input)
    elif model == "sbert":
        return sbert_nlu_process(user_input)
    else:
        return {"error": "Invalid model selected"}

# 🔹 Test NLU Outputs
if __name__ == "__main__":
    user_query = "How do I check my savings account balance?"
    print("🔹 Rasa Output:", json.dumps(process_nlu(user_query, "rasa"), indent=2))
    print("🔹 spaCy Output:", json.dumps(process_nlu(user_query, "spacy"), indent=2))
    print("🔹 GPT Output:", json.dumps(process_nlu(user_query, "gpt"), indent=2))
    print("🔹 BERT Output:", json.dumps(process_nlu(user_query, "bert"), indent=2))
    print("🔹 SBERT FAQ Output:", json.dumps(process_nlu(user_query, "sbert"), indent=2))
