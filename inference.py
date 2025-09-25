import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import pandas as pd
from sentence_transformers import util

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load models
print("Loading models...")
bert_tokenizer = BertTokenizer.from_pretrained("./models/bert_model")
bert_model = BertForSequenceClassification.from_pretrained("./models/bert_model").to(DEVICE)

gpt_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt_model")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2LMHeadModel.from_pretrained("./models/gpt_model").to(DEVICE)

#gpt_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt_model")
#gpt_model = GPT2LMHeadModel.from_pretrained("./models/gpt_model").to(DEVICE)

sbert_model = SentenceTransformer("./models/sbert_model")  # SBERT handles its own device

t5_tokenizer = T5Tokenizer.from_pretrained("./models/t5_model", legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained("./models/t5_model").to(DEVICE)

# Load GPT2 pipeline separately (optional)
gpt2_pipeline = pipeline("text-generation", model=gpt_model, tokenizer=gpt_tokenizer, device=0 if torch.cuda.is_available() else -1)

print("All models loaded.")

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["banking_db"]
logs_collection = mongo_db["model_outputs"]

# -------------------- CSV-Based FAQ Setup -------------------
# Load the CSV
csv_faq_df = pd.read_csv("faq_data.csv")

# Prepare model & precompute question embeddings
csv_model = SentenceTransformer("all-MiniLM-L6-v2")
csv_questions = csv_faq_df["question"].tolist()
csv_embeddings = csv_model.encode(csv_questions, convert_to_tensor=True)

def get_answer_from_csv(user_question):
    user_embedding = csv_model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, csv_embeddings)[0]
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_idx].item()
    best_question = csv_questions[best_idx]
    best_answer = csv_faq_df.iloc[best_idx]["answer"]
    return {
        "matched_question": best_question,
        "answer": best_answer,
        "score": round(best_score, 4)
    }

# -------------------- Functions --------------------

def classify_intent(text: str) -> int:
    inputs = bert_tokenizer(text, return_tensors="pt").to(DEVICE)
    outputs = bert_model(**inputs)
    label_id = torch.argmax(outputs.logits).item()
    return label_id

#def generate_response_gpt(text: str) -> str:
    #inputs = gpt_tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    #outputs = gpt_model.generate(inputs, max_length=60, num_return_sequences=1)
    #return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response_gpt(text: str) -> str:
    inputs = gpt_tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(DEVICE)
    outputs = gpt_model.generate(
        inputs,
        max_length=60,
        num_return_sequences=1,
        pad_token_id=gpt_tokenizer.eos_token_id
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text: str) -> str:
    inputs = t5_tokenizer("summarize: " + text, return_tensors="pt").to(DEVICE)
    summary_ids = t5_model.generate(inputs.input_ids, max_length=50)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def find_similar_faq(query: str) -> str:
    query_embedding = sbert_model.encode(query)
    faq_embeddings = sbert_model.encode(csv_questions)  # Use csv_questions from CSV data
    similarities = torch.cosine_similarity(
        torch.tensor([query_embedding]), torch.tensor(faq_embeddings)
    )
    best_match_index = similarities.argmax().item()
    return csv_questions[best_match_index]  # Use csv_questions here

def run_all_models(user_input: str) -> dict:
    # Get CSV-based FAQ response
    csv_answer = get_answer_from_csv(user_input)

    return {
        "bert_intent": classify_intent(user_input),
        "sbert_faq_match": find_similar_faq(user_input),
        "gpt2_response": generate_response_gpt(user_input),
        "t5_summary": summarize_text(user_input),
        "csv_faq_answer": csv_answer["answer"],
        "csv_faq_match": csv_answer["matched_question"],
        "csv_score": csv_answer["score"]
    }

def log_outputs_to_mongo(user_input: str, outputs: dict):
    logs_collection.insert_one({
        "input": user_input,
        "bert_output": outputs.get("bert_intent"),
        "sbert_output": outputs.get("sbert_faq_match"),
        "gpt2_output": outputs.get("gpt2_response"),
        "t5_output": outputs.get("t5_summary"),
        "csv_output": outputs.get("csv_faq_answer"),
        "csv_match": outputs.get("csv_faq_match"),
        "csv_score": outputs.get("csv_score")
    })

# Main test
if __name__ == "__main__":
    #test_input = "How do I set up a recurring payment for my bills?"
    test_input = "How do I check account balance?"
    print(f"\nUser Input: {test_input}\n")

    outputs = run_all_models(test_input)
    for model, result in outputs.items():
        print(f"[{model.upper()}]: {result}\n")

    print("Saving to MongoDB...")
    log_outputs_to_mongo(test_input, outputs)
    print("Done.")
