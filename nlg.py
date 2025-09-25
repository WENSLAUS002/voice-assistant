import os
from dotenv import load_dotenv
import openai
from transformers import pipeline
import spacy
from pymongo import MongoClient
import mysql.connector
import logging
import pandas as pd
from rasa.core.agent import Agent
import asyncio

# Load environment variables
load_dotenv()

# OpenAI API Setup (new syntax)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"
def generate_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"OpenAI Error: {e}")
        return "Error generating response from GPT."

# BERT/GPT2 Pipeline (text generation)
bert_pipeline = pipeline("text-generation", model="gpt2")  # Use GPT-2 for generation
def generate_bert_response(text):
    return bert_pipeline(text, max_length=50, num_return_sequences=1, truncation=True)[0]['generated_text']

# SpaCy Lemmatizer
nlp_spacy = spacy.load("en_core_web_sm")
def process_text_spacy(text):
    doc = nlp_spacy(text)
    return " ".join([token.lemma_ for token in doc])

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["banking_db"]
nlg_collection = mongo_db.get_collection("nlg")

# MySQL Setup
try:
    mysql_conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB", "banking_db")
    )
    mysql_cursor = mysql_conn.cursor()
    print("Connected to MySQL")
except Exception as e:
    logging.error(f"MySQL Connection Error: {e}")

# Load dataset
df = pd.read_csv("banking_dataset.csv")
def get_faq_response(question):
    response = df[df['question'] == question]['answer'].values
    return response[0] if len(response) > 0 else "I'm sorry, I don't have an answer."

# Async Rasa Response
async def rasa_response(message):
    try:
        agent = Agent.load("models")
        responses = await agent.handle_text(message)
        return responses[0]["text"] if responses else "No response from Rasa."
    except Exception as e:
        logging.error(f"Rasa error: {e}")
        return "Error generating response from Rasa."

# Save to Mongo
def save_response_to_mongo(user_input, generated_response):
    try:
        if nlg_collection is not None:
            nlg_collection.insert_one({
                "user_input": user_input,
                "generated_response": generated_response
            })
            print("Response saved to MongoDB")
    except Exception as e:
        logging.error(f"Error saving to MongoDB: {e}")

# Run if main
if __name__ == "__main__":
    user_query = "How can I check my account balance?"

    print("\n--- Responses ---")
    print("GPT-3.5 Response:", generate_gpt_response(user_query))
    print("GPT-2 Response:", generate_bert_response(user_query))
    print("SpaCy Lemmas:", process_text_spacy(user_query))
    print("FAQ Response:", get_faq_response(user_query))

    # Save GPT response to MongoDB
    response = generate_gpt_response(user_query)
    save_response_to_mongo(user_query, response)

    # Run Rasa async
    print("Rasa Response:", asyncio.run(rasa_response(user_query)))

    # Close connections
    if mysql_cursor: mysql_cursor.close()
    if mysql_conn: mysql_conn.close()
    if mongo_client: mongo_client.close()
