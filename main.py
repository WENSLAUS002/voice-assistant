import logging
from stt import speech_to_text
from nlu import process_nlu
from nlg import generate_response
from tts import text_to_speech
from db import MongoDBHandler, MySQLDBHandler  # ✅ Updated import
from t5_nlg import T5NLG
from config import Config

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Starting NLP Banking Assistant...")
    
    # Initializing databases
    mongo_handler = MongoDBHandler()
    mysql_handler = MySQLDBHandler()  # ✅ Replaced Postgres with MySQL
    
    # Capture speech input
    user_input = speech_to_text()
    if not user_input:
        print("No input detected. Exiting...")
        return
    
    # Process input using NLU (BERT, Rasa, Spacy)
    intent, entities = process_nlu(user_input)
    logging.info(f"Recognized Intent: {intent}, Entities: {entities}")
    
    # Fetch related data from databases
    faq_answer = mongo_handler.get_answer(user_input)
    structured_data = mysql_handler.get_customer_data(entities)  # ✅ Updated handler
    
    # Generate response using GPT/NLG
    response = generate_response(intent, faq_answer, structured_data)
    logging.info(f"Generated Response: {response}")
    
    # Convert response to speech
    text_to_speech(response)
    
    print("Response provided successfully.")
    
    # Additional T5 Model Processing
    t5_nlg = T5NLG(model_name="t5-small")
    input_text = "The banking system ensures secure transactions and customer support."
    summary = t5_nlg.generate_text("summarize:", input_text)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
