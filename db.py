from pymongo import MongoClient
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import logging

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["banking_db"]
faq_collection = mongo_db["Banking"]
logs_collection = mongo_db["logs"]  # MongoDB logs collection for interactions

# MySQL Connection
MYSQL_DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "wenslaus001",
    "database": "banking_db"
}
try:
    mysql_conn = mysql.connector.connect(**MYSQL_DB_CONFIG)
    mysql_cursor = mysql_conn.cursor(dictionary=True)
    print("Connected to MySQL")
except Error as e:
    logging.error(f"Error connecting to MySQL: {e}")

# Function to get FAQ response from MongoDB
def get_faq_response(question):
    response = faq_collection.find_one({"question": question})
    return response["answer"] if response else "Sorry, I don't have an answer for that."

# Function to fetch customer details from MySQL
def get_customer_info(customer_id):
    try:
        query = "SELECT * FROM customers WHERE id = %s"
        mysql_cursor.execute(query, (customer_id,))
        customer_data = mysql_cursor.fetchone()
        return customer_data if customer_data else "Customer not found."
    except Error as e:
        logging.error(f"Error fetching customer info: {e}")
        return "Error retrieving customer details."

# Function to get an available agent from MySQL
def get_available_agent():
    try:
        query = "SELECT * FROM agents WHERE status = 'available' LIMIT 1"
        mysql_cursor.execute(query)
        agent = mysql_cursor.fetchone()
        return agent
    except Error as e:
        logging.error(f"Error fetching available agent: {e}")
        return None

# Function to log an escalation in MySQL
def log_escalation(user_id, query_text, agent_id):
    try:
        query = "INSERT INTO escalations (user_id, query, agent_id) VALUES (%s, %s, %s)"
        mysql_cursor.execute(query, (user_id, query_text, agent_id))
        mysql_conn.commit()
    except Error as e:
        logging.error(f"Error logging escalation: {e}")

# ✅ Function to log interaction details to MongoDB with model_name and timestamp
def log_to_mongo(user_id, input_text, response_data, model_name):
    try:
        log_entry = {
            "user_id": user_id,
            "input_text": input_text,
            "response_data": response_data,
            "model_name": model_name,
            "timestamp": datetime.utcnow()
        }
        logs_collection.insert_one(log_entry)
    except Exception as e:
        logging.error(f"Error logging to MongoDB: {e}")

# Function to process queries using NLP (BERT/GPT)
def process_nlp_query(query, nlp_model):
    response = nlp_model.generate_response(query)
    return response

# Close connections on exit
def close_connections():
    mysql_cursor.close()
    mysql_conn.close()
    mongo_client.close()
    print("Database connections closed.")

if __name__ == "__main__":
    print(get_faq_response("What are the bank's working hours?"))
    print(get_customer_info(1))
    close_connections()
