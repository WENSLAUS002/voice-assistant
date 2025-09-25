from pymongo import MongoClient
import psycopg2
import datetime

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["nlp_system"]
mongo_logs = mongo_db["logs"]

# PostgreSQL Configuration
PG_CONN = psycopg2.connect(
    dbname="nlp_logs",
    user="your_user",
    password="wenslaus001",
    host="localhost",
    port="5432"
)
cursor = PG_CONN.cursor()

def init_db():
    """Creates the logs table in PostgreSQL if it does not exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            query TEXT,
            intent TEXT,
            confidence REAL,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    PG_CONN.commit()

def log_interaction(user_id, query, intent, confidence, response):
    """Logs each NLP interaction into both MongoDB and PostgreSQL."""
    timestamp = datetime.datetime.now().isoformat()
    
    # Log to MongoDB
    mongo_logs.insert_one({
        "user_id": user_id,
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "timestamp": timestamp
    })
    
    # Log to PostgreSQL
    cursor.execute('''
        INSERT INTO logs (user_id, query, intent, confidence, response, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (user_id, query, intent, confidence, response, timestamp))
    PG_CONN.commit()

def get_logs(limit=10):
    """Retrieves the latest interactions from both databases."""
    # Retrieve from MongoDB
    mongo_results = list(mongo_logs.find().sort("timestamp", -1).limit(limit))
    
    # Retrieve from PostgreSQL
    cursor.execute('''
        SELECT * FROM logs ORDER BY timestamp DESC LIMIT %s
    ''', (limit,))
    pg_results = cursor.fetchall()
    
    return {"mongodb_logs": mongo_results, "postgresql_logs": pg_results}

# usage
def main():
    init_db()
    log_interaction(123, "What is my balance?", "check_balance", 0.95, "$1,500 available.")
    logs = get_logs()
    print(logs)

if __name__ == "__main__":
    main()
