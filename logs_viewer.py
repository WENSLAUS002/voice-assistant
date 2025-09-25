from pymongo import MongoClient
from datetime import datetime
import pandas as pd

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["banking_db"]
logs_collection = db["logs"]

def view_all_logs():
    logs = list(logs_collection.find())
    for log in logs:
        print_log(log)

def filter_logs_by_model(model_name):
    logs = list(logs_collection.find({"model_name": model_name}))
    for log in logs:
        print_log(log)

def export_logs_to_csv(filename="mongo_logs_export.csv"):
    logs = list(logs_collection.find())
    if logs:
        df = pd.DataFrame(logs)
        df.drop(columns=["_id"], inplace=True, errors='ignore')
        df.to_csv(filename, index=False)
        print(f"Logs exported to {filename}")
    else:
        print("No logs to export.")

def print_log(log):
    print("---------------")
    print(f"User ID     : {log.get('user_id')}")
    print(f"Model       : {log.get('model_name')}")
    print(f"Timestamp   : {log.get('timestamp')}")
    print(f"Input Text  : {log.get('input_text')}")
    print(f"Response    : {log.get('response_data')}")
    print("---------------")

if __name__ == "__main__":
    while True:
        print("\n--- MongoDB Logs Viewer ---")
        print("1. View all logs")
        print("2. Filter logs by model name (e.g., BERT, GPT-2, SBERT, T5)")
        print("3. Export logs to CSV")
        print("4. Exit")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            view_all_logs()
        elif choice == "2":
            model = input("Enter model name to filter by: ").strip()
            filter_logs_by_model(model)
        elif choice == "3":
            filename = input("Enter filename for export (default: mongo_logs_export.csv): ").strip() or "mongo_logs_export.csv"
            export_logs_to_csv(filename)
        elif choice == "4":
            print("Exiting viewer.")
            break
        else:
            print("Invalid choice. Please select a valid option.")
