from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["recoverflow_db"]
sessions = db["sessions"]

# Clear old test data and insert a clean state
sessions.delete_many({})
sessions.insert_one({
    "invoice_id": "INV-0016",
    "status": "negotiating",
    "trust_score": 82,
    "financial_bounds": {
        "principal": 134000,
        "current_floor": 40200,
        "max_allowed_date": "2026-09-30"
    },
    "state_locks": {
        "first_counter_issued": False,
        "reason_collected": False
    },
    "chat_history": []
})

print("Database seeded. Ready for testing.")
