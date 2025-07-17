from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import uuid
from datetime import datetime
import nlp_model # Import your NLP logic

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="An intelligent chatbot for customer support/FAQs with contextual responses and logging."
)

# SQLite Database Setup
DATABASE_FILE = "chat_logs.db"

def init_db():
    """Initializes the SQLite database and creates the chat_logs table."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()

# Pydantic models for request and response data validation
class ChatRequest(BaseModel):
    message: str
    session_id: str = None # Optional: if not provided, a new one will be generated

class ChatResponse(BaseModel):
    response: str
    session_id: str
    # You could add other fields like 'intent_detected', 'confidence' for analytics

# Event handler for application startup
@app.on_event("startup")
async def startup_event():
    init_db()
    nlp_model.initialize_nlp_model() # Load your NLP model when the app starts

# API Endpoint for Chat
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4()) # Generate a new session ID if none provided

    user_message = request.message

    # --- Contextual Response Logic (simple example) ---
    # For a more advanced chatbot, you'd fetch previous messages from `chat_logs`
    # for the `session_id` and pass them to your NLP model to influence the response.
    # For now, `nlp_model.get_chatbot_response` can use session_id if it needs to
    # query the DB itself. Let's keep it simple for this guide.
    # Example:
    # try:
    #     conn = sqlite3.connect(DATABASE_FILE)
    #     cursor = conn.cursor()
    #     cursor.execute(
    #         "SELECT user_message, bot_response FROM chat_logs WHERE session_id = ? ORDER BY timestamp DESC LIMIT 5",
    #         (session_id,)
    #     )
    #     past_interactions = cursor.fetchall()
    #     conn.close()
    # except sqlite3.Error as e:
    #     print(f"Error fetching past interactions: {e}")
    #     past_interactions = []
    #
    # # Pass past_interactions to your NLP model if it can use them
    # bot_response = nlp_model.get_chatbot_response(user_message, past_interactions=past_interactions)

    bot_response = nlp_model.get_chatbot_response(user_message, session_id=session_id)

    # Log the interaction
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_logs (session_id, user_message, bot_response, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, user_message, bot_response, datetime.now().isoformat())
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during logging: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during logging.")
    finally:
        if conn:
            conn.close()

    return {"response": bot_response, "session_id": session_id}

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot API is running."}