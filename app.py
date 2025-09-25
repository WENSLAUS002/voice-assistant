import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
from models.bert_classifier import BERTClassifier
from models.sbert_similarity import SBERTFAQ
from models.gpt_chatbot import GPTChatbot
from models.t5_summarizer import T5Summarizer
from pymongo import MongoClient
import mysql.connector
import whisper
from transformers import pipeline
import pandas as pd
from inference import run_all_models
from sentence_transformers import SentenceTransformer, util
from db import get_available_agent, log_escalation, log_to_mongo
import logging
# from waitress import serve

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AUDIO_FOLDER = "static/audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Initialize models with error handling
logger.info("Loading models...")
try:
    classifier = BERTClassifier()
    faq_system = SBERTFAQ()
    chatbot = GPTChatbot()
    summarizer = T5Summarizer()
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# MongoDB setup
try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client["banking_db"]
    faq_collection = mongo_db["Banking"]
    history_collection = mongo_db["chat_history"]
    logs_collection = mongo_db["model_outputs"]
    mongo_client.admin.command('ping')  # Test connection
    logger.info("Connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# MySQL setup
try:
    mysql_conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="wenslaus001",
        database="banking_db"
    )
    mysql_cursor = mysql_conn.cursor()
    logger.info("Connected to MySQL")
except Exception as e:
    logger.error(f"Failed to connect to MySQL: {str(e)}")
    raise

# Whisper ASR model
try:
    asr_model = whisper.load_model("base")
    logger.info("Whisper ASR model loaded")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    raise

# CSV-Based FAQ Setup
try:
    csv_faq_df = pd.read_csv("faq_data.csv")
    csv_model = SentenceTransformer("all-MiniLM-L6-v2")
    csv_questions = csv_faq_df["question"].tolist()
    csv_embeddings = csv_model.encode(csv_questions, convert_to_tensor=True)
    logger.info("CSV FAQ setup completed")
except Exception as e:
    logger.error(f"Failed to set up CSV FAQ: {str(e)}")
    raise

def get_answer_from_csv(user_question):
    try:
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
    except Exception as e:
        logger.error(f"Error in get_answer_from_csv: {str(e)}")
        return {"matched_question": "", "answer": "Error processing FAQ", "score": 0.0}

# GPT-2 pipeline
try:
    nlp_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1,
        max_length=200,
        truncation=True
    )
    logger.info("GPT-2 pipeline initialized")
except Exception as e:
    logger.error(f"Failed to initialize GPT-2 pipeline: {str(e)}")
    raise

def run_all_models(user_input):
    try:
        outputs = {
            "bert": classifier.predict(user_input),
            "gpt2": nlp_pipeline(user_input, max_length=200)[0]["generated_text"],
            "t5": summarizer.summarize(user_input),
            "sbert": faq_system.get_best_match(user_input)
        }
        # Ensure sbert output is a dictionary
        if isinstance(outputs["sbert"], str):
            logger.warning(f"SBERT output is a string, converting to dict: {outputs['sbert']}")
            outputs["sbert"] = {"answer": outputs["sbert"], "score": 0.0}
        return outputs
    except Exception as e:
        logger.error(f"Error in run_all_models: {str(e)}")
        raise

def log_to_mongo(user_input, outputs):
    try:
        logs_collection.insert_one({
            "input": user_input,
            "bert_output": outputs["bert"],
            "gpt2_output": outputs["gpt2"],
            "t5_output": outputs["t5"],
            "sbert_output": outputs["sbert"]
        })
        logger.info("Logged to MongoDB")
    except Exception as e:
        logger.error(f"Error logging to MongoDB: {str(e)}")

@app.route("/health", methods=["GET"])
def health():
    try:
        mongo_client.admin.command('ping')
        mysql_cursor.execute("SELECT 1")
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/stt", methods=["POST"])
def speech_to_text():
    logger.info("Received STT request")
    try:
        if "audio" not in request.files:
            logger.error("No audio file uploaded")
            return jsonify({"error": "No audio file uploaded"}), 400
        audio_file = request.files["audio"]
        audio_file_path = os.path.join(AUDIO_FOLDER, "input_audio.wav")
        audio_file.save(audio_file_path)
        audio = AudioSegment.from_file(audio_file_path)
        audio.export(audio_file_path, format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        user_text = recognizer.recognize_google(audio_data)
        logger.info(f"Transcribed text: {user_text}")
        return jsonify({"text": user_text})
    except sr.UnknownValueError:
        logger.error("Speech not recognized")
        return jsonify({"error": "Speech not recognized"}), 400
    except sr.RequestError:
        logger.error("STT service unavailable")
        return jsonify({"error": "STT service unavailable"}), 500
    except Exception as e:
        logger.error(f"STT failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"STT failed: {str(e)}"}), 500

@app.route("/faq_csv", methods=["POST"])
def faq_csv():
    logger.info("Received FAQ CSV request")
    try:
        text = request.json.get("text")
        if not text:
            logger.error("No question provided")
            return jsonify({"error": "No question provided"}), 400
        result = get_answer_from_csv(text)
        tts = gTTS(text=result["answer"], lang="en")
        audio_file_path = os.path.join(AUDIO_FOLDER, "faq_csv_response.mp3")
        tts.save(audio_file_path)
        logger.info(f"FAQ CSV response: {result}")
        return jsonify({
            "matched_question": result["matched_question"],
            "answer": result["answer"],
            "score": result["score"],
            "audio_file": "/static/audio/faq_csv_response.mp3"
        })
    except Exception as e:
        logger.error(f"FAQ CSV failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"FAQ CSV failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    logger.info("Received chat request")
    try:
        data = request.json
        user_text = data.get("text", "")
        bot_response = chatbot.generate_response(user_text)
        tts = gTTS(text=bot_response, lang="en")
        audio_file_path = os.path.join(AUDIO_FOLDER, "response.mp3")
        tts.save(audio_file_path)
        logger.info(f"Chat response: {bot_response}")
        return jsonify({"response_text": bot_response, "audio_file": "/static/audio/response.mp3"})
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

@app.route("/classify", methods=["POST"])
def classify():
    logger.info("Received classify request")
    try:
        text = request.json.get("text")
        if not text:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        category = classifier.predict(text)
        logger.info(f"Classification: {category}")
        return jsonify({"category": category})
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

@app.route("/faq", methods=["POST"])
def faq():
    logger.info("Received FAQ request")
    try:
        text = request.json.get("text")
        if not text:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        mongo_result = faq_collection.find_one({"question": text})
        if mongo_result:
            answer = mongo_result.get("answer", "Sorry, answer not found in MongoDB.")
        else:
            closest_match = faq_system.get_best_match(text)
            # Handle both string and dict return types from get_best_match
            if isinstance(closest_match, str):
                answer = closest_match if closest_match else "Sorry, I don't have an answer for that."
            else:
                answer = closest_match.get("answer", "Sorry, I don't have an answer for that.")
        tts = gTTS(text=answer, lang="en")
        audio_file_path = os.path.join(AUDIO_FOLDER, "faq_response.mp3")
        tts.save(audio_file_path)
        logger.info(f"FAQ response: {answer}")
        return jsonify({"answer": answer, "audio_file": "/static/audio/faq_response.mp3"})
    except Exception as e:
        logger.error(f"FAQ failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"FAQ failed: {str(e)}"}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    logger.info("Received summarize request")
    try:
        text = request.json.get("text")
        if not text:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        summary = summarizer.summarize(text)
        logger.info(f"Summary: {summary}")
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route("/static/audio/<filename>")
def get_audio(filename):
    logger.info(f"Serving audio file: {filename}")
    try:
        return send_file(os.path.join(AUDIO_FOLDER, filename), mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Failed to serve audio: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to serve audio: {str(e)}"}), 500

@app.route("/process", methods=["POST"])
def process_text():
    logger.info("Received process text request")
    try:
        user_input = request.json.get("text", "")
        outputs = run_all_models(user_input)
        log_to_mongo(user_input, outputs)
        return jsonify(outputs)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/tts", methods=["POST"])
def text_to_speech():
    logger.info("Received TTS request")
    try:
        text = request.json.get("text")
        if not text:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        tts = gTTS(text=text, lang="en")
        audio_file_path = os.path.join(AUDIO_FOLDER, "tts_response.mp3")
        tts.save(audio_file_path)
        logger.info("TTS audio saved")
        return jsonify({"status": "saved", "audio_file": "/static/audio/tts_response.mp3"})
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received predict request")
    try:
        data = request.get_json()
        user_input = data.get("text")
        logger.info(f"Input: {user_input}")
        if not user_input:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        logger.info("Running models...")
        outputs = run_all_models(user_input)
        logger.info(f"Model outputs: {outputs}")
        logger.info("Logging to MongoDB...")
        log_to_mongo(user_input, outputs)
        # Generate TTS using the SBERT output
        tts_text = outputs["sbert"]["answer"]
        tts = gTTS(text=tts_text, lang="en")
        audio_file_path = os.path.join(AUDIO_FOLDER, "predict_response.mp3")
        tts.save(audio_file_path)
        logger.info("Returning response")
        # Return SBERT output as the response text
        return jsonify({
            "status": "success",
            "input": user_input,
            "response_text": outputs["sbert"]["answer"],
            "audio_file": "/static/audio/predict_response.mp3"
        })
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    logger.info("Received predict CSV request")
    try:
        file = request.files.get("file")
        if not file:
            logger.error("No file provided")
            return jsonify({"error": "No file provided"}), 400
        df = pd.read_csv(file)
        if "question" not in df.columns:
            logger.error("CSV must contain 'question' column")
            return jsonify({"error": "CSV must contain 'question' column"}), 400
        all_outputs = []
        for _, row in df.iterrows():
            question = row["question"]
            outputs = run_all_models(question)
            log_to_mongo(question, outputs)
            all_outputs.append({"input": question, **outputs})
        logger.info(f"CSV predictions completed: {len(all_outputs)} questions")
        return jsonify(all_outputs)
    except Exception as e:
        logger.error(f"Predict CSV failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Predict CSV failed: {str(e)}"}), 500

@app.route("/escalate", methods=["POST"])
def escalate():
    logger.info("Received escalate request")
    try:
        data = request.get_json()
        user_input = data.get("text")
        user_id = data.get("user_id", 1)
        logger.info(f"Input: {user_input}, User ID: {user_id}")
        if not user_input:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        outputs = run_all_models(user_input)
        log_to_mongo(user_input, outputs)
        sbert_output = outputs.get("sbert", {})
        sbert_answer = sbert_output.get("answer", "") if isinstance(sbert_output, dict) else str(sbert_output)
        sbert_score = sbert_output.get("score", 0.0) if isinstance(sbert_output, dict) else 0.0
        should_escalate = (
            "sorry" in outputs.get("bert", "").lower() or
            "sorry" in outputs.get("gpt2", "").lower() or
            sbert_score < 0.5 or
            "sorry" in sbert_answer.lower()
        )
        if should_escalate:
            agent = get_available_agent()
            if agent:
                agent_id = agent["id"]
                log_escalation(user_id, user_input, agent_id)
                message = (
                    f"Your query has been escalated to Agent {agent['name']} (ID: {agent_id}). "
                    "They will contact you shortly."
                )
            else:
                message = "All agents are currently busy. Please try again later."
            logger.info(f"Escalation triggered: {message}")
            return jsonify({
                "escalated": True,
                "message": message,
                "outputs": outputs
            })
        logger.info("No escalation needed")
        return jsonify({
            "escalated": False,
            "outputs": outputs
        })
    except Exception as e:
        logger.error(f"Escalation failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Escalation failed: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception: %s", e)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)