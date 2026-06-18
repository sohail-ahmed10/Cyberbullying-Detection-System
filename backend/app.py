import datetime
import os
import pickle
import re
from datetime import datetime

import certifi
import nltk
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ==============================================================================
# 🌐 MONGODB ATLAS CLOUD CONNECTION SETUP
# ==============================================================================
MONGO_URI = "mongodb+srv://cyberbullying:detection497@cluster0.sv9w8p2.mongodb.net/?appName=Cluster0"

try:
    # Using certifi.where() prevents cloud SSL validation handshake failures
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["Cyberbullying_Detection_System"]  # Database Name
    predictions_collection = db["predictions"]  # Collection (Table) Name
    print("✅ MongoDB Atlas Cloud Connected Successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

# ==============================================================================
# 📂 ML MODEL LOADING
# ==============================================================================
print("📂 Loading ML models...")
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
print("✅ Models loaded!")

# Setup NLTK Preprocessing
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# ==============================================================================
# 🧭 FRONTEND ROUTING FOR WEB APPLICATION
# ==============================================================================
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/style.css")
def css():
    return send_from_directory("../frontend", "style.css")


@app.route("/script.js")
def js():
    return send_from_directory("../frontend", "script.js")


# ==============================================================================
# 🔮 ML PREDICTION API ROUTE WITH DATABASE LOGGING
# ==============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocessing & Prediction
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        # Calculate Confidence Score
        if hasattr(model, "predict_proba"):
            conf = model.predict_proba(vec).max() * 100
        else:
            conf = 100.0

        prediction_result = str(pred)
        is_cyberbullying_bool = prediction_result != "not_cyberbullying"

        # 🗄️ LIVE MONGODB ATLAS INTEGRATION
        try:
            history_data = {
                "user_input": text,
                "cleaned_text": cleaned,
                "prediction": prediction_result,
                "is_threat": is_cyberbullying_bool,
                "confidence_score": round(float(conf), 2),
                "timestamp": datetime.utcnow(),
            }
            predictions_collection.insert_one(history_data)
            print("💾 Prediction successfully logged to MongoDB Cloud!")
        except Exception as db_err:
            print(f"⚠️ Warning: Failed to save log to Database: {db_err}")

        # Return JSON response to the Frontend UI
        return jsonify(
            {
                "is_cyberbullying": is_cyberbullying_bool,
                "confidence": round(float(conf), 2),
                "prediction": prediction_result,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    print("=" * 50)
    print("🛡️ Cyberbullying Detection System - Applied ML Evaluation Platform")
    print("📍 Local Web URL: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000)
    