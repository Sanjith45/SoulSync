from flask import Flask, request, render_template, jsonify, redirect, session, url_for
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import requests


# 🌱 Load environment variables
load_dotenv()

# 🚀 Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecretkey")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
# 🌐 MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["soulsync"]
users = db["users"]
chat_history = db["conversations"]

# 🔑 Groq API Setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 💡 Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🧠 FAISS Setup
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = []  # Format: (user_id, message)


# ✅ Load user’s past data into FAISS and memory
def preload_user_data(user_id):
    existing = chat_history.find_one({"user_id": user_id})
    if not existing:
        chat_history.insert_one({"user_id": user_id, "messages": []})
        return []

    messages = existing.get("messages", [])
    for msg in messages:
        if msg["role"] == "user":
            vec = embedder.encode(msg["content"])
            faiss_index.add(np.array([vec]))
            vector_store.append((user_id, msg["content"]))
    return messages


# ✅ Save conversation to MongoDB
def save_to_db(user_id, messages):
    chat_history.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages}},
        upsert=True
    )


# ===================== 🌐 ROUTES =====================

@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_TOKEN")

@app.route("/cool")
def cool():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("cool.html")  # Just loads the page with geolocation script


@app.route("/find_events", methods=["POST"])
def find_events():
    if "user_id" not in session:
        return jsonify([])

    if not EVENTBRITE_TOKEN:
        return jsonify([])

    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    url = (
        "https://www.eventbriteapi.com/v3/events/search/"
        f"?location.latitude={lat}"
        f"&location.longitude={lon}"
        "&location.within=15km"
        "&categories=107"  # Health & Wellness
    )

    headers = {"Authorization": f"Bearer {EVENTBRITE_TOKEN}"}
    response = requests.get(url, headers=headers)
    events_data = response.json()

    events = []
    for event in events_data.get("events", []):
        events.append({
            "name": event["name"]["text"],
            "start": event["start"]["local"],
            "url": event["url"]
        })

    return jsonify(events)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"].strip()
        dob = request.form["dob"]
        nickname = request.form["nickname"].strip()
        password = request.form["password"]

        if users.find_one({"name": name}):
            return "🛑 User already exists. Try logging in."

        hashed = generate_password_hash(password)
        users.insert_one({
            "name": name,
            "dob": dob,
            "nickname": nickname,
            "password": hashed,
            "interests": {}
        })
        session["user_id"] = name
        preload_user_data(name)
        return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form["name"].strip()
        password = request.form["password"]

        user = users.find_one({"name": name})
        if user and check_password_hash(user["password"], password):
            session["user_id"] = name
            preload_user_data(name)
            return redirect(url_for("home"))
        return "❌ Invalid credentials."
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat")
def chat_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html")


from transformers import pipeline

# ✅ Load sentiment analyzer once at startup (put this near other imports / globals)
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"reply": "Session expired. Please log in again."})

    user_id = session["user_id"]
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please say something to begin."})

    # 🧠 Run sentiment analysis on user message
    sentiment_result = sentiment_analyzer(user_msg)[0]
    sentiment_label = sentiment_result["label"]  # "POSITIVE" or "NEGATIVE"
    sentiment_score = sentiment_result["score"]

    # 🧠 Fetch user and previous messages
    user_data = users.find_one({"name": user_id})
    nickname = user_data.get("nickname", "friend")

    existing = chat_history.find_one({"user_id": user_id}) or {"messages": []}
    messages = existing["messages"]

    # 👋 Greet on first message
    if len(messages) == 0:
        greeting = f"Hey {nickname}! 😊 It's nice to meet you. How are you feeling today?"
        messages.append({"role": "assistant", "content": greeting})
        save_to_db(user_id, messages)
        return jsonify({"reply": greeting, "sentiment": sentiment_label})

    # 🧠 Store user message with sentiment
    messages.append({
        "role": "user",
        "content": user_msg,
        "sentiment": sentiment_label,
        "confidence": round(sentiment_score, 2)
    })
    user_vec = embedder.encode(user_msg)
    faiss_index.add(np.array([user_vec]))
    vector_store.append((user_id, user_msg))

    # 🔍 Retrieve relevant past messages via FAISS
    D, I = faiss_index.search(np.array([user_vec]), k=3)
    similar_msgs = [
        {"role": "user", "content": vector_store[idx][1]}
        for idx in I[0]
        if idx < len(vector_store) and vector_store[idx][0] == user_id
    ]

    # 🔁 Construct short memory summary (2 similar + last 3 messages)
    past_context = similar_msgs[-2:] + messages[-3:]

    # 🎯 Custom system prompt
    system_prompt = (
        f"You are SoulSync, an empathetic mental health chatbot. "
        f"You've previously spoken with a user named {nickname}. "
        f"Use a warm tone, be caring, and remember their emotional cues."
    )

    # 🧹 Clean messages (remove unsupported fields like 'sentiment')
    def clean_msgs(msgs):
        return [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
            if "role" in m and "content" in m
        ]

    prompt = [{"role": "system", "content": system_prompt}] + clean_msgs(past_context)

    # 🧠 Generate response
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=prompt
    )

    reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    save_to_db(user_id, messages)

    # ✅ Return both reply and sentiment
    return jsonify({
        "reply": reply,
        "sentiment": sentiment_label,
        "confidence": round(sentiment_score, 2)
    })



@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user = users.find_one({"name": user_id})

    if request.method == "POST":
        music = request.form.get("music_interest", "")
        sports = request.form.get("sports_interest", "")
        hobbies = request.form.get("hobby_interest", "")
        users.update_one(
            {"name": user_id},
            {"$set": {"interests": {
                "music": music,
                "sports": sports,
                "hobbies": hobbies
            }}}
        )
        user = users.find_one({"name": user_id})  # Refresh

    return render_template("dashboard.html", user=user)

# ===================== ✅ =====================

if __name__ == "__main__":
    app.run(debug=True)
