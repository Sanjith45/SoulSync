from flask import Flask, request, render_template, jsonify, redirect, session, url_for
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv

# 🌱 Load environment variables
load_dotenv()

# 🚀 Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecretkey")

# 🌐 MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["soulsync"]
users = db["users"]

# 🔑 Groq API Setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 💡 Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🧠 Memory + FAISS
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = []
conversations = {}

# 📁 Storage Paths
INDEX_PATH = "faiss_index.index"
HISTORY_PATH = "conversations.json"
VECTOR_PATH = "vectors.npy"

def load_memory():
    global conversations, faiss_index, vector_store
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            conversations = json.load(f)
    if os.path.exists(INDEX_PATH) and os.path.exists(VECTOR_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        vectors = np.load(VECTOR_PATH, allow_pickle=True)
        vector_store.extend(vectors.tolist())

def save_memory():
    with open(HISTORY_PATH, "w") as f:
        json.dump(conversations, f)
    faiss.write_index(faiss_index, INDEX_PATH)
    np.save(VECTOR_PATH, np.array(vector_store, dtype=object))

# 🔁 On Server Start
load_memory()

# ===================== 🌐 ROUTES =====================

@app.route("/")
def home():
    return redirect(url_for("login"))

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
            "password": hashed
        })
        session["user_id"] = name
        return redirect(url_for("chat_page"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form["name"].strip()
        password = request.form["password"]

        user = users.find_one({"name": name})
        if user and check_password_hash(user["password"], password):
            session["user_id"] = name
            return redirect(url_for("chat_page"))
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

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"reply": "Session expired. Please log in again."})

    user_id = session["user_id"]
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please say something to begin."})

    # 🔁 Track conversation
    conversations.setdefault(user_id, [])
    conversations[user_id].append({"role": "user", "content": user_msg})

    # 🔍 Vector storage
    user_vec = embedder.encode(user_msg)
    faiss_index.add(np.array([user_vec]))
    vector_store.append((user_id, user_msg))

    # 🧠 Semantic Recall
    D, I = faiss_index.search(np.array([user_vec]), k=3)
    similar_msgs = [
        {"role": "user", "content": vector_store[idx][1]}
        for idx in I[0]
        if idx < len(vector_store) and vector_store[idx][0] == user_id
    ]

    # 💬 Prompt Construction
    prompt = [{"role": "system", "content": "You are SoulSync, a kind and empathetic mental health chatbot."}]
    prompt += similar_msgs[-2:] + conversations[user_id][-3:]

    # 🧠 Model Response
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=prompt
    )
    reply = response.choices[0].message.content.strip()
    conversations[user_id].append({"role": "assistant", "content": reply})

    save_memory()
    return jsonify({"reply": reply})

# ===================== ✅ =====================

if __name__ == "__main__":
    app.run(debug=True)
