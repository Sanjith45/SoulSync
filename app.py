from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# 🔑 Groq API setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 💡 Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 📦 Paths
INDEX_PATH = "faiss_index.index"
HISTORY_PATH = "conversations.json"
VECTOR_PATH = "vectors.npy"

# 📏 Embedding dimension
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = []  # List of (user_id, message)
conversations = {}

# 🔄 Load from disk
def load_memory():
    global conversations, faiss_index, vector_store
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            conversations = json.load(f)
    if os.path.exists(INDEX_PATH) and os.path.exists(VECTOR_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        vector_store_data = np.load(VECTOR_PATH, allow_pickle=True)
        vector_store.extend(vector_store_data.tolist())

# 💾 Save to disk
def save_memory():
    with open(HISTORY_PATH, "w") as f:
        json.dump(conversations, f)
    faiss.write_index(faiss_index, INDEX_PATH)
    np.save(VECTOR_PATH, np.array(vector_store, dtype=object))

# ✅ Load on startup
load_memory()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    user_id = "user_001"  # Replace with session/email for real app

    # 1️⃣ Track conversation
    if user_id not in conversations:
        conversations[user_id] = []
    conversations[user_id].append({"role": "user", "content": user_msg})

    # 2️⃣ Embed & store
    user_vec = embedder.encode(user_msg)
    faiss_index.add(np.array([user_vec]))
    vector_store.append((user_id, user_msg))

    # 3️⃣ Semantic retrieval
    D, I = faiss_index.search(np.array([user_vec]), k=3)
    similar_msgs = []
    for idx in I[0]:
        if idx < len(vector_store) and vector_store[idx][0] == user_id:
            similar_msgs.append({"role": "user", "content": vector_store[idx][1]})

    # 4️⃣ Construct prompt
    prompt = [{"role": "system", "content": "You are SoulSync, a kind and empathetic mental health chatbot."}]
    prompt.extend(similar_msgs[-2:])                     # most relevant emotional messages
    prompt.extend(conversations[user_id][-3:])           # latest few exchanges

    # 5️⃣ Model response
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=prompt
    )
    bot_reply = response.choices[0].message.content.strip()
    conversations[user_id].append({"role": "assistant", "content": bot_reply})

    # 6️⃣ Save all
    save_memory()

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
