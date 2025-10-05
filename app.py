from flask import Flask, request, render_template, jsonify, redirect, session, url_for
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId
import requests
from datetime import datetime
import base64
import json


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
# New collections for modules and preferences
preferences = db["preferences"]
modules = db["modules"]  # stores generated/saved module instances
module_templates = db["module_templates"]  # default templates for instant fallback
rewards = db["rewards"]  # user rewards (badges, wallpapers)

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


# ---------- Utility: Ensure instant templates exist ----------
def ensure_default_templates():
    # Create minimal instant templates if collection is empty
    if module_templates.count_documents({}) == 0:
        module_templates.insert_many([
            {
                "type": "music",
                "title": "Calm Waves Loop",
                "tags": ["calm", "sleep", "focus"],
                "audio_url": "/static/instant/calm-loop.mp3"
            },
            {
                "type": "meditation",
                "title": "60s Box Breathing",
                "script": "Inhale 4... Hold 4... Exhale 4... Hold 4... Repeat gently.",
                "length": 60
            },
            {
                "type": "poem",
                "title": "Instant Haiku",
                "text": "Gentle breath arrives\nWorries drift like autumn leaves\nMorning light returns"
            },
            {
                "type": "art",
                "title": "Coloring Canvas",
                "prompt": "Pick a color and draw your calm."
            }
        ])

app_has_run = False
@app.before_request
def bootstrap_defaults():
    global app_has_run
    if not app_has_run:    
        app_has_run = True

    ensure_default_templates()


# ---------- Image Generation ----------
def generate_image(prompt, style="artistic"):
    """Generate an image using a free image generation service"""
    try:
        # Using Unsplash API for placeholder images (free tier)
        # In a real implementation, you'd use DALL-E, Midjourney, or Stable Diffusion
        unsplash_url = "https://api.unsplash.com/photos/random"
        params = {
            "query": prompt.replace(" ", "+"),
            "orientation": "landscape",
            "count": 1
        }
        headers = {
            "Authorization": f"Client-ID {os.getenv('UNSPLASH_ACCESS_KEY', 'demo')}"
        }
        
        response = requests.get(unsplash_url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]["urls"]["regular"]
            elif isinstance(data, dict) and "urls" in data:
                return data["urls"]["regular"]
        
        # Fallback to a placeholder service
        return f"https://picsum.photos/400/300?random={hash(prompt) % 1000}"
        
    except Exception as e:
        print(f"Image generation error: {e}")
        # Return a beautiful gradient placeholder
        return f"https://via.placeholder.com/400x300/6366f1/ffffff?text={prompt.replace(' ', '+')}"

# ---------- Topic Module Generator ----------
def create_topic_module(user_id: str, topic: str):
    topic_lower = (topic or "").lower()
    import random

    # Generate AI-powered content using Groq
    def generate_ai_content(content_type, topic, context=""):
        try:
            prompt = f"Create {content_type} about '{topic}'. {context} Make it engaging, educational, and emotionally supportive. Keep it concise but meaningful."
            response = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Let's explore {topic} together!"

    def create_comprehensive_quiz(topic):
        try:
            questions_prompt = f"""
    You are a creative and knowledgeable quiz designer. 
    Generate 12 unique, high-quality quiz questions about the topic: **'{topic}'**.

    🎯 Goals:
    - Each question must feel fresh, engaging, and educational.
    - Avoid generic or templated phrasing.
    - Include:
    • 3 factual knowledge questions  
    • 3 reasoning/conceptual  
    • 2 real-world/application  
    • 2 fun trivia/pop-culture  
    • 2 analogy/riddle-style
    - Each question must **teach something new**.

    📚 STRICT FORMAT:
    Q: [Question about '{topic}']
    A: [Option 1]
    B: [Option 2]
    C: [Option 3]
    D: [Option 4]
    Correct: [A/B/C/D]
    Explanation: [Short, factual explanation]

    ⚠️ Rules:
    - “Correct” must be a single letter: A, B, C, or D.
    - Explanations must be concise but meaningful.
    """

            response = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": questions_prompt}],
                max_tokens=2000,
            )

            quiz_text = response.choices[0].message.content.strip()

            import re
            pattern = r"Q:(.*?)A:(.*?)B:(.*?)C:(.*?)D:(.*?)Correct:(.*?)Explanation:(.*?)(?=\nQ:|$)"
            matches = re.findall(pattern, quiz_text, re.S)

            questions = []
            for match in matches:
                q, a, b, c, d, correct, explanation = [m.strip() for m in match]

                # 🧹 Clean up the 'Correct' field safely
                correct_letter = (
                    correct.strip().upper().replace(")", "").replace(".", "").replace("OPTION", "").strip()
                )
                if correct_letter and correct_letter[0] in ["A", "B", "C", "D"]:
                    correct_index = ord(correct_letter[0]) - 65  # Convert A→0, B→1, ...
                else:
                    # Default to A if model messed up
                    correct_index = 0

                questions.append({
                    "q": q,
                    "options": [a, b, c, d],
                    "a": correct_index,
                    "explanation": explanation
                })

            if not questions:
                raise ValueError("No valid questions generated from model output")

            return questions[:10]  # Return top 10 only

        except Exception as e:
            print(f"Error generating AI questions: {e}")
            return []


    def create_story_module():
        story_content = generate_ai_content("a captivating, immersive story", topic, "Create a detailed narrative with characters, setting, and plot. Make it engaging and emotionally rich. About 400-500 words.")
        image_prompt = f"A cinematic, beautiful illustration on {topic}, dramatic lighting, any particular famous image related to {topic}"
        story_image_url = generate_image(image_prompt)
        
        # Generate follow-up questions for discussion
        discussion_questions = [
            f"If you were the main character in this story about {topic}, what would you do differently?",
            f"What emotions did this story about {topic} evoke in you?",
            f"How does this story relate to your own experiences with {topic}?",
            f"What would you add to this story to make it even better?",
            f"If you could ask the main character one question about {topic}, what would it be?"
        ]
        
        return {
            "type": "story",
            "title": f"The Epic Tale of {topic.title()}",
            "content": story_content,
            "image_url": story_image_url,
            "image_prompt": image_prompt,
            "steps": [
                {"kind": "intro", "caption": f"Welcome to an epic journey through the world of {topic}...", "image_url": generate_image(f"Epic introduction scene for {topic}, dramatic")},
                {"kind": "story", "content": story_content, "image_url": story_image_url, "image_prompt": image_prompt, "voice_note": "Story narration available"},
                {"kind": "discussion", "prompts": discussion_questions, "title": "Let's Discuss This Amazing Story"},
                {"kind": "reflection", "prompts": [f"Write your own short story about {topic}", f"Create a character who loves {topic}", f"Describe your ideal {topic} adventure"]},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Story Master"}}
            ],
            "narrator": {"name": "Luna", "avatar": "📖", "persona": "epic storyteller"},
            "reward": {"type": "badge", "title": f"{topic.title()} Story Master"}
        }

    def create_meditation_module():
        meditation_script = generate_ai_content("a guided meditation script", topic, "Include breathing exercises and calming imagery. Make it 5-7 minutes long.")
        return {
            "type": "meditation",
            "title": f"Mindful {topic.title()} Meditation",
            "script": meditation_script,
            "steps": [
                {"kind": "intro", "caption": f"Let's find peace through {topic}..."},
                {"kind": "breathing", "seconds": 60, "caption": "Breathe deeply and center yourself"},
                {"kind": "meditation", "script": meditation_script, "duration": 300},
                {"kind": "gratitude", "prompts": [f"What are you grateful for about {topic}?", "How does {topic} bring peace to your life?"]},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Zen Master"}}
            ],
            "narrator": {"name": "Zen", "avatar": "🧘", "persona": "calm guide"},
            "reward": {"type": "badge", "title": f"{topic.title()} Zen Master"}
        }

    def create_art_therapy_module():
        art_prompts = [
            f"Draw your interpretation of {topic}",
            f"Create a color palette that represents {topic}",
            f"Sketch how {topic} makes you feel",
            f"Design a symbol for {topic}"
        ]
        return {
            "type": "art",
            "title": f"Artistic {topic.title()} Journey",
            "prompts": art_prompts,
            "steps": [
                {"kind": "intro", "caption": f"Let's explore {topic} through art..."},
                {"kind": "art", "prompts": art_prompts},
                {"kind": "sharing", "prompts": ["Describe your artwork", "What colors did you choose and why?", "How did creating art about this topic make you feel?"]},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Artist"}}
            ],
            "narrator": {"name": "Artemis", "avatar": "🎨", "persona": "creative guide"},
            "reward": {"type": "badge", "title": f"{topic.title()} Artist"}
        }

    def create_music_therapy_module():
        music_description = generate_ai_content("music recommendations", topic, "Suggest calming, inspiring music that relates to this topic.")
        return {
            "type": "music",
            "title": f"Musical {topic.title()} Experience",
            "description": music_description,
            "steps": [
                {"kind": "intro", "caption": f"Let's explore {topic} through music..."},
                {"kind": "music", "loop": "/static/instant/calm-loop.mp3", "duration": 180},
                {"kind": "reflection", "prompts": [f"How does music enhance your connection to {topic}?", "What sounds remind you of {topic}?"]},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Melodist"}}
            ],
            "narrator": {"name": "Melody", "avatar": "🎵", "persona": "musical guide"},
            "reward": {"type": "badge", "title": f"{topic.title()} Melodist"}
        }

    def create_poetry_module():
        poem_content = generate_ai_content("a beautiful poem", topic, "Make it inspiring and emotionally resonant. Include vivid imagery.")
        return {
            "type": "poetry",
            "title": f"Poetic {topic.title()}",
            "poem": poem_content,
            "steps": [
                {"kind": "intro", "caption": f"Let's find poetry in {topic}..."},
                {"kind": "poetry", "content": poem_content},
                {"kind": "writing", "prompts": [f"Write your own poem about {topic}", "What metaphors come to mind for {topic}?"]},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Poet"}}
            ],
            "narrator": {"name": "Verse", "avatar": "📝", "persona": "poetic soul"},
            "reward": {"type": "badge", "title": f"{topic.title()} Poet"}
        }

    def create_comprehensive_quiz_module(topic):
        questions = create_comprehensive_quiz(topic)
        return {
            "type": "quiz",
            "title": f"Deep Dive: {topic.title()} Quiz",
            "questions": questions,
            "steps": [
                {"kind": "intro", "caption": f"Let's test your knowledge of {topic}..."},
                {"kind": "quiz", "questions": questions},
                {"kind": "reward", "reward": {"type": "badge", "title": f"{topic.title()} Expert"}}
            ],
            "narrator": {"name": "Quizzy", "avatar": "🧠", "persona": "knowledgeable guide"},
            "reward": {"type": "badge", "title": f"{topic.title()} Expert"}
        }

    # Choose between quiz and story modules only
    if random.random() < 0.5:  # 50% chance for quiz
        spec = create_comprehensive_quiz_module(topic)
    else:
        spec = create_story_module()

    doc = {"user_id": user_id, "type": "generated", "topic": topic, "spec": spec, "created_at": datetime.utcnow()}
    inserted = modules.insert_one(doc)
    return inserted.inserted_id, spec


# ===================== 🌐 ROUTES =====================

@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


# ---------------------- Navigation & Pages ----------------------
@app.route("/explore")
def explore():
    if "user_id" not in session:
        return redirect(url_for("login"))
    # Fetch a few templates to show as default modules
    items = list(module_templates.find({}, {"_id": 0}).limit(12))
    return render_template("explore.html", items=items)


@app.route("/mymodules")
def mymodules():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    items = list(modules.find({"user_id": user_id}).sort("created_at", -1).limit(50))
    return render_template("mymodules.html", items=items)


@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    items = list(rewards.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1))
    return render_template("profile.html", items=items)


@app.route("/onboarding")
def onboarding():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("onboarding.html")


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


# ✅ Import BERT emotion analyzer for maximum accuracy
from emotion_analyzer_bert import analyze_emotion_bert as analyze_emotion

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"reply": "Session expired. Please log in again."})

    user_id = session["user_id"]
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please say something to begin."})

    # 🧠 Run emotion analysis on user message
    emotion_result = analyze_emotion(user_msg)
    emotion_label = emotion_result["emotion"]
    emotion_confidence = emotion_result["confidence"]

    # 🧠 Fetch user and previous messages
    user_data = users.find_one({"name": user_id})
    nickname = user_data.get("nickname", "friend")

    existing = chat_history.find_one({"user_id": user_id}) or {"messages": [], "emotions": []}
    messages = existing.get("messages", [])
    emotions = existing.get("emotions", [])

    # 👋 Greet on first message
    if len(messages) == 0:
        greeting = f"Hey {nickname}! 😊 It's nice to meet you. How are you feeling today?"
        messages.append({"role": "assistant", "content": greeting})
        # Track initial detected emotion as a baseline point as well
        emotions.append({
            "ts": datetime.utcnow(),
            "emotion": emotion_label,
            "confidence": round(emotion_confidence, 2)
        })
        chat_history.update_one(
            {"user_id": user_id},
            {"$set": {"messages": messages, "emotions": emotions}},
            upsert=True
        )
        return jsonify({"reply": greeting, "emotion": emotion_label, "confidence": emotion_confidence})

    # 🧠 Store user message with emotion
    messages.append({
        "role": "user",
        "content": user_msg,
        "emotion": emotion_label,
        "confidence": round(emotion_confidence, 2)
    })
    # Store emotion history point
    emotions.append({
        "ts": datetime.utcnow(),
        "emotion": emotion_label,
        "confidence": round(emotion_confidence, 2)
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

    # 🧹 Clean messages (remove unsupported fields like 'emotion')
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
    chat_history.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages, "emotions": emotions}},
        upsert=True
    )

    # ✅ Return both reply and emotion
    return jsonify({
        "reply": reply,
        "emotion": emotion_label,
        "confidence": round(emotion_confidence, 2)
    })


# ---------------------- Onboarding & Passive Preference Capture ----------------------
@app.route("/api/onboarding/submit", methods=["POST"])
def onboarding_submit():
    if "user_id" not in session:
        return jsonify({"ok": False}), 401
    user_id = session["user_id"]
    data = request.get_json() or {}
    # Expect fields: color_cloud, sound_choice, scene_choice, self_care
    pref_delta = {
        "color_cloud": data.get("color_cloud"),
        "sound": data.get("sound_choice"),
        "scene": data.get("scene_choice"),
        "self_care": data.get("self_care")
    }
    preferences.update_one(
        {"user_id": user_id},
        {
            "$setOnInsert": {"user_id": user_id, "created_at": datetime.utcnow()},
            "$push": {"events": {"ts": datetime.utcnow(), "type": "onboarding", "data": pref_delta}},
            "$set": {"last": pref_delta}
        },
        upsert=True
    )
    return jsonify({"ok": True})


@app.route("/api/rewards/grant", methods=["POST"])
def grant_reward():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    reward_doc = {
        "user_id": user_id,
        "kind": payload.get("kind", "badge"),
        "title": payload.get("title", "Achievement"),
        "meta": payload.get("meta", {}),
        "created_at": datetime.utcnow()
    }
    rewards.insert_one(reward_doc)
    return jsonify({"ok": True})


# ---------------------- Music Therapy ----------------------
@app.route("/module/music")
def module_music_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("module_music.html")


@app.route("/api/modules/music/recommend", methods=["POST"]) 
def music_recommend():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    current_mood = payload.get("currentMood", "neutral")
    # Instant fallback from templates
    fallback = module_templates.find_one({"type": "music"}, {"_id": 0})
    recommendation = {
        "title": fallback.get("title", "Calm Loop"),
        "mood": current_mood,
        "tracks": [
            {"title": fallback.get("title", "Calm Loop"), "url": fallback.get("audio_url", "")}
        ]
    }
    # Log module instance
    instance = {
        "user_id": user_id,
        "type": "music",
        "spec": recommendation,
        "created_at": datetime.utcnow()
    }
    modules.insert_one(instance)
    return jsonify(recommendation)


# ---------------------- Meditation Therapy ----------------------
@app.route("/module/meditation")
def module_meditation_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("module_meditation.html")


@app.route("/api/modules/meditation/generate", methods=["POST"]) 
def meditation_generate():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    length = int(payload.get("length", 60))
    style = payload.get("style", "calm")
    template = module_templates.find_one({"type": "meditation"}, {"_id": 0}) or {}
    script = template.get("script", "Breathe in... and out... you are safe.")
    spec = {
        "title": f"{style.title()} Meditation",
        "length": length,
        "scriptText": script,
        "ttsUrl": ""  # could be generated by a worker later
    }
    modules.insert_one({"user_id": user_id, "type": "meditation", "spec": spec, "created_at": datetime.utcnow()})
    return jsonify(spec)


# ---------------------- Art Therapy ----------------------
@app.route("/module/art")
def module_art_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("module_art.html")


@app.route("/api/modules/art/generate", methods=["POST"]) 
def art_generate():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    style_hint = payload.get("styleHint", "dreamy")
    # Instant client-side remix; backend returns a simple spec
    spec = {
        "title": "Remix Your Mood",
        "styleHint": style_hint,
        "generatedImageUrl": ""  # optional, could be filled by worker
    }
    modules.insert_one({"user_id": user_id, "type": "art", "spec": spec, "created_at": datetime.utcnow()})
    return jsonify(spec)


# ---------------------- Poetic / Story Therapy ----------------------
@app.route("/module/poem")
def module_poem_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("module_poem.html")


@app.route("/api/modules/poem/generate", methods=["POST"]) 
def poem_generate():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    tone = payload.get("tone", "gentle")
    template = module_templates.find_one({"type": "poem"}, {"_id": 0}) or {}
    text = template.get("text", "A gentle breeze sings,\nWorries soften into light,\nYou are held by dawn.")
    spec = {
        "title": f"{tone.title()} Poem",
        "text": text,
        "ttsUrl": ""
    }
    modules.insert_one({"user_id": user_id, "type": "poem", "spec": spec, "created_at": datetime.utcnow()})
    return jsonify(spec)


# ---------------------- Micro-Quizzes ----------------------
@app.route("/api/modules/quiz/answer", methods=["POST"]) 
def quiz_answer():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    choice_id = payload.get("choiceId")
    # Map choiceId to simple preference delta
    delta = {"choice": choice_id}
    preferences.update_one(
        {"user_id": user_id},
        {"$push": {"events": {"ts": datetime.utcnow(), "type": "quiz", "data": delta}}, "$set": {"last_quiz": delta}},
        upsert=True
    )
    return jsonify({"ok": True})


# ---------------------- "More" — Dynamic Module Generator ----------------------
@app.route("/more")
def more_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("more.html")


@app.route("/api/modules/more/generate", methods=["POST"]) 
def more_generate():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    user_input = (payload.get("input", "") or "").strip()
    mod_id, spec = create_topic_module(user_id, user_input)
    # grant small reward
    rewards.insert_one({
        "user_id": user_id,
        "kind": "badge",
        "title": "Spark Creator",
        "meta": {"topic": user_input},
        "created_at": datetime.utcnow()
    })
    return jsonify({"id": str(mod_id), "spec": spec, "url": f"/module/{str(mod_id)}"})


@app.route("/api/modules/more/regenerate", methods=["POST"]) 
def more_regenerate():
    if "user_id" not in session:
        return jsonify({}), 401
    user_id = session["user_id"]
    payload = request.get_json() or {}
    topic = (payload.get("topic", "") or "").strip()
    mod_id, spec = create_topic_module(user_id, topic)
    return jsonify({"id": str(mod_id), "spec": spec, "url": f"/module/{str(mod_id)}"})


@app.route("/module/<mod_id>")
def module_dynamic(mod_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        doc = modules.find_one({"_id": ObjectId(mod_id)})
    except Exception:
        doc = None
    if not doc:
        return redirect(url_for("explore"))
    return render_template("module_dynamic.html", spec=doc.get("spec", {}), topic=doc.get("topic", ""), mod_id=mod_id)


@app.route("/api/generate-image", methods=["POST"])
def generate_image_api():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    
    try:
        image_url = generate_image(prompt)
        return jsonify({"image_url": image_url, "prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------- Privacy / Consent ----------------------
@app.route("/privacy")
def privacy_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("privacy.html")


# ========= 📈 Emotion Analytics =========
@app.route("/emotions")
def emotions_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("emotions.html")


@app.route("/emotion_history")
def emotion_history():
    if "user_id" not in session:
        return jsonify([])
    user_id = session["user_id"]
    existing = chat_history.find_one({"user_id": user_id}) or {"emotions": []}
    data = existing.get("emotions", [])
    # Serialize datetime to ISO
    serialized = [
        {
            "ts": (e.get("ts").isoformat() + "Z") if isinstance(e.get("ts"), datetime) else e.get("ts"),
            "emotion": e.get("emotion"),
            "confidence": e.get("confidence", 0)
        }
        for e in data
    ]
    return jsonify(serialized)



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
