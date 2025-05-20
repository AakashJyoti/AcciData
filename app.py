# app.py
from flask import Flask, request, jsonify, send_from_directory
from motor import ABHI
import os
import uuid
import json
from flask_cors import CORS

app = Flask(__name__)

app = Flask(__name__, static_folder="dist")

CORS(app)

CHAT_HISTORY_DIR = "sessions"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def get_history_path(session_id):
    return os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")


def load_history(session_id):
    try:
        with open(get_history_path(session_id), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_history(session_id, messages):
    with open(get_history_path(session_id), "w") as f:
        json.dump(messages, f)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/new_session", methods=["POST"])
def new_session():
    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    session_id = str(uuid.uuid4())
    abhi = ABHI()
    abhi.reset_system_message()
    save_history(session_id, abhi.messages)
    return jsonify({"session_id": session_id})


@app.route("/chat", methods=["POST"])
def chat():
    session_id = request.json.get("session_id")
    user_input = request.json.get("user_input")

    if not session_id or not user_input:
        return jsonify({"error": "Missing session_id or user_input"}), 400

    history = load_history(session_id)
    abhi = ABHI(history)
    response = abhi.chat(user_input)
    save_history(session_id, abhi.messages)
    return jsonify({"response": response})


@app.route("/history", methods=["POST"])
def get_history():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    messages = load_history(session_id)
    return jsonify({"messages": messages})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
