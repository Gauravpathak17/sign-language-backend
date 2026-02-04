import os
import socketio
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# FASTAPI
# =========================
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SOCKET.IO
# =========================
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
)
app = socketio.ASGIApp(sio, fastapi_app)

# =========================
# ROOM STATE
# =========================
room_users = {}

# =========================
# LOAD ML MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_phrases")

model = joblib.load(os.path.join(MODEL_DIR, "asl_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ Model loaded")

# =========================
# REQUEST SCHEMA
# =========================
class LandmarkRequest(BaseModel):
    landmarks: list

# =========================
# PREPROCESS
# =========================
def preprocess_landmarks(landmarks):
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    lm = lm - lm[0]
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm /= max_val
    return lm.flatten()

# =========================
# PREDICT API
# =========================
@fastapi_app.post("/predict")
async def predict(req: LandmarkRequest):
    if len(req.landmarks) != 63:
        return {"prediction": None, "confidence": 0.0}

    X = scaler.transform([preprocess_landmarks(req.landmarks)])
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    if conf < 0.35:
        return {"prediction": None, "confidence": conf}

    return {
        "prediction": label_encoder.inverse_transform([idx])[0],
        "confidence": conf,
    }

# =========================
# SOCKET EVENTS
# =========================
@sio.event
async def connect(sid, environ):
    print("🟢 Connected:", sid)

@sio.event
async def disconnect(sid):
    print("🔴 Disconnected:", sid)
    for room in list(room_users):
        room_users[room].discard(sid)
        if not room_users[room]:
            del room_users[room]

@sio.event
async def join(sid, data):
    room = data["room"]
    await sio.enter_room(sid, room)

    room_users.setdefault(room, set()).add(sid)
    print("👥 Room:", room_users[room])

    if len(room_users[room]) == 2:
        caller = list(room_users[room])[0]
        await sio.emit("ready", {"caller": caller}, room=room)

@sio.event
async def offer(sid, data):
    await sio.emit("offer", data, room=data["room"], skip_sid=sid)

@sio.event
async def answer(sid, data):
    await sio.emit("answer", data, room=data["room"], skip_sid=sid)

@sio.event
async def ice(sid, data):
    await sio.emit("ice", data, room=data["room"], skip_sid=sid)

@sio.event
async def chat_message(sid, data):
    await sio.emit("chat_message", data, room=data["room"], skip_sid=sid)
