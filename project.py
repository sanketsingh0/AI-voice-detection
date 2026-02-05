from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import os
import librosa
import numpy as np
import traceback

app = FastAPI()

# this key for authentification 
API_KEY = "GUVI1234"

class AudioInput(BaseModel):
    audio_base64: str


def safe_b64decode(data: str) -> bytes:
    data = data.strip()
    data = data.replace("\n", "").replace("\r", "").replace(" ", "")
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return base64.b64decode(data)


@app.post("/detect")
def detect_audio(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    #  API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Base64 decode
        audio_bytes = safe_b64decode(data.audio_base64)

        # Save audio
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "input_audio.mp3")

        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std = np.std(mfcc)

        # Decision logic
        if mfcc_std < 15:
            result = "AI_GENERATED"
            confidence = round(1 - (mfcc_std / 20), 2)
        else:
            result = "HUMAN"
            confidence = round(min(mfcc_std / 40, 1.0), 2)

        return {
            "result": result,
            "confidence": confidence
        }

    except Exception:
        return {
            "error": "failed",
            "details": traceback.format_exc()

        }
