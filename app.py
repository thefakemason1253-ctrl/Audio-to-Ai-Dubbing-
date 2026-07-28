from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
import os
import uuid
from pathlib import Path
import subprocess
import shutil
import requests
import math
import json

# Optional integrations are demonstrated below using environment variables:
# - OPENAI_API_KEY: for transcription (Whisper-like) and/or TTS if you choose
# - ELEVENLABS_API_KEY: for ElevenLabs TTS synthesis
# If no API keys are provided, the app will save uploaded files and return them unchanged as a fallback.

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm", "ogg"}
MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024  # 4 GB max file upload size, tune as needed

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(input_path: Path, out_path: Path) -> Path:
    """Use ffmpeg to extract audio and convert to WAV 16k mono (compatible with many STT/TTS tools)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return out_path


def transcribe_with_openai(audio_path: Path) -> str:
    """Example: call OpenAI's speech->text API if OPENAI_API_KEY is set. This function is optional; you can swap in any transcription provider.
    This is a minimal example that streams the file to the OpenAI API (`/v1/audio/transcriptions`).
    """
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    # NOTE: you may need the latest openai python package and an endpoint name. This example uses requests directly.
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": open(audio_path, "rb")}
    data = {"model": "gpt-4o-transcribe"} if False else {"model": "whisper-1"}

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=3600)
    resp.raise_for_status()
    j = resp.json()
    return j.get("text", "")


def synthesize_with_elevenlabs(text: str, out_path: Path, voice: str = "alloy") -> Path:
    """Example ElevenLabs TTS call. Requires ELEVENLABS_API_KEY environment var.
    See https://api.elevenlabs.io for up-to-date docs. This is a minimal example and may need adjustment.
    """
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}}
    resp = requests.post(url, headers=headers, json=payload, timeout=3600)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def convert_voice_via_transcribe_and_tts(extracted_audio: Path, output_audio: Path, voice: str = "alloy") -> Path:
    """Simple approach: transcribe audio -> synthesize text with target voice TTS.
    This is not true voice cloning but yields a dubbed audio track that matches the spoken content.
    For full voice cloning (preserving prosody/voice characteristics), you will need specialized models (e.g., voice conversion models) or vendor APIs.
    """
    try:
        # 1) Transcribe
        transcript = transcribe_with_openai(extracted_audio)
    except Exception as e:
        app.logger.warning("Transcription failed or is not configured: %s", e)
        transcript = ""

    if not transcript:
        raise RuntimeError("No transcript available; configure a transcription API or provide text input")

    # 2) Synthesize
    # Prefer ElevenLabs if configured. Fallback to OpenAI TTS could be implemented similarly.
    try:
        if os.environ.get("ELEVENLABS_API_KEY"):
            return synthesize_with_elevenlabs(transcript, output_audio, voice=voice)
        else:
            raise RuntimeError("No TTS provider configured")
    except Exception as e:
        app.logger.exception("Synthesis failed: %s", e)
        raise


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{file_id}.{ext}"
    path = UPLOAD_DIR / saved_name
    file.save(path)
    return jsonify({"file_id": file_id, "filename": saved_name})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    target_voice = data.get("voice", "alloy")
    if not file_id:
        return jsonify({"error": "file_id required"}), 400

    # find upload
    candidates = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not candidates:
        return jsonify({"error": "file not found"}), 404
    input_path = candidates[0]

    # extract audio
    extracted = OUTPUT_DIR / f"{file_id}_extracted.wav"
    try:
        extract_audio(input_path, extracted)
    except Exception as e:
        app.logger.exception("Error extracting audio: %s", e)
        return jsonify({"error": "failed to extract audio", "details": str(e)}), 500

    # convert voice (this may call external APIs and take time)
    output_file = OUTPUT_DIR / f"{file_id}_dub.mp3"
    try:
        # If user has not configured any API keys, do a fallback copy of extracted audio to output
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")):
            # fallback: just copy extracted WAV to MP3 using ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(extracted),
                "-acodec",
                "libmp3lame",
                "-b:a",
                "192k",
                str(output_file),
            ]
            subprocess.check_call(cmd)
        else:
            # attempt real conversion pipeline
            convert_voice_via_transcribe_and_tts(extracted, output_file, voice=target_voice)
    except Exception as e:
        app.logger.exception("Error converting voice: %s", e)
        return jsonify({"error": "voice conversion failed", "details": str(e)}), 500

    download_url = url_for("download", file=file_id + "_dub.mp3")
    return jsonify({"download_url": download_url})


@app.route("/download/<path:file>")
def download(file):
    path = OUTPUT_DIR / file
    if not path.exists():
        return ("Not found", 404)
    return send_from_directory(OUTPUT_DIR, file, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
