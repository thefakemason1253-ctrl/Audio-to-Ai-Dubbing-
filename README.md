# Audio-to-Ai-Dubbing-

This project is a small prototype web app that allows uploading audio or video files (any length) and generating a dubbed AI voice file you can download.

What I added
- A Flask app (app.py) with endpoints to upload files, generate a dubbed voice, and download the resulting audio.
- A simple frontend (templates/index.html) for uploading files and showing a "Generate dubbed voice" button after upload.
- A conversion pipeline that:
  1. Extracts audio from uploaded audio/video files using ffmpeg (to WAV 16k mono).
  2. Optionally transcribes the audio using OpenAI (if OPENAI_API_KEY is set) and synthesizes speech using ElevenLabs (if ELEVENLABS_API_KEY is set) to produce a dubbed audio file.
  3. Falls back to returning the extracted audio (converted to MP3) if no API keys are configured.

How to run locally

1. Install ffmpeg on your system (the app calls the ffmpeg binary). On macOS: `brew install ffmpeg`. On Debian/Ubuntu: `sudo apt install ffmpeg`.

2. Create a virtual environment and install Python requirements:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. (Optional) Configure API keys as environment variables if you want real transcription and TTS:

   export OPENAI_API_KEY="sk-..."
   export ELEVENLABS_API_KEY="..."

4. Run the app:

   python app.py

5. Open http://localhost:5000 in your browser. Upload an audio or video file, then click "Generate dubbed voice".

Notes and next steps
- This prototype uses a transcript->TTS pipeline for dubbing. That gives accurate text matching but is not a full voice-cloning solution. For real voice cloning (preserving original speaker voice), integrate a voice-conversion model or vendor that supports voice cloning (many providers offer an API).
- The generate step may take a long time on large files; in production you should run conversion tasks asynchronously (Celery/RQ/Kubernetes jobs) and provide task status endpoints.
- The code includes example hooks for OpenAI and ElevenLabs APIs; you may need to adapt payloads to match the current vendor API versions.

