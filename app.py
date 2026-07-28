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
# - PYANNOTE_AUTH_TOKEN: for speaker diarization using pyannote pipeline (optional)
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


def run_speaker_diarization(wav_path: Path):
    """Run speaker diarization and return a list of segments: [{"start": float, "end": float, "speaker": "SPEAKER_0"}, ...]
    This uses pyannote if PYANNOTE_AUTH_TOKEN is set. Otherwise returns a single speaker covering the whole file as a fallback.
    """
    token = os.environ.get("PYANNOTE_AUTH_TOKEN")
    if token:
        try:
            # Lazy-import pyannote to avoid making it a hard requirement
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
            diarization = pipeline(str(wav_path))
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
            # Normalize speaker labels to simple names
            speakers_map = {}
            out = []
            idx = 0
            for s in segments:
                sp = s["speaker"]
                if sp not in speakers_map:
                    speakers_map[sp] = f"speaker_{len(speakers_map)+1}"
                out.append({"start": s["start"], "end": s["end"], "speaker": speakers_map[sp]})
            return out
        except Exception as e:
            app.logger.exception("pyannote diarization failed: %s", e)
            # fall-through to fallback

    # Fallback: single speaker covering whole file
    # Get duration using ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ]
        out = subprocess.check_output(cmd).decode().strip()
        duration = float(out) if out else 0.0
    except Exception:
        duration = 0.0
    return [{"start": 0.0, "end": duration, "speaker": "speaker_1"}]


def extract_segment(input_path: Path, start: float, end: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        str(start),
        "-to",
        str(end),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return out_path


def convert_voice_via_transcribe_and_tts_for_segments(input_path: Path, segments, output_audio: Path, voice_map) -> Path:
    """For each segment, extract audio, transcribe, synthesize using the selected voice for that speaker, then concatenate."""
    tmp_dir = OUTPUT_DIR / f"tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    segment_files = []

    try:
        for i, seg in enumerate(segments):
            seg_wav = tmp_dir / f"seg_{i}.wav"
            # extract from original input audio (not the pre-extracted wav) to preserve timing
            extract_segment(input_path, seg["start"], seg["end"], seg_wav)

            # Transcribe
            transcript = ""
            try:
                transcript = transcribe_with_openai(seg_wav)
            except Exception as e:
                app.logger.warning("Transcription failed for segment %d: %s", i, e)

            # Choose voice for speaker
            speaker = seg.get("speaker", "speaker_1")
            voice = voice_map.get(speaker, voice_map.get("default", "alloy"))

            # Synthesize
            out_seg_mp3 = tmp_dir / f"seg_{i}.mp3"
            try:
                if transcript:
                    if os.environ.get("ELEVENLABS_API_KEY"):
                        synth_path = tmp_dir / f"seg_{i}_synth.wav"
                        synthesize_with_elevenlabs(transcript, synth_path, voice=voice)
                        # convert synth output to mp3 with consistent params
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(synth_path),
                            "-acodec",
                            "libmp3lame",
                            "-b:a",
                            "192k",
                            str(out_seg_mp3),
                        ]
                        subprocess.check_call(cmd)
                    else:
                        # If no TTS provider, fallback: keep original segment as mp3
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(seg_wav),
                            "-acodec",
                            "libmp3lame",
                            "-b:a",
                            "192k",
                            str(out_seg_mp3),
                        ]
                        subprocess.check_call(cmd)
                else:
                    # No transcript: fallback to original audio
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(seg_wav),
                        "-acodec",
                        "libmp3lame",
                        "-b:a",
                        "192k",
                        str(out_seg_mp3),
                    ]
                    subprocess.check_call(cmd)
            except Exception as e:
                app.logger.exception("Synthesis/conversion failed for segment %d: %s", i, e)
                # as a last resort, copy the segment wav to mp3
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(seg_wav),
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(out_seg_mp3),
                ]
                subprocess.check_call(cmd)

            segment_files.append(out_seg_mp3)

        # Concatenate segment files into final output
        list_file = tmp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for p in segment_files:
                # ffmpeg concat requires paths like: file '/absolute/path'
                f.write(f"file '{str(p.resolve())}'\n")

        # Create final mp3
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_audio),
        ]
        subprocess.check_call(cmd)

        return output_audio
    finally:
        # cleanup
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


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


@app.route("/diarize", methods=["POST"])
def diarize():
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    if not file_id:
        return jsonify({"error": "file_id required"}), 400

    # find upload
    candidates = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not candidates:
        return jsonify({"error": "file not found"}), 404
    input_path = candidates[0]

    # extract a working WAV for diarization
    wav_path = OUTPUT_DIR / f"{file_id}_for_diarize.wav"
    try:
        extract_audio(input_path, wav_path)
    except Exception as e:
        app.logger.exception("Error extracting audio for diarization: %s", e)
        return jsonify({"error": "failed to extract audio for diarization", "details": str(e)}), 500

    try:
        segments = run_speaker_diarization(wav_path)
    except Exception as e:
        app.logger.exception("Diarization failed: %s", e)
        return jsonify({"error": "diarization failed", "details": str(e)}), 500

    # Build list of unique speakers
    speakers = []
    seen = set()
    for s in segments:
        if s["speaker"] not in seen:
            seen.add(s["speaker"])
            speakers.append({"id": s["speaker"]})

    return jsonify({"segments": segments, "speakers": speakers})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    voice_map = data.get("voice_map", {})
    if not file_id:
        return jsonify({"error": "file_id required"}), 400

    # find upload
    candidates = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not candidates:
        return jsonify({"error": "file not found"}), 404
    input_path = candidates[0]

    # extract a working WAV
    extracted = OUTPUT_DIR / f"{file_id}_extracted.wav"
    try:
        extract_audio(input_path, extracted)
    except Exception as e:
        app.logger.exception("Error extracting audio: %s", e)
        return jsonify({"error": "failed to extract audio", "details": str(e)}), 500

    # Run diarization to get segments
    try:
        segments = run_speaker_diarization(extracted)
    except Exception as e:
        app.logger.exception("Error running diarization: %s", e)
        segments = [{"start": 0.0, "end": 0.0, "speaker": "speaker_1"}]

    # convert voice per segment
    output_file = OUTPUT_DIR / f"{file_id}_dub.mp3"
    try:
        # If no APIs configured, fallback to a simple conversion of the whole file
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")):
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
            convert_voice_via_transcribe_and_tts_for_segments(extracted, segments, output_file, voice_map)
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
