# Audio-to-Ai-Dubbing-

This project is a small prototype web app that allows uploading audio or video files (any length) and generating a dubbed AI voice file you can download.

New: multi-speaker support
- Added a speaker detection step (diarization) that detects speakers and returns time-aligned segments.
- Frontend shows a "Detect speakers" button after upload and then displays detected speakers with a voice selector for each.
- The generate step will transcribe each speaker segment and synthesize the assigned voice for that speaker, then re-assemble segments into a single dubbed audio file.

What I changed
- app.py: added endpoints and functions for speaker diarization (/diarize) and segment-based synthesis. Uses pyannote's pretrained speaker-diarization pipeline if `PYANNOTE_AUTH_TOKEN` environment variable is set; otherwise falls back to a single-speaker assumption.
- templates/index.html: UI for detecting speakers and selecting voices per speaker.
- README.md: updated explanations.

How it works (short)
1. Upload audio/video.
2. Click "Detect speakers". Server will extract audio and run diarization (pyannote if configured). The server responds with detected segments and a list of speakers.
3. Choose a voice for each detected speaker.
4. Click "Generate dubbed voice". The server will:
   - Extract each segment
   - Transcribe the segment (OpenAI if `OPENAI_API_KEY` is set)
   - Synthesize the transcript using the selected voice (ElevenLabs if `ELEVENLABS_API_KEY` is set)
   - Concatenate synthesized segments into a single MP3 file

Notes & Limitations
- Diarization requires `PYANNOTE_AUTH_TOKEN` to be set and `pyannote.audio` installed; otherwise the app will treat the whole file as a single speaker.
- The pipeline is still transcript->TTS (dubbing), not full voice cloning. If you need true voice conversion/clone (preserve original voice), integrate a dedicated voice conversion model.
- Generation is synchronous and can take a long time for long files — consider moving to background jobs for production.
- For large files, consider resumable uploads and chunking.

Environment variables (optional)
- OPENAI_API_KEY: transcription
- ELEVENLABS_API_KEY: TTS
- PYANNOTE_AUTH_TOKEN: speaker diarization

