from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import tempfile

app = Flask(__name__)

# Load the transcription model once
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]

    # Save the uploaded audio temporarily
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as temp:
        audio.save(temp.name)
        path = temp.name

    try:
        # Convert speech to text
        segments, info = model.transcribe(path)

        text = " ".join(segment.text for segment in segments)

        return jsonify({"text": text.strip()})

    finally:
        os.remove(path)

if __name__ == "__main__":
    app.run(port=8000)