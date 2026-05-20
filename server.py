"""
server.py — Flask web server for BlipBlop.
Serves the UI and exposes a /generate endpoint that streams a WAV file.
"""

import io
import random
import sys
import os

from flask import Flask, render_template, request, send_file, jsonify

# Import synthesis from blipblop.py in the same directory
sys.path.insert(0, os.path.dirname(__file__))
import main as bb

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.get_json(silent=True) or {}
    duration = float(data.get("duration", 10.0))
    quality  = data.get("quality", "high")

    duration = max(1.0, min(duration, 300.0))
    if quality not in bb.QUALITY_PRESETS:
        quality = "high"

    # Run synthesis into an in-memory buffer
    sample_rate, subtype, _ = bb.QUALITY_PRESETS[quality]
    bb.SAMPLE_RATE = sample_rate

    import numpy as np
    import soundfile as sf

    parts = []
    total = 0.0
    while total < duration:
        signal, _style, _freq = bb.make_blip()
        gap = random.uniform(0.005, 0.225)
        dur = len(signal) / bb.SAMPLE_RATE
        parts.append(signal)
        parts.append(bb.silence(gap))
        total += dur + gap

    audio = np.concatenate(parts)
    target_samples = int(duration * bb.SAMPLE_RATE)
    audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, ((0, target_samples - len(audio)), (0, 0)))

    # Normalize to -0.1 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9886

    buf = io.BytesIO()
    sf.write(buf, audio, bb.SAMPLE_RATE, subtype=subtype, format="WAV")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="blipblop.wav",
    )


if __name__ == "__main__":
    print("🎵 Glorb server → http://localhost:5000")
    app.run(debug=False, port=5000)
