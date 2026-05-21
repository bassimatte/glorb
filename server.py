"""
server.py — Flask web server for Glorb.
Serves the UI and exposes /generate (WAV or ZIP) and /nature-variants endpoints.
"""

import io
import zipfile
import sys
import os

import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, send_file

sys.path.insert(0, os.path.dirname(__file__))
import main as bb

app = Flask(__name__)


def _wav_bytes(audio, sample_rate, subtype):
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype=subtype, format="WAV")
    buf.seek(0)
    return buf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.get_json(silent=True) or {}
    duration = float(data.get("duration", 10.0))
    quality  = data.get("quality", "high")
    mode     = data.get("mode", "glorb")
    variant  = data.get("variant", None)   # for nature mode

    duration = max(1.0, min(duration, 300.0))
    if quality not in bb.QUALITY_PRESETS:
        quality = "high"

    sample_rate, subtype, _ = bb.QUALITY_PRESETS[quality]
    bb.SAMPLE_RATE = sample_rate

    # ── UI Pack → ZIP of individual WAVs ──────────────────────────
    if mode == "ui-pack":
        sounds = bb.make_ui_pack()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, audio in sounds.items():
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.9886
                wav = _wav_bytes(audio, sample_rate, subtype)
                zf.writestr(f"{name}.wav", wav.read())
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype="application/zip",
                         as_attachment=False, download_name="glorb_ui_pack.zip")

    # ── Nature → single WAV ───────────────────────────────────────
    if mode == "nature":
        audio, used_variant = bb.make_nature_sequence(duration, variant)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False,
                         download_name=f"glorb_nature_{used_variant}.wav")

    # ── Retro → single WAV ────────────────────────────────────────
    if mode == "retro":
        audio = bb.make_retro_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_retro.wav")

    # ── Glorb (default) → single WAV ─────────────────────────────
    import random
    parts, total = [], 0.0
    while total < duration:
        signal, _style, _freq = bb.make_blip()
        gap = random.uniform(0.005, 0.225)
        parts.append(signal)
        parts.append(bb.silence(gap))
        total += len(signal) / bb.SAMPLE_RATE + gap

    audio = bb._finalise(np.concatenate(parts), duration)
    buf = _wav_bytes(audio, sample_rate, subtype)
    return send_file(buf, mimetype="audio/wav",
                     as_attachment=False, download_name="glorb.wav")


@app.route("/nature-variants")
def nature_variants():
    from flask import jsonify
    return jsonify(list(bb._NATURE_VARIANTS.keys()))


if __name__ == "__main__":
    print("🎵 Glorb server → http://localhost:5000")
    app.run(debug=False, port=5000)
