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
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
import main as bb

app = Flask(__name__)
CORS(app)  # allow GitHub Pages to call Render backend


_DITHER_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}

def _apply_tpdf_dither(audio, subtype):
    """TPDF dither: add triangular noise of amplitude 1 LSB before quantisation.
    Converts deterministic quantisation distortion into inaudible white noise.
    No-op for floating-point subtypes.
    """
    bits = _DITHER_BITS.get(subtype)
    if bits is None:
        return audio
    lsb = 2.0 / (2 ** bits)
    r1 = np.random.uniform(-0.5, 0.5, audio.shape).astype(np.float32)
    r2 = np.random.uniform(-0.5, 0.5, audio.shape).astype(np.float32)
    return audio + (r1 + r2) * lsb   # triangular PDF, peak amplitude = 1 LSB


def _normalize(audio):
    """Peak-normalize to -0.1 dBFS (0.9886 linear)."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9886
    return audio.astype(np.float32)


def _wav_bytes(audio, sample_rate, subtype):
    audio = _normalize(audio)
    audio = _apply_tpdf_dither(audio, subtype)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype=subtype, format="WAV")
    buf.seek(0)
    return buf


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/")
def index():
    return render_template("index.html")


# Server-side duration cap — set GLORB_MAX_DURATION=60 on Render free tier
_MAX_DURATION = float(os.environ.get("GLORB_MAX_DURATION", 300))


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.get_json(silent=True) or {}
    duration = float(data.get("duration", 10.0))
    quality  = data.get("quality", "high")
    mode     = data.get("mode", "glorb")
    variant  = data.get("variant", None)   # for nature mode

    # Knob params (0–100, default 50)
    bb.KNOB_ENERGY     = max(0, min(100, int(data.get("energy",     50))))
    bb.KNOB_BRIGHTNESS = max(0, min(100, int(data.get("brightness", 50))))
    bb.KNOB_CHAOS      = max(0, min(100, int(data.get("chaos",      50))))

    duration = max(1.0, min(duration, _MAX_DURATION))
    if quality not in bb.QUALITY_PRESETS:
        quality = "high"

    sample_rate, subtype, _ = bb.QUALITY_PRESETS[quality]
    bb.SAMPLE_RATE = sample_rate

    # ── UI Pack → concatenated WAV for playback ───────────────────
    if mode == "ui-pack":
        sounds = bb.make_ui_pack()
        silence_gap = bb.silence(0.25)
        parts = []
        for audio in sounds.values():
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9886
            parts.append(audio)
            parts.append(silence_gap)
        combined = np.concatenate(parts[:-1])  # drop trailing gap
        buf = _wav_bytes(combined, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_ui_pack_preview.wav")

    # ── Nature → single WAV ───────────────────────────────────────
    if mode == "nature":
        audio, used_variant = bb.make_nature_sequence(duration, variant)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False,
                         download_name=f"glorb_nature_{used_variant}.wav")

    # ── Sci-Fi → single WAV ───────────────────────────────────────
    if mode == "scifi":
        audio = bb.make_scifi_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_scifi.wav")

    # ── Haptic → single WAV ───────────────────────────────────────
    if mode == "haptic":
        audio = bb.make_haptic_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_haptic.wav")

    # ── Radio → single WAV ────────────────────────────────────────
    if mode == "radio":
        audio = bb.make_radio_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_radio.wav")

    # ── Retro → single WAV ────────────────────────────────────────
    if mode == "retro":
        audio = bb.make_retro_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_retro.wav")

    if mode == "foley":
        audio = bb.make_foley_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_foley.wav")

    if mode == "underwater":
        audio = bb.make_underwater_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_underwater.wav")

    if mode == "weather":
        audio = bb.make_weather_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_weather.wav")

    if mode == "bell":
        audio = bb.make_bell_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_bell.wav")

    if mode == "bass":
        audio = bb.make_bass_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_bass.wav")

    if mode == "glitch":
        audio = bb.make_glitch_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_glitch.wav")

    if mode == "pinball":
        audio = bb.make_pinball_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_pinball.wav")

    if mode == "horror":
        audio = bb.make_horror_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_horror.wav")

    if mode == "granular":
        audio = bb.make_granular_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_granular.wav")

    if mode == "lofi":
        audio = bb.make_lofi_sequence(duration)
        buf = _wav_bytes(audio, sample_rate, subtype)
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name="glorb_lofi.wav")

    # ── Glorb (default) → single WAV ─────────────────────────────
    import random
    parts, total = [], 0.0
    gap_lo = bb._knob(bb.KNOB_ENERGY, 0.22, 0.003)   # high energy → tiny gaps
    gap_hi = bb._knob(bb.KNOB_ENERGY, 0.80, 0.06)
    while total < duration:
        signal, _style, _freq = bb.make_blip()
        gap = random.uniform(gap_lo, gap_hi)
        parts.append(signal)
        parts.append(bb.silence(gap))
        total += len(signal) / bb.SAMPLE_RATE + gap

    audio = bb._finalise(np.concatenate(parts), duration)
    buf = _wav_bytes(audio, sample_rate, subtype)
    return send_file(buf, mimetype="audio/wav",
                     as_attachment=False, download_name="glorb.wav")


@app.route("/ui-pack-zip")
def ui_pack_zip():
    """Return all UI Pack sounds as a ZIP of individual WAV files."""
    quality = request.args.get("quality", "high")
    if quality not in bb.QUALITY_PRESETS:
        quality = "high"
    sample_rate, subtype, _ = bb.QUALITY_PRESETS[quality]
    bb.SAMPLE_RATE = sample_rate

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


@app.route("/nature-variants")
def nature_variants():
    from flask import jsonify
    return jsonify(list(bb._NATURE_VARIANTS.keys()))


if __name__ == "__main__":
    print("🎵 Glorb server → http://localhost:5000")
    app.run(debug=False, port=5000)
