"""
server.py — Flask web server for Glorb.
Serves the UI and exposes /generate (WAV or ZIP) and /nature-variants endpoints.
"""

import gc
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
CORS(app)  # allow GitHub Pages to call the hosted backend

import threading
_render_sem = threading.Semaphore(1)  # one render at a time; prevents RAM spikes from concurrent requests
_MAX_WAIT   = 120                     # seconds to wait for a slot before returning 503


_DITHER_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}

def _apply_tpdf_dither(audio, subtype):
    """TPDF dither in-place: triangular noise of amplitude 1 LSB before quantisation.
    No-op for floating-point subtypes.
    """
    bits = _DITHER_BITS.get(subtype)
    if bits is None:
        return audio
    lsb = 2.0 / (2 ** bits)
    r1 = np.random.uniform(-0.5, 0.5, audio.shape).astype(np.float32)
    r2 = np.random.uniform(-0.5, 0.5, audio.shape).astype(np.float32)
    np.add(r1, r2, out=r1)
    np.multiply(r1, lsb, out=r1)
    np.add(audio, r1, out=audio)
    del r1, r2
    return audio


def _normalize(audio):
    """Peak-normalize to -0.1 dBFS in-place (0.9886 linear)."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        np.multiply(audio, 0.9886 / peak, out=audio)
    return audio


def _wav_bytes(audio, sample_rate, subtype):
    _normalize(audio)
    _apply_tpdf_dither(audio, subtype)
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


# Server-side duration cap for hosted deployments.
_MAX_DURATION = float(os.environ.get("GLORB_MAX_DURATION", 300))


@app.route("/generate", methods=["POST"])
def generate():
    data     = request.get_json(silent=True) or {}
    duration = float(data.get("duration", 10.0))
    quality  = data.get("quality", "high")
    # New architecture: playback (continuous/arp/groove) + preset (sound source)
    # Legacy fallback: if only 'mode' is sent treat it as playback=continuous, preset=mode
    playback = data.get("playback", "continuous")
    preset   = data.get("preset",   data.get("mode", "glorb"))
    variant  = data.get("variant",  None)   # for nature preset

    # Knob params (0–100, default 50)
    bb.KNOB_ENERGY     = max(0, min(100, int(data.get("energy",     50))))
    bb.KNOB_BRIGHTNESS = max(0, min(100, int(data.get("brightness", 50))))
    bb.KNOB_CHAOS      = max(0, min(100, int(data.get("chaos",      50))))

    # Playback-specific params
    bpm             = data.get("bpm",            None)
    groove_template = data.get("groove_template", "four-on-floor")
    arp_scale       = data.get("arp_scale",       None)
    arp_root        = data.get("arp_root",        None)
    arp_wave        = data.get("arp_wave",        None)

    duration = max(1.0, min(duration, _MAX_DURATION))
    if quality not in bb.QUALITY_PRESETS:
        quality = "high"

    sample_rate, subtype, _ = bb.QUALITY_PRESETS[quality]
    bb.SAMPLE_RATE = sample_rate

    def _finish(audio, name):
        """Encode audio to WAV bytes, then immediately free the numpy array."""
        buf = _wav_bytes(audio, sample_rate, subtype)
        del audio
        gc.collect()
        return send_file(buf, mimetype="audio/wav",
                         as_attachment=False, download_name=name)

    if not _render_sem.acquire(timeout=_MAX_WAIT):
        return {"error": "server busy — please wait a moment and try again"}, 503
    try:
        # ── Groove playback ───────────────────────────────────────────
        if playback == "groove":
            audio = bb.make_groove_sequence(
                duration,
                bpm=int(bpm or 90),
                template=groove_template,
                preset=preset,
            )
            return _finish(audio, f"glorb_groove_{preset}.wav")

        # ── Arp playback ──────────────────────────────────────────────
        if playback == "arp":
            audio = bb.make_arp_sequence(
                duration,
                bpm=bpm,
                scale=arp_scale,
                root=arp_root,
                wave=arp_wave,
                preset=preset,
            )
            return _finish(audio, f"glorb_arp_{preset}.wav")

        # ── Continuous playback (original behaviour) ──────────────────
        # UI Pack special case
        if preset == "ui-pack":
            sounds = bb.make_ui_pack()
            silence_gap = bb.silence(0.25)
            parts = []
            for audio in sounds.values():
                peak = np.max(np.abs(audio))
                if peak > 0:
                    np.multiply(audio, 0.9886 / peak, out=audio)
                parts.append(audio)
                parts.append(silence_gap)
            combined = np.concatenate(parts[:-1])
            del parts, sounds, silence_gap
            gc.collect()
            return _finish(combined, "glorb_ui_pack_preview.wav")

        _CONTINUOUS_FN = {
            "nature":    lambda: bb.make_nature_sequence(duration, variant)[0],
            "scifi":     lambda: bb.make_scifi_sequence(duration),
            "haptic":    lambda: bb.make_haptic_sequence(duration),
            "radio":     lambda: bb.make_radio_sequence(duration),
            "retro":     lambda: bb.make_retro_sequence(duration),
            "foley":     lambda: bb.make_foley_sequence(duration),
            "underwater":lambda: bb.make_underwater_sequence(duration),
            "weather":   lambda: bb.make_weather_sequence(duration),
            "bell":      lambda: bb.make_bell_sequence(duration),
            "bass":      lambda: bb.make_bass_sequence(duration),
            "glitch":    lambda: bb.make_glitch_sequence(duration),
            "pinball":   lambda: bb.make_pinball_sequence(duration),
            "horror":    lambda: bb.make_horror_sequence(duration),
            "granular":  lambda: bb.make_granular_sequence(duration),
            "lofi":      lambda: bb.make_lofi_sequence(duration),
            "modem":     lambda: bb.make_modem_sequence(duration),
            "insects":   lambda: bb.make_insects_sequence(duration),
            "gamelan":   lambda: bb.make_gamelan_sequence(duration),
            "glass":     lambda: bb.make_glass_sequence(duration),
            "clockwork": lambda: bb.make_clockwork_sequence(duration),
            "creature":  lambda: bb.make_creature_sequence(duration),
            "electricity":lambda: bb.make_electricity_sequence(duration),
            "cave":      lambda: bb.make_cave_sequence(duration),
            # legacy: if someone sends mode=arp or mode=groove, honour it
            "arp":       lambda: bb.make_arp_sequence(duration, bpm=bpm, scale=arp_scale, root=arp_root, wave=arp_wave),
            "groove":    lambda: bb.make_groove_sequence(duration, bpm=int(bpm or 90), template=groove_template),
        }

        if preset in _CONTINUOUS_FN:
            audio = _CONTINUOUS_FN[preset]()
            return _finish(audio, f"glorb_{preset}.wav")

        # ── Glorb (default) ───────────────────────────────────────────
        import random
        parts, total = [], 0.0
        gap_lo = bb._knob(bb.KNOB_ENERGY, 0.22, 0.003)
        gap_hi = bb._knob(bb.KNOB_ENERGY, 0.80, 0.06)
        while total < duration:
            signal, _style, _freq = bb.make_blip()
            gap = random.uniform(gap_lo, gap_hi)
            parts.append(signal)
            parts.append(bb.silence(gap))
            total += len(signal) / bb.SAMPLE_RATE + gap

        audio = bb._finalise(np.concatenate(parts), duration)
        del parts
        gc.collect()
        return _finish(audio, "glorb.wav")
    finally:
        _render_sem.release()


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
                np.multiply(audio, 0.9886 / peak, out=audio)
            wav = _wav_bytes(audio, sample_rate, subtype)
            zf.writestr(f"{name}.wav", wav.read())
    del sounds
    gc.collect()
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
