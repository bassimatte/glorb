"""
blipblop.py — A random blip-blop sound generator.
Modelled after reference recordings: short electronic blips (5-95ms),
mid-high pitched (120-1764Hz), with metallic texture and reverb/echo effects.

Usage:
    python blipblop.py                        # 10s, high quality → blipblop.wav
    python blipblop.py -d 30                  # 30s output
    python blipblop.py -d 60 -o my.wav        # custom filename
    python blipblop.py -d 10 --play           # also play after rendering
    python blipblop.py -q standard            # 44100 Hz / 16-bit
    python blipblop.py -q high                # 44100 Hz / 24-bit  (default)
    python blipblop.py -q studio              # 48000 Hz / 24-bit
    python blipblop.py -q float               # 44100 Hz / 32-bit float
"""

import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import random

# Quality presets: (sample_rate, sf_subtype, label)
QUALITY_PRESETS = {
    "standard": (44100, "PCM_16", "44100 Hz / 16-bit PCM"),
    "high":     (44100, "PCM_24", "44100 Hz / 24-bit PCM"),
    "studio":   (48000, "PCM_24", "48000 Hz / 24-bit PCM"),
    "float":    (44100, "FLOAT",  "44100 Hz / 32-bit float"),
}

SAMPLE_RATE = 44100  # may be overridden by --quality preset


# ---------------------------------------------------------------------------
# Waveform generators
# ---------------------------------------------------------------------------

def _t(duration):
    return np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)


def sine_wave(freq, duration):
    return np.sin(2 * np.pi * freq * _t(duration))


def fm_metallic(freq, duration):
    """FM synthesis — carrier modulated by itself for a metallic ring."""
    t = _t(duration)
    mod_index = random.uniform(1.5, 4.0)
    mod_ratio = random.choice([1.41, 2.0, 2.5, 3.0])
    modulator = mod_index * np.sin(2 * np.pi * freq * mod_ratio * t)
    return np.sin(2 * np.pi * freq * t + modulator)


def additive_harmonics(freq, duration):
    """Sum of harmonics with decaying amplitudes — bright, bell-like tone."""
    t = _t(duration)
    n_harmonics = random.randint(3, 7)
    signal = np.zeros(len(t))
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / (k ** random.uniform(0.8, 1.5))
        signal += amp * np.sin(2 * np.pi * freq * k * t)
    return signal / np.max(np.abs(signal) + 1e-9)


def chirp(freq, duration):
    """Frequency sweep — classic blip chirp sound."""
    t = _t(duration)
    freq2 = freq * random.choice([0.5, 1.5, 2.0, 0.75, 3.0])
    phase = 2 * np.pi * (freq * t + (freq2 - freq) / (2 * duration) * t ** 2)
    return np.sin(phase)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

def apply_envelope(signal):
    """Fast attack, quick exponential decay — short percussive blip shape."""
    n = len(signal)
    attack_samples = max(1, int(random.uniform(0.001, 0.005) * SAMPLE_RATE))
    decay = np.exp(-np.linspace(0, random.uniform(20, 80), n - attack_samples))
    env = np.concatenate([np.linspace(0, 1, attack_samples), decay])
    return signal * env[:n]


def apply_reverb(signal, decay=0.3, num_echoes=4):
    """Simple comb-filter reverb using delayed copies."""
    out = signal.copy().astype(np.float64)
    for i in range(1, num_echoes + 1):
        delay_samples = int(random.uniform(0.015, 0.06) * SAMPLE_RATE * i)
        amp = decay ** i
        if delay_samples < len(out):
            out[delay_samples:] += amp * signal[:len(out) - delay_samples]
    peak = np.max(np.abs(out))
    return (out / peak * 0.8) if peak > 0 else out


def apply_echo(signal, delay=0.08, feedback=0.4):
    """Single echo with feedback."""
    delay_samples = int(delay * SAMPLE_RATE)
    out = np.zeros(len(signal) + delay_samples * 3)
    out[:len(signal)] += signal
    for i in range(1, 4):
        start = delay_samples * i
        amp = feedback ** i
        out[start:start + len(signal)] += amp * signal
    peak = np.max(np.abs(out))
    return (out / peak * 0.8) if peak > 0 else out


# ---------------------------------------------------------------------------
# Organic processing — makes sounds feel alive and analog
# ---------------------------------------------------------------------------

def pitch_wobble(signal, rate_hz=6.0, depth=0.003):
    """Subtle pitch vibrato via phase modulation — removes the 'frozen' feel."""
    n = len(signal)
    t = np.arange(n) / SAMPLE_RATE
    lfo = depth * np.sin(2 * np.pi * rate_hz * t + random.uniform(0, 2 * np.pi))
    # Resample via fractional delay using linear interpolation
    indices = np.arange(n) + lfo * SAMPLE_RATE
    indices = np.clip(indices, 0, n - 1)
    lo = indices.astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = indices - lo
    return signal[lo] * (1 - frac) + signal[hi] * frac


def soft_saturate(signal, drive=1.8):
    """Tanh waveshaper — analog warmth, rounds off harsh transients."""
    return np.tanh(signal * drive) / np.tanh(drive)


def add_noise(signal, amount=0.004):
    """Tiny noise floor — mimics analog circuit hiss and adds 'air'."""
    return signal + amount * np.random.randn(len(signal))


def to_stereo(signal, haas_ms=None):
    """
    Stereo widening via Haas effect:
    L = dry signal, R = very slightly delayed + tiny pitch/level difference.
    haas_ms: delay in ms (random 1–12ms if None).
    Returns (N, 2) float32 array.
    """
    if haas_ms is None:
        haas_ms = random.uniform(1.0, 12.0)
    delay_samples = int(haas_ms / 1000 * SAMPLE_RATE)
    level_r = random.uniform(0.85, 0.98)  # slight level difference L vs R

    L = signal.copy()
    R = np.zeros(len(signal))
    if delay_samples < len(signal):
        R[delay_samples:] = signal[:len(signal) - delay_samples] * level_r
    else:
        R = signal * level_r

    # Swap L/R randomly so stereo image is not always the same side
    if random.random() < 0.5:
        L, R = R, L

    return np.stack([L, R], axis=1).astype(np.float32)




STYLES = ["metallic", "echo", "plain"]


def make_blip():
    """Synthesize a single blip/blop matching the reference characteristics."""
    # Frequency: 120–1764Hz, weighted toward 600–1000Hz
    freq = random.choices(
        [random.uniform(120, 400),
         random.uniform(400, 900),
         random.uniform(900, 1764)],
        weights=[1, 3, 2],
    )[0]

    # Duration: 5–95ms (reference mean ~20ms)
    duration = random.uniform(0.005, 0.095)

    # Pick waveform
    waveform_fn = random.choices(
        [sine_wave, fm_metallic, additive_harmonics, chirp],
        weights=[2, 3, 2, 3],
    )[0]

    signal = waveform_fn(freq, duration)
    signal = apply_envelope(signal)

    # Style: metallic keeps dry signal with harmonics; echo adds tail
    style = random.choices(STYLES, weights=[3, 3, 2])[0]
    if style == "metallic":
        # Add a subtle ring/metallic layer
        ring_freq = freq * random.choice([1.41, 2.73, 3.14])
        ring = sine_wave(ring_freq, duration)
        ring = apply_envelope(ring)
        signal = 0.7 * signal + 0.3 * ring[:len(signal)]
    elif style == "echo":
        delay = random.uniform(0.04, 0.12)
        feedback = random.uniform(0.25, 0.5)
        signal = apply_echo(signal, delay=delay, feedback=feedback)

    # Always add a touch of reverb
    if random.random() < 0.6:
        signal = apply_reverb(signal, decay=random.uniform(0.2, 0.5), num_echoes=random.randint(2, 5))

    # Organic processing chain
    signal = pitch_wobble(signal, rate_hz=random.uniform(4.0, 10.0), depth=random.uniform(0.001, 0.005))
    signal = soft_saturate(signal, drive=random.uniform(1.2, 2.5))
    signal = add_noise(signal, amount=random.uniform(0.002, 0.006))

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    # Stereo widening
    stereo = to_stereo(signal)
    return stereo, style, round(freq)


# ---------------------------------------------------------------------------
# Sequence — fill a target duration with random blip-blops
# ---------------------------------------------------------------------------

def silence(duration):
    return np.zeros((int(SAMPLE_RATE * duration), 2), dtype=np.float32)


def make_sequence(target_duration=10.0, output_file="blipblop.wav", play=False, quality="high"):
    """Concatenate random blip-blops until target_duration is reached, then save to WAV."""
    global SAMPLE_RATE
    sample_rate, subtype, quality_label = QUALITY_PRESETS[quality]
    SAMPLE_RATE = sample_rate  # propagate to all synthesis functions

    print(f"🎵 Rendering {target_duration:.1f}s of blip-blops → {output_file}")
    print(f"   Quality: {quality_label}\n")

    parts = []
    total = 0.0
    i = 0

    while total < target_duration:
        signal, style, freq = make_blip()
        label = random.choice(["blip", "blop", "bleep", "bloop", "zap", "pip", "click", "tok"])
        gap = random.uniform(0.005, 0.225)
        dur = len(signal) / SAMPLE_RATE
        i += 1
        print(f"  {i:3}. {label:<6}  [{style:<8}]  {freq}Hz  gap={gap*1000:.0f}ms")
        parts.append(signal)
        parts.append(silence(gap))
        total += dur + gap

    audio = np.concatenate(parts)

    # Trim to exact target duration
    target_samples = int(target_duration * SAMPLE_RATE)
    audio = audio[:target_samples]

    # Pad with silence if the very first blip was longer than target
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    # Normalize to -0.1 dBFS (peak normalization)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9886  # -0.1 dBFS

    sf.write(output_file, audio, SAMPLE_RATE, subtype=subtype)
    print(f"\n💾 Saved {target_duration:.1f}s to '{output_file}'  [{quality_label}, normalized to -0.1 dBFS]")

    if play:
        print("▶  Playing...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a WAV file of random blip-blop sounds.")
    parser.add_argument("-d", "--duration", type=float, default=10.0,
                        help="Duration of the output in seconds (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="blipblop.wav",
                        help="Output WAV filename (default: blipblop.wav)")
    parser.add_argument("-q", "--quality", choices=QUALITY_PRESETS.keys(), default="high",
                        help="Output quality preset (default: high = 44100Hz/24-bit)")
    parser.add_argument("--play", action="store_true",
                        help="Play the audio after rendering")
    args = parser.parse_args()

    make_sequence(target_duration=args.duration, output_file=args.output, play=args.play, quality=args.quality)


# =============================================================================
# RETRO MODE — 8-bit / chiptune synthesis
# =============================================================================

# C major pentatonic across 3 octaves (Hz)
_PENTA = [130.8, 146.8, 164.8, 196.0, 220.0,
          261.6, 293.7, 329.6, 392.0, 440.0,
          523.3, 587.3, 659.3, 784.0, 880.0]


def _square(freq, duration, duty=0.5):
    t = _t(duration)
    return np.where(np.sin(2 * np.pi * freq * t) >= np.cos(np.pi * duty), 1.0, -1.0).astype(np.float64)


def _pulse_env(n, hold_frac=0.4):
    """Hard attack, stepped decay — classic chiptune envelope."""
    a = max(1, int(0.002 * SAMPLE_RATE))
    h = int(n * hold_frac)
    d = max(1, n - a - h)
    return np.concatenate([
        np.linspace(0, 1, a),
        np.ones(h),
        np.linspace(1, 0, d),
    ])[:n]


def _retro_arpeggio(freqs, note_dur=0.04):
    """Rapid arpeggio across a list of frequencies."""
    parts = []
    for f in freqs:
        s = _square(f, note_dur, duty=random.choice([0.25, 0.5]))
        s = s * _pulse_env(len(s), 0.5)
        parts.append(s)
    sig = np.concatenate(parts)
    return to_stereo(sig / (np.max(np.abs(sig)) + 1e-9) * 0.7)


# Named retro sound recipes
def _retro_coin():
    freqs = [random.choice(_PENTA[6:10]), random.choice(_PENTA[9:13])]
    return _retro_arpeggio(freqs, note_dur=0.06)

def _retro_jump():
    f0 = random.choice(_PENTA[3:7])
    f1 = f0 * 2.0
    t = _t(0.18)
    freq_curve = f0 * (f1 / f0) ** (t / 0.18)
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s = np.sign(np.sin(phase))
    s = s * _pulse_env(len(s), 0.3)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)

def _retro_powerup():
    freqs = [_PENTA[i] for i in sorted(random.sample(range(len(_PENTA)), 5))]
    return _retro_arpeggio(freqs, note_dur=0.045)

def _retro_levelup():
    freqs = [_PENTA[i] for i in [4, 6, 8, 10, 13]]
    return _retro_arpeggio(freqs, note_dur=0.07)

def _retro_death():
    freqs = [_PENTA[i] for i in sorted(random.sample(range(6, 14), 5), reverse=True)]
    return _retro_arpeggio(freqs, note_dur=0.06)

def _retro_error():
    f = random.choice(_PENTA[:4])
    s = _square(f, 0.12, duty=0.25)
    s = s * _pulse_env(len(s), 0.1)
    # add dissonant layer
    s2 = _square(f * 1.06, 0.12, duty=0.25) * 0.5
    s = (s + s2[:len(s)]) * 0.5
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)

def _retro_blip():
    f = random.choice(_PENTA)
    dur = random.uniform(0.03, 0.08)
    s = _square(f, dur, duty=random.choice([0.25, 0.5]))
    s = s * _pulse_env(len(s))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)

_RETRO_SOUNDS = [
    ("blip",    _retro_blip,    6),
    ("coin",    _retro_coin,    3),
    ("jump",    _retro_jump,    2),
    ("powerup", _retro_powerup, 1),
    ("levelup", _retro_levelup, 1),
    ("death",   _retro_death,   1),
    ("error",   _retro_error,   2),
]


def make_retro_sequence(target_duration=10.0):
    names, fns, weights = zip(*_RETRO_SOUNDS)
    parts, total = [], 0.0
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.04, 0.25)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    return _finalise(np.concatenate(parts), target_duration)


# =============================================================================
# NATURE MODE — procedural ambient textures
# =============================================================================

def _bandpass(signal, low_hz, high_hz, order=2):
    from scipy.signal import butter, sosfilt
    nyq = SAMPLE_RATE / 2
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return sosfilt(sos, signal)

def _lowpass(signal, cutoff_hz, order=2):
    from scipy.signal import butter, sosfilt
    nyq = SAMPLE_RATE / 2
    sos = butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return sosfilt(sos, signal)


def _rain_drop():
    """Single raindrop: filtered noise burst."""
    dur = random.uniform(0.008, 0.04)
    n = int(SAMPLE_RATE * dur)
    noise = np.random.randn(n)
    noise = _bandpass(noise, random.uniform(800, 2000), random.uniform(4000, 8000))
    env = np.exp(-np.linspace(0, random.uniform(8, 25), n))
    return (noise * env * random.uniform(0.3, 1.0)).astype(np.float32)


def _make_rain(target_duration):
    """Dense rain: many drops at random times with stereo panning."""
    total_samples = int(target_duration * SAMPLE_RATE)
    L = np.zeros(total_samples)
    R = np.zeros(total_samples)
    t = 0
    while t < total_samples:
        drop = _rain_drop()
        pan = random.random()  # 0=left 1=right
        end = min(t + len(drop), total_samples)
        L[t:end] += drop[:end-t] * (1 - pan)
        R[t:end] += drop[:end-t] * pan
        gap = int(random.uniform(0.001, 0.02) * SAMPLE_RATE)
        t += gap
    stereo = np.stack([L, R], axis=1).astype(np.float32)
    return stereo


def _make_fire(target_duration):
    """Fire: low rumble + random high-freq crackles."""
    total_samples = int(target_duration * SAMPLE_RATE)
    # Base rumble: low-pass noise with slow amplitude modulation
    noise = np.random.randn(total_samples)
    rumble = _lowpass(noise, 200).astype(np.float32)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * random.uniform(0.3, 1.2) *
                               np.arange(total_samples) / SAMPLE_RATE)
    rumble = (rumble * lfo * 0.5).astype(np.float32)

    # Crackles: random sharp transients
    crackle = np.zeros(total_samples, dtype=np.float32)
    t = 0
    while t < total_samples:
        dur = random.uniform(0.002, 0.015)
        n = int(dur * SAMPLE_RATE)
        burst = np.random.randn(n) * random.uniform(0.1, 0.8)
        burst = _bandpass(burst, 1000, 8000).astype(np.float32)
        env = np.exp(-np.linspace(0, 20, n))
        end = min(t + n, total_samples)
        crackle[t:end] += (burst * env)[:end-t]
        t += int(random.uniform(0.005, 0.06) * SAMPLE_RATE)

    mono = rumble + crackle * 0.6
    # Slight stereo difference
    L = mono + np.random.randn(total_samples).astype(np.float32) * 0.01
    R = mono + np.random.randn(total_samples).astype(np.float32) * 0.01
    return np.stack([L, R], axis=1).astype(np.float32)


def _make_insects(target_duration):
    """Insects at night: dense high-freq chirp bursts."""
    total_samples = int(target_duration * SAMPLE_RATE)
    L = np.zeros(total_samples)
    R = np.zeros(total_samples)
    t = 0
    while t < total_samples:
        # Each chirp: short sine burst 2–8kHz
        freq = random.uniform(2000, 8000)
        dur = random.uniform(0.005, 0.04)
        n = int(dur * SAMPLE_RATE)
        s = np.sin(2 * np.pi * freq * np.arange(n) / SAMPLE_RATE)
        env = np.exp(-np.linspace(0, 15, n))
        chirp_s = (s * env * random.uniform(0.1, 0.5)).astype(np.float32)
        pan = random.random()
        end = min(t + n, total_samples)
        L[t:end] += chirp_s[:end-t] * (1 - pan)
        R[t:end] += chirp_s[:end-t] * pan
        t += int(random.uniform(0.002, 0.05) * SAMPLE_RATE)

    return np.stack([L.astype(np.float32), R.astype(np.float32)], axis=1)


_NATURE_VARIANTS = {
    "rain":    _make_rain,
    "fire":    _make_fire,
    "insects": _make_insects,
}


def make_nature_sequence(target_duration=10.0, variant=None):
    if variant is None or variant not in _NATURE_VARIANTS:
        variant = random.choice(list(_NATURE_VARIANTS.keys()))
    audio = _NATURE_VARIANTS[variant](target_duration)
    return _finalise(audio, target_duration), variant


# =============================================================================
# UI PACK MODE — purposeful interface sound effects
# =============================================================================

def _ui_sine_blip(freq, duration, fade_out=True):
    t = _t(duration)
    s = np.sin(2 * np.pi * freq * t)
    n = len(s)
    a = max(1, int(0.003 * SAMPLE_RATE))
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    if fade_out:
        env[a:] = np.exp(-np.linspace(0, 8, n - a))
    return s * env


def _ui_hover():
    s = _ui_sine_blip(random.uniform(1200, 1800), 0.015)
    s = add_noise(s, 0.002)
    return to_stereo(s * 0.5)

def _ui_click():
    freq = random.uniform(300, 600)
    s = _ui_sine_blip(freq, 0.025)
    # add a tiny thud
    thud = _lowpass(np.random.randn(int(0.01 * SAMPLE_RATE)), 300) * 0.4
    n = min(len(s), len(thud))
    s[:n] += thud[:n]
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)

def _ui_confirm():
    f0 = random.choice([523.3, 587.3, 659.3])
    f1 = f0 * 1.5
    s0 = _ui_sine_blip(f0, 0.09)
    s1 = _ui_sine_blip(f1, 0.09)
    gap = silence(0.04)[:, 0]
    s = np.concatenate([s0, gap, s1])
    return to_stereo(s * 0.65)

def _ui_error():
    f = random.choice([180, 200, 220])
    t = _t(0.15)
    # Buzz: square-ish with dissonance
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.5
    s += np.sign(np.sin(2 * np.pi * f * 1.08 * t)) * 0.3
    n = len(s)
    env = np.concatenate([np.linspace(0,1, int(0.005*SAMPLE_RATE)),
                           np.exp(-np.linspace(0, 4, n - int(0.005*SAMPLE_RATE)))])
    s = s * env[:n]
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.6)

def _ui_success():
    freqs = [523.3, 659.3, 784.0]
    parts = []
    for f in freqs:
        parts.append(_ui_sine_blip(f, 0.08))
        parts.append(np.zeros(int(0.025 * SAMPLE_RATE)))
    s = np.concatenate(parts)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)

def _ui_notification():
    f0 = random.choice([880, 1046, 1174])
    s0 = _ui_sine_blip(f0, 0.07)
    s1 = _ui_sine_blip(f0 * 1.25, 0.07)
    gap = np.zeros(int(0.03 * SAMPLE_RATE))
    s = np.concatenate([s0, gap, s1])
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.6)

def _ui_whoosh():
    n = int(0.18 * SAMPLE_RATE)
    noise = np.random.randn(n)
    # Sweep bandpass upward
    out = np.zeros(n)
    for i in range(n):
        progress = i / n
        flo = 200 + progress * 2000
        fhi = flo * 3
        # simple single-sample approximation: just weight noise by freq profile
        out[i] = noise[i] * np.exp(-((progress - 0.4) ** 2) / 0.1)
    out = _bandpass(out, 300, 4000)
    env = np.exp(-np.linspace(0, 6, n))
    s = (out * env).astype(np.float32)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.55)

UI_PACK_SOUNDS = {
    "hover":        _ui_hover,
    "click":        _ui_click,
    "confirm":      _ui_confirm,
    "error":        _ui_error,
    "success":      _ui_success,
    "notification": _ui_notification,
    "whoosh":       _ui_whoosh,
}


def make_ui_pack():
    """Generate all UI sounds. Returns dict of {name: stereo_array}."""
    return {name: fn() for name, fn in UI_PACK_SOUNDS.items()}


# =============================================================================
# Shared helpers
# =============================================================================

def _finalise(audio, target_duration):
    """Trim/pad to exact duration and peak-normalise."""
    target_samples = int(target_duration * SAMPLE_RATE)
    audio = audio[:target_samples]
    if len(audio) < target_samples:
        pad = target_samples - len(audio)
        audio = np.pad(audio, ((0, pad), (0, 0)))
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9886
    return audio.astype(np.float32)

