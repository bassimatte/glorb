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
