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
try:
    import sounddevice as sd
except ImportError:
    sd = None   # not available on the hosted server; playback disabled
import soundfile as sf
import random
from scipy.ndimage import maximum_filter1d

# Quality presets: (sample_rate, sf_subtype, label)
QUALITY_PRESETS = {
    "standard": (44100, "PCM_16", "44100 Hz / 16-bit PCM"),
    "high":     (44100, "PCM_24", "44100 Hz / 24-bit PCM"),
    "studio":   (48000, "PCM_24", "48000 Hz / 24-bit PCM"),
    "float":    (44100, "FLOAT",  "44100 Hz / 32-bit float"),
}

# Playback modes define how events are arranged; presets define their source.
PLAYBACK_MODES = {
    "continuous": "Free-running generative soundscape",
    "arp":        "Arpeggiated events with scale, root, waveform, and BPM controls",
    "groove":     "Rhythmically quantised events using groove templates",
}

SOUND_PRESETS = {
    "glorb":      "Abstract electronic blips and bloops",
    "retro":      "8-bit chiptune square and pulse waves",
    "nature":     "Rain, fire, insects, and forest textures",
    "scifi":      "Phasers, warp drives, and laser bursts",
    "haptic":     "Tactile vibration pulses and clicks",
    "radio":      "AM/FM static, morse, and transmission artefacts",
    "ui-pack":    "Short interface clicks, pops, and chimes",
    "foley":      "Footsteps, paper, cloth, and everyday impacts",
    "underwater": "Sonar pings, bubble trains, and deep pressure tones",
    "weather":    "Thunder, lightning crackle, and rain layers",
    "bell":       "Marimba, tubular bells, and resonant metal strikes",
    "bass":       "Deep sub-bass pulses and 808-style hits",
    "glitch":     "Buffer corruption, bit-crush, and digital stutter",
    "pinball":    "Flippers, bumpers, drain, and mechanical clicks",
    "horror":     "Dissonant drones, breath, reversed tails, and tension",
    "granular":   "Scattered grain clouds across noise and pitched sources",
    "lofi":       "Vinyl crackle, tape wow/flutter, and degraded bandwidth",
    "modem":      "DTMF tones, FSK handshakes, and dial-up screech",
    "insects":    "Crickets, cicadas, grasshoppers, and water bugs",
    "gamelan":    "Inharmonic metal bars, detuned pairs, and gong resonance",
    "glass":      "Brittle shards, singing rims, and crystalline resonances",
    "clockwork":  "Ticks, ratchets, springs, and tiny mechanisms",
    "creature":   "Imaginary chirps, calls, growls, and breath",
    "cat":        "Meows, mewls, trills, purrs, and hisses",
    "birds":      "Peeps, whistles, trills, warbles, and calls",
    "electricity":"Arcs, transformer hum, relays, and rising charge",
    "cave":       "Drips, stones, subterranean wind, and deep rumbles",
    "synth":      "Dedicated oscillator voice for Arp playback",
}


def print_available_modes_and_presets():
    """Print the playback-mode and sound-preset catalog."""
    print("Playback modes:")
    for name, description in PLAYBACK_MODES.items():
        print(f"  {name:<12} {description}")
    print("\nPresets:")
    for name, description in SOUND_PRESETS.items():
        print(f"  {name:<12} {description}")

SAMPLE_RATE = 44100  # may be overridden by --quality preset

# ---------------------------------------------------------------------------
# Knob parameters (set by server from UI sliders, 0–100 each)
# ---------------------------------------------------------------------------
KNOB_ENERGY     = 50   # sparse(0) ↔ dense(100) — gap size & event density
KNOB_BRIGHTNESS = 50   # dark(0) ↔ bright(100)  — frequency range bias
KNOB_CHAOS      = 50   # ordered(0) ↔ glitchy(100) — randomness & effect depth

def _knob(val, lo, hi):
    """Map a 0–100 knob value linearly to [lo, hi]."""
    return lo + (hi - lo) * (val / 100.0)


def _apply_knob_brightness(audio):
    """Low-shelf cut (dark) or high-shelf boost (bright) based on KNOB_BRIGHTNESS."""
    if abs(KNOB_BRIGHTNESS - 50) < 3:
        return audio
    from scipy.signal import butter, sosfilt
    nyq = SAMPLE_RATE / 2
    stereo = audio.ndim == 2
    if KNOB_BRIGHTNESS < 50:
        # Low-pass: brightness=0 → cutoff ~600 Hz, brightness=50 → near-flat
        cutoff = max(200, _knob(KNOB_BRIGHTNESS, 600, nyq * 0.98))
        sos = butter(2, cutoff / nyq, btype='low', output='sos')
        if stereo:
            return np.stack([sosfilt(sos, audio[:, 0]),
                             sosfilt(sos, audio[:, 1])], axis=1).astype(np.float32)
        return sosfilt(sos, audio).astype(np.float32)
    else:
        # High-shelf add: brightness=50 → 0 boost, brightness=100 → +55 % high content
        boost = _knob(KNOB_BRIGHTNESS, 0.0, 0.55)
        sos = butter(2, min(3000 / nyq, 0.99), btype='high', output='sos')
        if stereo:
            return np.stack([audio[:, 0] + boost * sosfilt(sos, audio[:, 0]),
                             audio[:, 1] + boost * sosfilt(sos, audio[:, 1])], axis=1).astype(np.float32)
        return (audio + boost * sosfilt(sos, audio)).astype(np.float32)


_NOISE_MODES = {'nature', 'weather', 'underwater', 'granular', 'lofi'}

def _apply_knob_chaos(audio, mode='glorb'):
    """Tanh drive + pitch wobble based on KNOB_CHAOS.
    Pitch wobble is skipped for noise-dominated modes — broadband noise has no pitch
    to wobble, so the operation wastes ~484 MB RAM with zero audible effect.
    """
    if KNOB_CHAOS < 5:
        return audio
    drive = _knob(KNOB_CHAOS, 1.0, 4.0)
    audio = soft_saturate(audio, drive)
    if KNOB_CHAOS > 55 and mode not in _NOISE_MODES:
        depth = _knob(KNOB_CHAOS, 0.0, 0.007)
        rate  = _knob(KNOB_CHAOS, 0.2, 2.5)
        if audio.ndim == 2:
            L = pitch_wobble(audio[:, 0], rate_hz=rate, depth=depth)
            R = pitch_wobble(audio[:, 1], rate_hz=rate, depth=depth)
            audio = np.stack([L, R], axis=1)
        else:
            audio = pitch_wobble(audio, rate_hz=rate, depth=depth)
    return audio.astype(np.float32)


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
        if freq * k >= SAMPLE_RATE / 2:
            break  # stop before Nyquist to prevent aliasing
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
    """Schroeder reverb: 4 fixed-prime-delay parallel combs + 2 allpass diffusers.

    Deterministic prime delays (vs old random delays) eliminate flutter echo
    and produce smooth, dense reverb tails.  'decay' maps to room size /
    feedback gain; 'num_echoes' is accepted for API compatibility but unused.
    Uses FIR approximation via fftconvolve for vectorised performance.
    """
    from scipy.signal import fftconvolve

    sig = np.asarray(signal, dtype=np.float32)
    n   = len(sig)
    if n == 0:
        return sig

    # Map legacy 'decay' [0.15–0.50] → comb feedback gain and wet-mix level
    g_c = float(np.clip(0.70 + 0.36 * decay, 0.70, 0.94))
    wet = float(np.clip(0.20 + 0.40 * decay, 0.23, 0.40))

    # Fixed prime comb delays (≈29.6 / 37.1 / 41.1 / 43.8 ms at 44 100 Hz)
    COMB_D = (1307, 1637, 1811, 1931)
    # Allpass diffuser delays (≈10 ms / 5 ms at 44 100 Hz)
    AP_D   = (441, 221)
    g_a    = 0.70

    # FIR tail length: enough terms so g_c^N < −60 dB
    n_terms = int(np.ceil(-3.0 / np.log10(g_c + 1e-12)))
    n_terms = min(max(n_terms, 16), 96)

    # ---- 4 parallel comb filters ----------------------------------------
    # Impulse response: h[kD] = g_c^k  (k = 0, 1, 2, …, n_terms)
    comb_mix = np.zeros(n, dtype=np.float32)
    for D in COMB_D:
        h = np.zeros(D * n_terms + 1, dtype=np.float32)
        for k in range(n_terms + 1):
            h[k * D] = g_c ** k
        comb_mix += fftconvolve(sig, h)[:n].astype(np.float32)
        del h
    comb_mix *= np.float32(0.25)

    # ---- 2 series allpass diffusers -------------------------------------
    # Impulse response: h[0] = -g_a,  h[kD] = g_a^(k-1)*(1-g_a²)  k ≥ 1
    ap    = comb_mix
    scale = float(1.0 - g_a * g_a)
    for D in AP_D:
        h = np.zeros(D * n_terms + 1, dtype=np.float32)
        h[0] = -g_a
        for k in range(1, n_terms + 1):
            h[k * D] = (g_a ** (k - 1)) * scale
        ap = fftconvolve(ap, h)[:n].astype(np.float32)
        del h

    out  = sig * np.float32(1.0 - wet) + ap * np.float32(wet)
    peak = float(np.max(np.abs(out)))
    return (out * np.float32(0.8 / peak)) if peak > 1e-6 else out


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
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    lfo = (depth * np.sin(2 * np.pi * rate_hz * t + random.uniform(0, 2 * np.pi))).astype(np.float32)
    del t
    # Resample via fractional delay using linear interpolation
    indices = np.arange(n, dtype=np.float32)
    indices += lfo * SAMPLE_RATE                # in-place
    del lfo
    np.clip(indices, 0, n - 1, out=indices)     # in-place
    lo   = indices.astype(np.int32)             # int32 = half the size of int64
    frac = indices - lo
    del indices
    hi = np.clip(lo + 1, 0, n - 1)
    out = signal[lo] * (1 - frac) + signal[hi] * frac
    del lo, hi, frac
    return out.astype(np.float32)


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
    """Synthesize a single blip/blop — shaped by KNOB_ENERGY, KNOB_BRIGHTNESS, KNOB_CHAOS."""
    # BRIGHTNESS: low → darker frequencies, high → brighter
    freq_lo = _knob(KNOB_BRIGHTNESS, 80,  400)
    freq_hi = _knob(KNOB_BRIGHTNESS, 400, 3500)
    freq = random.choices(
        [random.uniform(freq_lo, freq_lo * 2.5),
         random.uniform(freq_lo * 2.5, freq_hi * 0.7),
         random.uniform(freq_hi * 0.7, freq_hi)],
        weights=[1, 3, 2],
    )[0]

    # ENERGY: low → shorter blips, high → longer blips
    dur_lo = _knob(KNOB_ENERGY, 0.004, 0.04)
    dur_hi = _knob(KNOB_ENERGY, 0.02,  0.18)
    duration = random.uniform(dur_lo, dur_hi)

    # Pick waveform — CHAOS shifts toward more complex waveforms
    chaos_w = KNOB_CHAOS / 100.0
    waveform_fn = random.choices(
        [sine_wave, fm_metallic, additive_harmonics, chirp],
        weights=[max(0.3, 2 - chaos_w * 1.5), 3, 2, max(1, 1 + chaos_w * 3)],
    )[0]

    # 1. Generation
    signal = waveform_fn(freq, duration)

    # Style selection (metallic combines raw waveforms before saturation)
    style = random.choices(STYLES, weights=[3, 3, 2])[0]
    if style == "metallic":
        ring_freq = freq * random.choice([1.41, 2.73, 3.14])
        ring = sine_wave(ring_freq, duration)
        signal = 0.7 * signal + 0.3 * ring[:len(signal)]

    # 2. Saturation — distorts the source; reverb/echo process the already-distorted signal
    drive = _knob(KNOB_CHAOS, 1.1, 3.5)
    signal = soft_saturate(signal, drive=drive)

    # 3. Envelope — shape after saturation so the decay isn't re-distorted
    signal = apply_envelope(signal)

    # 4. Pitch wobble — before reverb; wobble AM is not fed into saturation
    wobble_depth = _knob(KNOB_CHAOS, 0.001, 0.012)
    signal = pitch_wobble(signal, rate_hz=random.uniform(4.0, 10.0), depth=wobble_depth)

    # 5. Echo / Reverb — spatial effects on the shaped, wobbled source
    if style == "echo":
        delay = random.uniform(0.04, 0.12)
        feedback = random.uniform(0.25, 0.5)
        signal = apply_echo(signal, delay=delay, feedback=feedback)

    if random.random() < 0.6:
        signal = apply_reverb(signal, decay=random.uniform(0.2, 0.5), num_echoes=random.randint(2, 5))

    # 6. Normalize — establish a consistent peak before adding noise
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    # 7. Noise — after normalize so level is relative to signal, not raw amplitude
    noise_amount = _knob(KNOB_CHAOS, 0.001, 0.015)
    signal = add_noise(signal, amount=noise_amount)

    # 8. Stereo — last stage
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
    del parts

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
        if sd is None:
            print("⚠  sounddevice not available — skipping playback.")
        else:
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
    parser.add_argument("--list", action="store_true",
                        help="List available playback modes and sound presets, then exit")
    args = parser.parse_args()

    if args.list:
        print_available_modes_and_presets()
        raise SystemExit(0)

    make_sequence(target_duration=args.duration, output_file=args.output, play=args.play, quality=args.quality)


# =============================================================================
# RETRO MODE — 8-bit / chiptune synthesis
# =============================================================================

# C major pentatonic across 3 octaves (Hz)
_PENTA = [130.8, 146.8, 164.8, 196.0, 220.0,
          261.6, 293.7, 329.6, 392.0, 440.0,
          523.3, 587.3, 659.3, 784.0, 880.0]


def _square(freq, duration, duty=0.5):
    """Bandlimited pulse/square wave via truncated Fourier series.
    Eliminates aliasing by summing only harmonics below the Nyquist limit.
    Fourier coefficients for a bipolar pulse with duty cycle D:
        a_k = 2*sin(2*pi*k*D) / (pi*k)
        b_k = 2*(1 - cos(2*pi*k*D)) / (pi*k)
        DC  = 2*D - 1
    """
    t = _t(duration)
    max_h = min(500, max(1, int(SAMPLE_RATE / (2.0 * max(float(freq), 1.0)))))
    ks = np.arange(1, max_h + 1, dtype=np.float64)
    two_pi_k_D = 2.0 * np.pi * ks * duty
    ak = 2.0 * np.sin(two_pi_k_D) / (np.pi * ks)
    bk = 2.0 * (1.0 - np.cos(two_pi_k_D)) / (np.pi * ks)
    # Process harmonics in chunks to avoid a (max_h × N) matrix in memory
    CHUNK_H = 50
    signal = np.full(len(t), float(2 * duty - 1))
    for start in range(0, max_h, CHUNK_H):
        k_chunk = ks[start:start + CHUNK_H]
        ph = 2.0 * np.pi * freq * np.outer(k_chunk, t)  # (≤50, N) — 10× smaller
        signal += np.dot(ak[start:start + CHUNK_H], np.cos(ph))
        signal += np.dot(bk[start:start + CHUNK_H], np.sin(ph))
        del ph
    peak = np.max(np.abs(signal)) + 1e-9
    return (signal / peak).astype(np.float64)


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
    sig = np.concatenate(parts); del parts
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
    # Use bandlimited _square() at mean frequency to avoid aliasing.
    # np.sign(np.sin()) produces infinite odd harmonics above Nyquist.
    f_mean = float(np.mean(freq_curve))
    s = _square(f_mean, 0.18, duty=0.5)
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
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.04 * gap_scale, 0.25 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


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
    return _finalise(audio, target_duration, mode='nature'), variant


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
    n = int(0.15 * SAMPLE_RATE)
    # Bandlimited square + detuned square for dissonance
    s = _square(f,        0.15, duty=0.5) * 0.5
    s += _square(f * 1.08, 0.15, duty=0.5) * 0.3
    env = np.concatenate([np.linspace(0, 1, int(0.005 * SAMPLE_RATE)),
                          np.exp(-np.linspace(0, 4, n - int(0.005 * SAMPLE_RATE)))])
    s = s * env[:n]
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.6)

def _ui_success():
    freqs = [523.3, 659.3, 784.0]
    parts = []
    for f in freqs:
        parts.append(_ui_sine_blip(f, 0.08))
        parts.append(np.zeros(int(0.025 * SAMPLE_RATE)))
    s = np.concatenate(parts); del parts
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
# SCI-FI MODE — laser zaps, shields, warps, pings, teleporters
# =============================================================================

def _scifi_laser():
    """Downward FM chirp — classic laser shot."""
    dur = random.uniform(0.04, 0.12)
    f_start = random.uniform(1200, 3000)
    f_end   = f_start * random.uniform(0.15, 0.45)
    t = _t(dur)
    freq_curve = f_start * (f_end / f_start) ** (t / dur)
    mod = random.uniform(2.0, 5.0) * np.sin(2 * np.pi * freq_curve * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(freq_curve + mod * freq_curve * 0.1) / SAMPLE_RATE
    s = np.sin(phase)
    env = np.exp(-np.linspace(0, random.uniform(15, 40), len(t)))
    s = soft_saturate(s * env, 1.5)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.75)


def _scifi_shield():
    """Rising chirp + shimmer + reverb — energy shield hit."""
    dur = random.uniform(0.08, 0.18)
    f0  = random.uniform(400, 800)
    f1  = f0 * random.uniform(1.5, 3.0)
    t   = _t(dur)
    freq_curve = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s = np.sin(phase) + 0.3 * np.sin(phase * 2.5)
    n = len(s)
    a = max(1, int(0.005 * SAMPLE_RATE))
    env = np.concatenate([np.linspace(0, 1, a),
                          np.exp(-np.linspace(0, 10, n - a))])[:n]
    s = apply_reverb(s * env, decay=0.35, num_echoes=3)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _scifi_warp():
    """AM tremolo + pitch sweep + echo — warp drive."""
    dur = random.uniform(0.2, 0.5)
    f0  = random.uniform(200, 600)
    f1  = f0 * random.choice([2.0, 3.0, 0.5])
    t   = _t(dur)
    freq_curve = f0 * (f1 / f0) ** (t / dur)
    phase  = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    am     = 0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(15, 40) * t)
    s = np.sin(phase) * am
    n = len(s)
    hold = int(n * 0.5)
    ramp = n - int(0.01 * SAMPLE_RATE) - hold
    env  = np.concatenate([np.linspace(0, 1, int(0.01 * SAMPLE_RATE)),
                            np.ones(hold),
                            np.linspace(1, 0, max(1, ramp))])[:n]
    s = apply_echo(s * env, delay=random.uniform(0.04, 0.08), feedback=0.35)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _scifi_ping():
    """Clean sonar ping with long reverb tail."""
    freq = random.uniform(800, 2000)
    dur  = random.uniform(0.15, 0.35)
    t    = _t(dur)
    s    = np.sin(2 * np.pi * freq * t)
    n    = len(s)
    a    = max(1, int(0.002 * SAMPLE_RATE))
    env  = np.concatenate([np.linspace(0, 1, a),
                           np.exp(-np.linspace(0, 12, n - a))])[:n]
    s    = apply_reverb(s * env, decay=0.4, num_echoes=4)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.6)


def _scifi_zap():
    """Short FM burst — electric arc / energy discharge."""
    freq = random.uniform(600, 1800)
    dur  = random.uniform(0.02, 0.06)
    s    = fm_metallic(freq, dur)
    s    = apply_envelope(s)
    s    = soft_saturate(s, random.uniform(2.5, 4.5))
    s    = add_noise(s, 0.015)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.78)


def _scifi_teleport():
    """Multi-frequency cluster + wobble — matter transporter."""
    dur   = random.uniform(0.12, 0.28)
    freqs = [random.uniform(300, 3000) for _ in range(random.randint(3, 6))]
    s     = sum(np.sin(2 * np.pi * f * _t(dur)) for f in freqs) / len(freqs)
    n     = len(s)
    a     = max(1, int(0.02 * SAMPLE_RATE))
    env   = np.concatenate([np.linspace(0, 1, a),
                             np.exp(-np.linspace(0, 8, n - a))])[:n]
    s     = pitch_wobble(s * env, rate_hz=random.uniform(8, 20), depth=0.008)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_SCIFI_SOUNDS = [
    ("laser",    _scifi_laser,    4),
    ("shield",   _scifi_shield,   2),
    ("warp",     _scifi_warp,     1),
    ("ping",     _scifi_ping,     2),
    ("zap",      _scifi_zap,      4),
    ("teleport", _scifi_teleport, 1),
]


def make_scifi_sequence(target_duration=10.0):
    names, fns, weights = zip(*_SCIFI_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.03 * gap_scale, 0.3 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# HAPTIC MODE — micro-impacts, taps, buzzes, thuds, rumbles
# =============================================================================

def _haptic_tap():
    """Single soft tap — short low-freq noise burst."""
    dur = random.uniform(0.008, 0.025)
    n   = int(dur * SAMPLE_RATE)
    s   = _lowpass(np.random.randn(n), random.uniform(200, 500))
    env = np.exp(-np.linspace(0, random.uniform(30, 80), n))
    s   = (s * env).astype(np.float32)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.8)


def _haptic_click():
    """Hard transient click."""
    n = int(0.012 * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 600, 3000)
    env = np.exp(-np.linspace(0, 60, n))
    s   = (s * env).astype(np.float32)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.88)


def _haptic_buzz():
    """Rapid tap sequence — vibration buzz."""
    n_taps   = random.randint(3, 8)
    interval = random.uniform(0.015, 0.04)
    parts    = []
    for _ in range(n_taps):
        n = int(0.01 * SAMPLE_RATE)
        s = _lowpass(np.random.randn(n), 300) * random.uniform(0.5, 1.0)
        env = np.exp(-np.linspace(0, 50, n))
        parts.append((s * env).astype(np.float32))
        parts.append(np.zeros(int(interval * SAMPLE_RATE), dtype=np.float32))
    mono = np.concatenate(parts); del parts
    return to_stereo(mono / (np.max(np.abs(mono)) + 1e-9) * 0.78)


def _haptic_thud():
    """Sub-bass impact thud."""
    dur  = random.uniform(0.04, 0.1)
    n    = int(dur * SAMPLE_RATE)
    freq = random.uniform(40, 100)
    t    = np.arange(n) / SAMPLE_RATE
    s    = np.sin(2 * np.pi * freq * t)
    s   += _lowpass(np.random.randn(n), 200) * 0.4
    env  = np.exp(-np.linspace(0, random.uniform(20, 50), n))
    s    = soft_saturate((s * env).astype(np.float32), 2.0)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.85)


def _haptic_rumble():
    """Rolling low-freq burst — motor/rumble."""
    dur = random.uniform(0.08, 0.2)
    n   = int(dur * SAMPLE_RATE)
    s   = _lowpass(np.random.randn(n), 150)
    am  = 0.6 + 0.4 * np.sin(2 * np.pi * random.uniform(20, 50) * np.arange(n) / SAMPLE_RATE)
    s   = (s * am).astype(np.float32)
    a   = max(1, int(0.01 * SAMPLE_RATE))
    env = np.concatenate([np.linspace(0, 1, a),
                          np.exp(-np.linspace(0, 8, n - a))])[:n]
    s   = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.8)


_HAPTIC_SOUNDS = [
    ("tap",    _haptic_tap,    4),
    ("click",  _haptic_click,  3),
    ("buzz",   _haptic_buzz,   2),
    ("thud",   _haptic_thud,   2),
    ("rumble", _haptic_rumble, 1),
]


def make_haptic_sequence(target_duration=10.0):
    names, fns, weights = zip(*_HAPTIC_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.01 * gap_scale, 0.15 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# RADIO MODE — static bursts, morse bleeps, tuning sweeps, signal lock
# =============================================================================

def _radio_static():
    """Short burst of band-limited noise static."""
    dur = random.uniform(0.02, 0.15)
    n   = int(dur * SAMPLE_RATE)
    s   = _bandpass(np.random.randn(n),
                    random.uniform(300, 800), random.uniform(2000, 6000)).astype(np.float32)
    env = np.exp(-np.linspace(0, random.uniform(2, 10), n)) * random.uniform(0.4, 1.0)
    s   = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


def _radio_bleep():
    """Morse-style sine blip with flat-top envelope."""
    freq = random.uniform(600, 1400)
    dur  = random.uniform(0.04, 0.12)
    t    = _t(dur)
    s    = np.sin(2 * np.pi * freq * t)
    n    = len(s)
    a    = max(1, int(0.005 * SAMPLE_RATE))
    env  = np.concatenate([np.linspace(0, 1, a),
                            np.ones(max(0, n - 2 * a)),
                            np.linspace(1, 0, a)])[:n]
    s    = s * env + np.random.randn(n) * 0.025
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _radio_sweep():
    """Tuning sweep — scanning across frequencies with noise."""
    dur      = random.uniform(0.15, 0.4)
    t        = _t(dur)
    f_start  = random.uniform(200, 600)
    f_end    = f_start * random.uniform(3.0, 8.0)
    freq_curve = f_start + (f_end - f_start) * (t / dur)
    phase    = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s        = np.sin(phase) * 0.6 + np.random.randn(len(t)) * 0.3
    n        = len(s)
    hold     = int(n * 0.6)
    ramp     = max(1, n - int(0.02 * SAMPLE_RATE) - hold)
    env      = np.concatenate([np.linspace(0, 1, int(0.02 * SAMPLE_RATE)),
                                np.ones(hold),
                                np.linspace(1, 0, ramp)])[:n]
    s        = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


def _radio_lock():
    """Sweep snaps to a clean locked tone — tuner lock."""
    scan_dur   = random.uniform(0.1, 0.2)
    lock_dur   = random.uniform(0.05, 0.12)
    f_target   = random.uniform(800, 1600)
    f_start    = f_target * random.uniform(1.5, 3.0)
    t_scan     = _t(scan_dur)
    freq_curve = f_start + (f_target - f_start) * (t_scan / scan_dur)
    phase_scan = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    scan       = np.sin(phase_scan) + np.random.randn(len(t_scan)) * 0.2
    t_lock     = _t(lock_dur)
    lock       = np.sin(2 * np.pi * f_target * t_lock)
    n_lock     = len(lock)
    a          = max(1, int(0.005 * SAMPLE_RATE))
    env_lock   = np.concatenate([np.linspace(0, 1, a),
                                  np.exp(-np.linspace(0, 6, n_lock - a))])[:n_lock]
    s          = np.concatenate([scan, lock * env_lock])
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _radio_crackle():
    """Intermittent noise crackle — poor reception dropout."""
    dur = random.uniform(0.06, 0.2)
    n   = int(dur * SAMPLE_RATE)
    s   = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        on_len  = int(random.uniform(0.005, 0.02) * SAMPLE_RATE)
        off_len = int(random.uniform(0.005, 0.04) * SAMPLE_RATE)
        end     = min(pos + on_len, n)
        chunk   = _bandpass(np.random.randn(end - pos), 500, 5000).astype(np.float32)
        s[pos:end] = chunk * random.uniform(0.2, 0.8)
        pos += on_len + off_len
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


_RADIO_SOUNDS = [
    ("static",  _radio_static,  3),
    ("bleep",   _radio_bleep,   4),
    ("sweep",   _radio_sweep,   2),
    ("lock",    _radio_lock,    2),
    ("crackle", _radio_crackle, 3),
]


def make_radio_sequence(target_duration=10.0):
    names, fns, weights = zip(*_RADIO_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.02 * gap_scale, 0.25 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# Shared helpers
# =============================================================================

def lookahead_limiter(audio, threshold=0.98, lookahead_ms=5.0, release_ms=50.0):
    """True-peak lookahead limiter.
    Detects peaks up to `lookahead_ms` ahead and applies smooth gain reduction,
    recovering at release_ms. Catches transient clips while preserving dynamics.
    """
    stereo = audio.ndim == 2
    x = audio if audio.dtype == np.float32 else audio.astype(np.float32)
    envelope = (np.max(np.abs(x), axis=1) if stereo else np.abs(x)).astype(np.float32)

    la  = max(1, int(lookahead_ms * SAMPLE_RATE / 1000))
    rel = max(1, int(release_ms   * SAMPLE_RATE / 1000))

    # Rolling peak over future `la` samples (lookahead window)
    peak_env = maximum_filter1d(envelope, size=la, origin=-(la // 2))
    del envelope

    # Gain ceiling: reduce instantly when peak exceeds threshold
    gain = np.where(peak_env > threshold, threshold / (peak_env + 1e-12), 1.0).astype(np.float32)
    del peak_env

    # Release: gain recovers at most 1/rel per sample (smooth upward ramp)
    release_step = 1.0 / rel
    for i in range(1, len(gain)):
        ceiling = min(1.0, gain[i - 1] + release_step)
        if gain[i] > ceiling:
            gain[i] = ceiling

    return (x * (gain[:, None] if stereo else gain)).astype(np.float32)


def _finalise(audio, target_duration, mode='glorb'):
    """Trim/pad, apply knob post-processing, lookahead limiting, peak-normalise."""
    target_samples = int(target_duration * SAMPLE_RATE)
    audio = audio[:target_samples]
    if len(audio) < target_samples:
        pad = target_samples - len(audio)
        audio = np.pad(audio, ((0, pad), (0, 0)))
    audio = _apply_knob_brightness(audio)
    audio = _apply_knob_chaos(audio, mode=mode)
    audio = lookahead_limiter(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9886
    return audio.astype(np.float32)


# =============================================================================
# FOLEY MODE  everyday object sounds
# =============================================================================

def _foley_knock():
    """Wood knock: short low-mid filtered noise burst."""
    dur = random.uniform(0.02, 0.06)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 100, 600)
    body = np.sin(2 * np.pi * random.uniform(140, 240) * np.arange(n) / SAMPLE_RATE)
    env = np.exp(-np.linspace(0, random.uniform(25, 55), n))
    s = soft_saturate((0.8 * s + 0.35 * body) * env, 1.8)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _foley_step():
    """Footstep: sub thud plus crunchy mid transient."""
    dur = random.uniform(0.05, 0.14)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sub_freq = random.uniform(50, 120)
    sub = np.sin(2 * np.pi * sub_freq * t) * np.exp(-np.linspace(0, 16, n))
    crunch = _bandpass(np.random.randn(n), 500, 2000) * np.exp(-np.linspace(0, 26, n))
    heel = np.exp(-np.linspace(0, 80, n))
    s = 0.95 * sub + 0.55 * crunch + 0.15 * heel
    s = soft_saturate(s, 2.0)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _foley_cloth():
    """Cloth swipe: filtered noise sweep from low-mid to airy highs."""
    dur = random.uniform(0.08, 0.2)
    n = int(dur * SAMPLE_RATE)
    progress = np.linspace(0, 1, n)
    noise = np.random.randn(n)
    low = _bandpass(noise, 200, 1200)
    high = _bandpass(noise, 1200, 4000)
    env = np.sin(np.pi * progress) ** 1.3
    s = ((1 - progress) * low + progress * high) * env
    s = add_noise(s, 0.002)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _foley_paper():
    """Paper rustle: dense bursts of high frequency crackle."""
    dur = random.uniform(0.06, 0.18)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    pos = 0
    while pos < n:
        burst_n = max(2, int(random.uniform(0.002, 0.009) * SAMPLE_RATE))
        burst = _bandpass(np.random.randn(burst_n), 2000, 8000)
        burst *= np.exp(-np.linspace(0, random.uniform(10, 25), burst_n))
        end = min(pos + burst_n, n)
        s[pos:end] += burst[:end - pos]
        pos += int(random.uniform(0.001, 0.008) * SAMPLE_RATE)
    s = soft_saturate(s, 1.6)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _foley_impact():
    """Hard surface impact: wide transient with decaying body."""
    dur = random.uniform(0.03, 0.12)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    transient = np.random.randn(n) * np.exp(-np.linspace(0, random.uniform(18, 36), n))
    body = np.sin(2 * np.pi * random.uniform(90, 180) * t) * np.exp(-np.linspace(0, 20, n))
    s = soft_saturate(transient + 0.45 * body, 2.2)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_FOLEY_SOUNDS = [
    ("knock",  _foley_knock,  4),
    ("step",   _foley_step,   3),
    ("cloth",  _foley_cloth,  2),
    ("paper",  _foley_paper,  3),
    ("impact", _foley_impact, 2),
]


def make_foley_sequence(target_duration=10.0):
    names, fns, weights = zip(*_FOLEY_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.08 * gap_scale, 0.4 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# UNDERWATER MODE  submarine / ocean sounds
# =============================================================================

def _uw_bubble_mono(dur=None):
    dur = random.uniform(0.01, 0.04) if dur is None else dur
    t = _t(dur)
    f0 = random.uniform(100, 260)
    f1 = random.uniform(max(f0 + 80, 220), 800)
    freq_curve = f0 * (f1 / f0) ** (t / max(dur, 1e-6))
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s = np.sin(phase)
    env = np.sin(np.pi * np.linspace(0, 1, len(s))) ** 1.8
    s = _lowpass(s * env, random.uniform(1200, 2400))
    return s


def _uw_ping():
    """Sonar ping with a long reverb tail."""
    dur = random.uniform(0.12, 0.28)
    t = _t(dur)
    freq = random.uniform(200, 600)
    s = np.sin(2 * np.pi * freq * t)
    env = np.exp(-np.linspace(0, 10, len(s)))
    s = apply_reverb(s * env, decay=0.5, num_echoes=8)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _uw_bubble():
    """Single underwater bubble chirp."""
    s = _uw_bubble_mono()
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _uw_bubble_stream():
    """Rapid layered bubble stream."""
    dur = random.uniform(0.12, 0.35)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    pos = 0
    while pos < n:
        bubble = _uw_bubble_mono(random.uniform(0.01, 0.03))
        end = min(pos + len(bubble), n)
        s[pos:end] += bubble[:end - pos] * random.uniform(0.4, 1.0)
        pos += int(random.uniform(0.004, 0.02) * SAMPLE_RATE)
    s = _lowpass(s, 1800)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _uw_hydrophone():
    """Low watery hydrophone ambience."""
    dur = random.uniform(0.3, 0.8)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    noise = _bandpass(np.random.randn(n), 20, 200)
    wobble = np.sin(2 * np.pi * random.uniform(35, 90) * t + 0.7 * np.sin(2 * np.pi * random.uniform(0.15, 0.5) * t))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * random.uniform(0.08, 0.25) * t + random.uniform(0, 2 * np.pi))
    s = 0.75 * noise * env + 0.35 * wobble
    s = _lowpass(s, 300)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _uw_creak():
    """Metal creak: slow FM sweep with pitch wobble."""
    dur = random.uniform(0.12, 0.45)
    t = _t(dur)
    f0 = random.uniform(50, 110)
    f1 = random.uniform(180, 400)
    freq_curve = f0 + (f1 - f0) * np.linspace(0, 1, len(t)) ** 1.4
    mod = 0.45 * np.sin(2 * np.pi * random.uniform(1.2, 3.5) * t)
    phase = 2 * np.pi * np.cumsum(freq_curve * (1 + mod)) / SAMPLE_RATE
    s = np.sin(phase + 1.3 * np.sin(2 * np.pi * freq_curve * 0.35 * t))
    env = np.exp(-np.linspace(0, random.uniform(4, 10), len(s)))
    s = pitch_wobble(s * env, rate_hz=random.uniform(0.8, 2.2), depth=0.004)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_UNDERWATER_SOUNDS = [
    ("ping",          _uw_ping,          2),
    ("bubble",        _uw_bubble,        4),
    ("bubble_stream", _uw_bubble_stream, 3),
    ("hydrophone",    _uw_hydrophone,    2),
    ("creak",         _uw_creak,         2),
]


def make_underwater_sequence(target_duration=10.0):
    names, fns, weights = zip(*_UNDERWATER_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.05 * gap_scale, 0.5 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration, mode='underwater')


# =============================================================================
# WEATHER MODE  atmospheric sounds
# =============================================================================

def _wx_gust():
    """Wind gust with broad band-limited noise and slow motion."""
    dur = random.uniform(0.3, 0.8)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    s = _bandpass(np.random.randn(n), 100, 1000)
    am = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(0.4, 1.5) * t - np.pi / 2))
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.4
    s = _lowpass(s * am * env, 1800)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _wx_thunder():
    """Thunder rumble with slow attack and long decay."""
    dur = random.uniform(0.5, 2.0)
    n = int(dur * SAMPLE_RATE)
    low = _bandpass(np.random.randn(n), 20, 200)
    mid = _bandpass(np.random.randn(n), 120, 500) * 0.25
    attack = max(1, int(random.uniform(0.05, 0.2) * SAMPLE_RATE))
    env = np.concatenate([
        np.linspace(0, 1, attack),
        np.exp(-np.linspace(0, random.uniform(2, 6), n - attack)),
    ])[:n]
    s = soft_saturate((low + mid) * env, 1.7)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _wx_raindrop():
    """Heavy raindrop transient with a short ring."""
    dur = random.uniform(0.02, 0.07)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    burst = _bandpass(np.random.randn(n), 1000, 4000) * np.exp(-np.linspace(0, 30, n))
    ring = np.sin(2 * np.pi * random.uniform(1200, 2800) * t) * np.exp(-np.linspace(0, 18, n))
    s = burst + 0.6 * ring
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _wx_hail():
    """Dense cluster of short hard drops."""
    dur = random.uniform(0.08, 0.22)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    pos = 0
    while pos < n:
        drop_n = max(2, int(random.uniform(0.002, 0.008) * SAMPLE_RATE))
        burst = _bandpass(np.random.randn(drop_n), 1500, 6000)
        burst *= np.exp(-np.linspace(0, random.uniform(20, 45), drop_n))
        end = min(pos + drop_n, n)
        s[pos:end] += burst[:end - pos]
        pos += int(random.uniform(0.001, 0.01) * SAMPLE_RATE)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _wx_crackle_lightning():
    """Very short electric crackle burst."""
    dur = random.uniform(0.01, 0.03)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 500, 8000)
    env = np.exp(-np.linspace(0, random.uniform(30, 70), n))
    s = soft_saturate(s * env, 2.4)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_WEATHER_SOUNDS = [
    ("gust",      _wx_gust,              2),
    ("thunder",   _wx_thunder,           1),
    ("raindrop",  _wx_raindrop,          5),
    ("hail",      _wx_hail,              3),
    ("lightning", _wx_crackle_lightning, 2),
]


def make_weather_sequence(target_duration=10.0):
    names, fns, weights = zip(*_WEATHER_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.02 * gap_scale, 0.3 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration, mode='weather')


# =============================================================================
# BELL MODE  tonal percussion synthesis
# =============================================================================

_BELL_PENTA = [1.0, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2.0]


def _bell_tubular():
    """Tubular bell with inharmonic FM partials and long decay."""
    freq = random.uniform(220, 720)
    dur = random.uniform(2.0, 4.0)
    t = _t(dur)
    partials = [1.0, 2.76, 5.4, 8.93]
    s = np.zeros(len(t))
    for i, ratio in enumerate(partials):
        pf = freq * ratio
        mod = 0.15 * pf * np.sin(2 * np.pi * (0.12 + 0.05 * i) * t)
        phase = 2 * np.pi * np.cumsum(pf + mod) / SAMPLE_RATE
        s += (1.0 / (1 + i)) * np.sin(phase)
    env = np.exp(-np.linspace(0, random.uniform(4, 7), len(t)))
    s = soft_saturate(s * env, 1.5)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bell_marimba():
    """Marimba-like struck bar with fast decay."""
    freq = random.uniform(220, 900)
    dur = random.uniform(0.3, 0.8)
    t = _t(dur)
    partials = [1.0, 4.0, 9.8]
    s = sum((1.0 / (i + 1)) * np.sin(2 * np.pi * freq * ratio * t) for i, ratio in enumerate(partials))
    env = np.exp(-np.linspace(0, random.uniform(8, 16), len(t)))
    s = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bell_bowl():
    """Singing bowl with slow pitch wobble."""
    freq = random.uniform(200, 520)
    dur = random.uniform(0.4, 1.2)
    t = _t(dur)
    s = np.sin(2 * np.pi * freq * t)
    s += 0.55 * np.sin(2 * np.pi * freq * 2.0 * t)
    s = pitch_wobble(s, rate_hz=random.uniform(0.15, 0.5), depth=0.0025)
    env = np.exp(-np.linspace(0, random.uniform(3, 6), len(t)))
    s = apply_reverb(s * env, decay=0.28, num_echoes=3)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bell_chime():
    """Wind chime cluster at pentatonic intervals."""
    base = random.uniform(220, 520)
    dur = random.uniform(0.7, 1.6)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    count = random.randint(3, 5)
    intervals = random.sample(_BELL_PENTA[:-1], count)
    for ratio in intervals:
        partial_dur = random.uniform(0.45, dur)
        t = _t(partial_dur)
        freq = min(1200, base * ratio)
        tone = np.sin(2 * np.pi * freq * t)
        tone += 0.35 * np.sin(2 * np.pi * freq * 2.76 * t)
        tone *= np.exp(-np.linspace(0, random.uniform(3, 7), len(t)))
        start = int(random.uniform(0, 0.12) * SAMPLE_RATE)
        end = min(start + len(tone), n)
        s[start:end] += tone[:end - start]
    s = apply_reverb(s, decay=0.32, num_echoes=4)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_BELL_SOUNDS = [
    ("tubular", _bell_tubular, 2),
    ("marimba", _bell_marimba, 4),
    ("bowl",    _bell_bowl,    3),
    ("chime",   _bell_chime,   2),
]


def make_bell_sequence(target_duration=10.0):
    names, fns, weights = zip(*_BELL_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.15 * gap_scale, 0.8 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# BASS MODE  sub-bass and low synthesis
# =============================================================================

def _bass_808():
    """808 kick with falling pitch envelope."""
    dur = random.uniform(0.3, 0.8)
    t = _t(dur)
    f_end = random.uniform(50, 80)
    f_start = 200.0
    freq_curve = f_end + (f_start - f_end) * np.exp(-t / random.uniform(0.03, 0.08))
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s = np.sin(phase)
    click = np.exp(-np.linspace(0, 90, len(t)))
    env = np.concatenate([
        np.linspace(0, 1, max(1, int(0.004 * SAMPLE_RATE))),
        np.exp(-np.linspace(0, random.uniform(4, 7), len(t) - max(1, int(0.004 * SAMPLE_RATE)))),
    ])[:len(t)]
    s = soft_saturate(s * env + 0.1 * click, 2.4)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bass_sub_punch():
    """Sub hit with a hard front transient."""
    dur = random.uniform(0.12, 0.35)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    freq = random.uniform(40, 80)
    sub = np.sin(2 * np.pi * freq * t) * np.exp(-np.linspace(0, 12, n))
    transient = _lowpass(np.random.randn(n), 250) * np.exp(-np.linspace(0, 60, n))
    s = soft_saturate(sub + 0.35 * transient, 2.1)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bass_growl():
    """Growling bass with harmonic saw content and moving emphasis."""
    dur = random.uniform(0.2, 0.5)
    t = _t(dur)
    freq = random.uniform(40, 100)
    saw = np.zeros(len(t))
    for k in range(1, 9):
        saw += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    low = _bandpass(saw, max(30, freq * 0.8), min(600, freq * 4.5))
    high = _bandpass(saw, max(80, freq * 1.5), min(1200, freq * 8.0))
    mix = np.linspace(0, 1, len(t))
    s = ((1 - mix) * low + mix * high) * np.exp(-np.linspace(0, random.uniform(5, 10), len(t)))
    s = soft_saturate(s, 2.8)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _bass_rumble():
    """Subsonic noise rumble."""
    dur = random.uniform(0.18, 0.45)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 20, 50)
    env = np.exp(-np.linspace(0, random.uniform(3, 7), n))
    s = soft_saturate(s * env, 2.0)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_BASS_SOUNDS = [
    ("808",       _bass_808,       3),
    ("sub_punch", _bass_sub_punch, 3),
    ("growl",     _bass_growl,     2),
    ("rumble",    _bass_rumble,    2),
]


def make_bass_sequence(target_duration=10.0):
    names, fns, weights = zip(*_BASS_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.05 * gap_scale, 0.3 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# GLITCH MODE  digital artifacts
# =============================================================================

def _glitch_bitcrush():
    """Quantized short sine burst."""
    dur = random.uniform(0.03, 0.09)
    t = _t(dur)
    s = np.sin(2 * np.pi * random.uniform(220, 1600) * t)
    levels = 2 ** random.randint(2, 6)
    s = np.round(s * (levels / 2)) / (levels / 2)
    env = np.exp(-np.linspace(0, random.uniform(6, 18), len(t)))
    s = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _glitch_stutter():
    """Repeated rapid micro-grain with pitch shifts."""
    grain_dur = random.uniform(0.006, 0.018)
    repeats = random.randint(3, 12)
    parts = []
    for _ in range(repeats):
        t = _t(grain_dur)
        freq = random.uniform(300, 2600) * random.choice([0.75, 1.0, 1.25, 1.5])
        grain = np.sin(2 * np.pi * freq * t)
        grain *= np.sin(np.pi * np.linspace(0, 1, len(grain)))
        parts.append(grain)
        parts.append(np.zeros(int(random.uniform(0.001, 0.006) * SAMPLE_RATE)))
    s = np.concatenate(parts); del parts
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _glitch_corrupt():
    """Noise with spikes, dropouts, and DC offsets."""
    dur = random.uniform(0.03, 0.12)
    n = int(dur * SAMPLE_RATE)
    s = np.random.randn(n) * np.random.uniform(0.15, 0.5, n)
    for _ in range(random.randint(4, 14)):
        idx = random.randrange(n)
        s[idx:min(n, idx + random.randint(1, 6))] += random.uniform(-2.0, 2.0)
    for _ in range(random.randint(2, 8)):
        start = random.randrange(n)
        end = min(n, start + random.randint(5, max(6, n // 8)))
        s[start:end] = random.uniform(-0.3, 0.3)
    s += random.uniform(-0.2, 0.2)
    s = soft_saturate(s, 2.6)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _glitch_decimate():
    """Downsampled then upsampled alias-heavy tone."""
    dur = random.uniform(0.04, 0.08)
    t = _t(dur)
    base = np.sin(2 * np.pi * random.uniform(300, 2200) * t)
    factor = random.randint(6, 10)
    dec = base[::factor]
    s = np.repeat(dec, factor)[:len(base)]
    env = np.exp(-np.linspace(0, random.uniform(8, 18), len(s)))
    s = s * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _glitch_freeze():
    """Silence-burst-silence grain freeze."""
    pre = np.zeros(int(random.uniform(0.004, 0.015) * SAMPLE_RATE))
    grain_dur = random.uniform(0.006, 0.02)
    t = _t(grain_dur)
    burst = np.sin(2 * np.pi * random.uniform(500, 3000) * t)
    burst *= np.sin(np.pi * np.linspace(0, 1, len(burst)))
    post = np.zeros(int(random.uniform(0.004, 0.015) * SAMPLE_RATE))
    s = np.concatenate([pre, burst, post])
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_GLITCH_SOUNDS = [
    ("bitcrush", _glitch_bitcrush, 3),
    ("stutter",  _glitch_stutter,  3),
    ("corrupt",  _glitch_corrupt,  2),
    ("decimate", _glitch_decimate, 2),
    ("freeze",   _glitch_freeze,   2),
]


def make_glitch_sequence(target_duration=10.0):
    names, fns, weights = zip(*_GLITCH_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.01 * gap_scale, 0.15 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# PINBALL MODE  electromechanical game sounds
# =============================================================================

def _pb_flipper():
    """Flipper coil snap plus cabinet thud."""
    dur = random.uniform(0.02, 0.04)
    n = int(dur * SAMPLE_RATE)
    click = _bandpass(np.random.randn(n), 200, 800) * np.exp(-np.linspace(0, 55, n))
    thud = np.sin(2 * np.pi * random.uniform(100, 180) * np.arange(n) / SAMPLE_RATE) * np.exp(-np.linspace(0, 25, n))
    s = click + 0.45 * thud
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _pb_bumper():
    """Bright metallic bumper ding."""
    dur = random.uniform(0.08, 0.18)
    t = _t(dur)
    freq = random.uniform(800, 2000)
    s = np.sin(2 * np.pi * freq * t)
    s += 0.45 * np.sin(2 * np.pi * freq * 1.6 * t)
    s += 0.25 * np.sin(2 * np.pi * freq * 2.2 * t)
    env = np.exp(-np.linspace(0, random.uniform(8, 16), len(t)))
    s = apply_reverb(s * env, decay=0.25, num_echoes=3)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _pb_ball_roll():
    """Low rolling rattle."""
    dur = random.uniform(0.1, 0.3)
    n = int(dur * SAMPLE_RATE)
    noise = _bandpass(np.random.randn(n), 100, 400)
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(16, 40) * np.arange(n) / SAMPLE_RATE)
    s = noise * trem * np.exp(-np.linspace(0, random.uniform(2, 5), n))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _pb_drain():
    """Descending drain melody with a final thud."""
    freqs = [random.choice(_PENTA[6:11]), random.choice(_PENTA[4:8]), random.choice(_PENTA[2:6])]
    parts = []
    for f in sorted(freqs, reverse=True):
        t = _t(0.05)
        note = np.sin(2 * np.pi * f * t) * np.exp(-np.linspace(0, 10, len(t)))
        parts.append(note)
        parts.append(np.zeros(int(0.012 * SAMPLE_RATE)))
    thud_n = int(0.05 * SAMPLE_RATE)
    thud = np.sin(2 * np.pi * random.uniform(90, 150) * np.arange(thud_n) / SAMPLE_RATE)
    thud *= np.exp(-np.linspace(0, 20, thud_n))
    parts.append(thud)
    s = np.concatenate(parts); del parts
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _pb_target():
    """Pentatonic target blip."""
    freq = random.choice([f for f in _PENTA if 220 <= f <= 880])
    t = _t(random.uniform(0.04, 0.08))
    s = np.sign(np.sin(2 * np.pi * freq * t))
    s *= np.exp(-np.linspace(0, 14, len(t)))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_PINBALL_SOUNDS = [
    ("flipper",   _pb_flipper,   4),
    ("bumper",    _pb_bumper,    3),
    ("ball_roll", _pb_ball_roll, 2),
    ("drain",     _pb_drain,     1),
    ("target",    _pb_target,    3),
]


def make_pinball_sequence(target_duration=10.0):
    names, fns, weights = zip(*_PINBALL_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.03 * gap_scale, 0.2 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# HORROR MODE  dark and tense sound design
# =============================================================================

def _horror_drone():
    """Low drone with tremolo and harmonic grime."""
    dur = random.uniform(0.5, 2.0)
    t = _t(dur)
    freq = random.uniform(30, 80)
    trem = 0.55 + 0.45 * np.sin(2 * np.pi * random.uniform(0.5, 2.0) * t)
    s = np.sin(2 * np.pi * freq * t)
    s += 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    s += 0.08 * _bandpass(np.random.randn(len(t)), 150, 700)
    env = np.exp(-np.linspace(0, random.uniform(1.5, 3.5), len(t)))
    s = soft_saturate(s * trem * env, 2.0)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _horror_stinger():
    """Bright jump-scare sweep."""
    dur = random.uniform(0.1, 0.2)
    t = _t(dur)
    f0, f1 = 200.0, 3000.0
    freq_curve = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    s = np.sin(phase) + 0.35 * np.sin(2 * phase)
    env = np.concatenate([
        np.linspace(0, 1, max(1, int(0.01 * SAMPLE_RATE))),
        np.exp(-np.linspace(0, 12, len(t) - max(1, int(0.01 * SAMPLE_RATE)))),
    ])[:len(t)]
    s = soft_saturate(s * env, 2.6)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _horror_reverse_sweep():
    """Downward eerie sweep with blooming reverb."""
    dur = random.uniform(0.2, 0.6)
    t = _t(dur)
    f0, f1 = 2000.0, 100.0
    freq_curve = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.8
    s = np.sin(phase) * env + 0.25 * _bandpass(np.random.randn(len(t)), 500, 4000) * env
    s = apply_reverb(s, decay=0.4, num_echoes=5)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _horror_whisper():
    """Ghostly band-passed noise whisper."""
    dur = random.uniform(0.3, 1.0)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    s = _bandpass(np.random.randn(n), 200, 800)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(0.2, 0.8) * t + random.uniform(0, 2 * np.pi))
    env = np.sin(np.pi * np.linspace(0, 1, n))
    s = s * am * env
    s = pitch_wobble(s, rate_hz=random.uniform(0.3, 1.0), depth=0.0015)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _horror_scratch():
    """Short high frequency scratch burst."""
    dur = random.uniform(0.02, 0.08)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 4000, 12000)
    env = np.exp(-np.linspace(0, random.uniform(18, 45), n))
    s = soft_saturate(s * env, 2.2)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_HORROR_SOUNDS = [
    ("drone",         _horror_drone,         2),
    ("stinger",       _horror_stinger,       2),
    ("reverse_sweep", _horror_reverse_sweep, 2),
    ("whisper",       _horror_whisper,       3),
    ("scratch",       _horror_scratch,       2),
]


def make_horror_sequence(target_duration=10.0):
    names, fns, weights = zip(*_HORROR_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.1 * gap_scale, 1.5 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# GRANULAR MODE  micro-grain textures
# =============================================================================

def _grain_sine_mono(duration=None, freq=None):
    dur = random.uniform(0.005, 0.03) if duration is None else duration
    freq = random.uniform(100, 4000) if freq is None else freq
    t = _t(dur)
    env = np.exp(-0.5 * ((np.linspace(-2.5, 2.5, len(t))) ** 2))
    return np.sin(2 * np.pi * freq * t) * env


def _grain_noise_mono(duration=None):
    dur = random.uniform(0.005, 0.02) if duration is None else duration
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), random.uniform(200, 1500), random.uniform(1800, 6000))
    env = np.exp(-0.5 * ((np.linspace(-2.5, 2.5, n)) ** 2))
    return s * env


def _grain_sine():
    """Single gaussian-windowed sine grain."""
    s = _grain_sine_mono()
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _grain_noise():
    """Single gaussian-windowed noise grain."""
    s = _grain_noise_mono()
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _grain_cloud(target_duration=None):
    """Dense cloud of overlapping grains."""
    dur = random.uniform(0.2, 0.8) if target_duration is None else target_duration
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    for _ in range(random.randint(20, 60)):
        grain = _grain_sine_mono() if random.random() < 0.6 else _grain_noise_mono()
        start = random.randint(0, max(0, n - 1))
        end = min(start + len(grain), n)
        s[start:end] += grain[:end - start] * random.uniform(0.2, 0.9)
    s = soft_saturate(s, 1.7)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _grain_sparse(target_duration=None):
    """Sparse separated grains with silence between them."""
    dur = random.uniform(0.2, 0.8) if target_duration is None else target_duration
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    pos = 0
    while pos < n:
        grain = _grain_sine_mono() if random.random() < 0.7 else _grain_noise_mono()
        end = min(pos + len(grain), n)
        s[pos:end] += grain[:end - pos] * random.uniform(0.3, 0.9)
        pos += len(grain) + int(random.uniform(0.01, 0.08) * SAMPLE_RATE)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _grain_pitched(target_duration=None):
    """Cluster of similarly pitched grains for chorus texture."""
    dur = random.uniform(0.3, 0.9) if target_duration is None else target_duration
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    center = random.uniform(180, 900)
    for _ in range(random.randint(20, 50)):
        freq = center * random.uniform(0.96, 1.04)
        grain = _grain_sine_mono(duration=random.uniform(0.01, 0.035), freq=freq)
        start = random.randint(0, max(0, n - 1))
        end = min(start + len(grain), n)
        s[start:end] += grain[:end - start] * random.uniform(0.25, 0.8)
    s = pitch_wobble(s, rate_hz=random.uniform(0.2, 0.7), depth=0.0015)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_GRANULAR_SOUNDS = [
    ("grain_sine",   _grain_sine,   2),
    ("grain_noise",  _grain_noise,  2),
    ("grain_cloud",  _grain_cloud,  4),
    ("grain_sparse", _grain_sparse, 2),
    ("grain_pitched", _grain_pitched, 3),
]


def make_granular_sequence(target_duration=10.0):
    names, fns, weights = zip(*_GRANULAR_SOUNDS)
    total_samples = int(target_duration * SAMPLE_RATE)
    audio = np.zeros((total_samples, 2), dtype=np.float32)
    pos = 0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while pos < total_samples:
        idx = random.choices(range(len(names)), weights=weights)[0]
        if names[idx] in {"grain_cloud", "grain_sparse", "grain_pitched"}:
            sig = fns[idx](random.uniform(0.15, 0.7))
        else:
            sig = fns[idx]()
        start = min(pos + int(random.uniform(0.0, 0.03) * SAMPLE_RATE), total_samples - 1)
        end = min(start + len(sig), total_samples)
        audio[start:end] += sig[:end - start]
        pos = end + int(random.uniform(0.002 * gap_scale, 0.06 * gap_scale) * SAMPLE_RATE)
    return _finalise(audio, target_duration, mode='granular')


# =============================================================================
# LOFI MODE  telephone and degraded textures
# =============================================================================

def _lofi_phone():
    """Telephone band-limited tone with 8-bit quantization."""
    dur = random.uniform(0.08, 0.22)
    t = _t(dur)
    carrier = np.sin(2 * np.pi * random.uniform(320, 880) * t)
    carrier += 0.45 * np.sin(2 * np.pi * random.uniform(900, 1800) * t)
    carrier *= 0.6 + 0.4 * np.sin(2 * np.pi * random.uniform(4, 10) * t)
    s = _bandpass(carrier, 300, 3400)
    s = np.round(s * 127) / 127.0
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _lofi_vinyl():
    """Vinyl crackle with low rumble bed."""
    dur = random.uniform(0.12, 0.35)
    n = int(dur * SAMPLE_RATE)
    rumble = _bandpass(np.random.randn(n), 20, 120) * 0.35
    crackle = np.zeros(n)
    for _ in range(random.randint(8, 24)):
        idx = random.randrange(n)
        crackle[idx:min(n, idx + random.randint(1, 10))] += random.uniform(-1.0, 1.0)
    crackle = _bandpass(crackle, 1500, 8000)
    s = rumble + crackle
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _lofi_cassette():
    """Cassette hiss with wow/flutter wobble."""
    dur = random.uniform(0.15, 0.5)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 2000, 8000)
    s *= 0.4 + 0.6 * np.sin(2 * np.pi * random.uniform(0.2, 0.9) * np.arange(n) / SAMPLE_RATE + random.uniform(0, 2 * np.pi))
    s = pitch_wobble(s, rate_hz=random.uniform(0.4, 1.4), depth=0.003)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _lofi_line_buzz():
    """Power-line buzz at 50/60Hz with harmonics."""
    dur = random.uniform(0.12, 0.3)
    t = _t(dur)
    base = random.choice([50.0, 60.0])
    s = np.zeros(len(t))
    for k, amp in zip([1, 2, 3, 4], [1.0, 0.45, 0.3, 0.2]):
        s += amp * np.sin(2 * np.pi * base * k * t)
    s *= np.exp(-np.linspace(0, random.uniform(0.8, 2.0), len(t)))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _lofi_dropout():
    """Tape dropout: short silence followed by noisy recovery."""
    silent = np.zeros(int(random.uniform(0.01, 0.04) * SAMPLE_RATE))
    burst_n = int(random.uniform(0.02, 0.08) * SAMPLE_RATE)
    burst = _bandpass(np.random.randn(burst_n), 400, 5000)
    burst *= np.exp(-np.linspace(0, random.uniform(8, 18), burst_n))
    s = np.concatenate([silent, burst])
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_LOFI_SOUNDS = [
    ("phone",    _lofi_phone,    3),
    ("vinyl",    _lofi_vinyl,    3),
    ("cassette", _lofi_cassette, 2),
    ("line_buzz", _lofi_line_buzz, 2),
    ("dropout",  _lofi_dropout,  2),
]


def make_lofi_sequence(target_duration=10.0):
    names, fns, weights = zip(*_LOFI_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.02 * gap_scale, 0.2 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration, mode='lofi')


# =============================================================================
# MODEM MODE  dial-up handshake & data tones
# =============================================================================

def _modem_fsk(freq_lo, freq_hi, duration, baud_rate=300):
    """FSK: alternate randomly between two frequencies at baud_rate."""
    bit_dur = 1.0 / baud_rate
    n_bits = max(1, int(duration / bit_dur))
    parts = []
    for _ in range(n_bits):
        f = random.choice([freq_lo, freq_hi])
        t = _t(bit_dur)
        parts.append(np.sin(2 * np.pi * f * t))
    s = np.concatenate(parts); del parts
    fade = int(0.01 * SAMPLE_RATE)
    if len(s) > 2 * fade:
        s[:fade] *= np.linspace(0, 1, fade)
        s[-fade:] *= np.linspace(1, 0, fade)
    return s


def _modem_answer_tone():
    """2100 Hz answer tone with periodic 180° phase reversals."""
    dur = random.uniform(0.3, 0.8)
    t = _t(dur)
    s = np.sin(2 * np.pi * 2100 * t)
    period = int(0.45 * SAMPLE_RATE)
    for i in range(0, len(s), period * 2):
        s[i:min(i + period, len(s))] *= -1
    fade = int(0.015 * SAMPLE_RATE)
    s[:fade] *= np.linspace(0, 1, fade)
    s[-fade:] *= np.linspace(1, 0, fade)
    return to_stereo(s * 0.7)


def _modem_handshake():
    """V.21/V.22 negotiation: two FSK bursts at different frequencies."""
    s1 = _modem_fsk(1080, 1750, random.uniform(0.08, 0.22), baud_rate=300)
    gap = np.zeros(int(random.uniform(0.02, 0.07) * SAMPLE_RATE))
    s2 = _modem_fsk(2025, 2225, random.uniform(0.1, 0.28), baud_rate=1200)
    s = np.concatenate([s1, gap, s2])
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _modem_dtmf():
    """DTMF dial tone: sum of row + column frequencies."""
    rows = [697, 770, 852, 941]
    cols = [1209, 1336, 1477]
    f1, f2 = random.choice(rows), random.choice(cols)
    dur = random.uniform(0.05, 0.16)
    t = _t(dur)
    s = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    fade = int(0.008 * SAMPLE_RATE)
    if len(s) > 2 * fade:
        s[:fade] *= np.linspace(0, 1, fade)
        s[-fade:] *= np.linspace(1, 0, fade)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.6)


def _modem_screech():
    """Initial connect screech: beating high carriers with FM sweep."""
    dur = random.uniform(0.2, 0.55)
    t = _t(dur)
    f1 = random.uniform(2000, 3200)
    f2 = f1 + random.uniform(200, 700)
    sweep = np.cumsum(np.sin(2 * np.pi * random.uniform(300, 900) * t)
                      * random.uniform(50, 180) / SAMPLE_RATE)
    s = (np.sin(2 * np.pi * f1 * t) + 0.8 * np.sin(2 * np.pi * f2 * t)) \
        * np.sin(2 * np.pi * 1200 * t + 2 * np.pi * sweep)
    s = _bandpass(s, 600, 4000)
    fade = int(0.02 * SAMPLE_RATE)
    s[:fade] *= np.linspace(0, 1, fade)
    s[-fade:] *= np.linspace(1, 0, fade)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


_MODEM_SOUNDS = [
    ("answer",    _modem_answer_tone, 2),
    ("handshake", _modem_handshake,   4),
    ("dtmf",      _modem_dtmf,        4),
    ("screech",   _modem_screech,     2),
]


def make_modem_sequence(target_duration=10.0):
    names, fns, weights = zip(*_MODEM_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.03 * gap_scale, 0.22 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# INSECTS MODE  crickets, cicadas, grasshoppers
# =============================================================================

def _insect_cricket():
    """Cricket chirp: 3–5 short pulses at 3.8–5.2 kHz."""
    base = random.uniform(3800, 5200)
    n_pulses = random.randint(2, 5)
    parts = []
    for _ in range(n_pulses):
        dur = random.uniform(0.007, 0.022)
        t = _t(dur)
        s = (np.sin(2 * np.pi * base * t)
             + 0.4 * np.sin(2 * np.pi * base * 2 * t)
             + 0.15 * np.sin(2 * np.pi * base * 3 * t))
        env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.5
        parts.append(s * env)
        parts.append(np.zeros(int(random.uniform(0.003, 0.012) * SAMPLE_RATE)))
    s = np.concatenate(parts); del parts
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


def _insect_cicada():
    """Cicada buzz: AM-modulated band noise, rapid wing-beat amplitude."""
    dur = random.uniform(0.15, 0.55)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    fc = random.uniform(3000, 7000)
    wing = random.uniform(100, 220)
    am = 0.5 + 0.5 * np.abs(np.sin(np.pi * wing * t))
    noise = _bandpass(np.random.randn(n), fc * 0.85, min(fc * 1.15, SAMPLE_RATE * 0.45))
    env = np.exp(-np.linspace(0, random.uniform(1.5, 5), n))
    env[:max(1, int(0.03 * n))] = np.linspace(0, 1, max(1, int(0.03 * n)))
    s = noise * am * env
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


def _insect_grasshopper():
    """Grasshopper rasp: irregular high-frequency scratch bursts."""
    dur = random.uniform(0.04, 0.12)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    pos = 0
    while pos < n:
        pn = max(2, int(random.uniform(0.003, 0.009) * SAMPLE_RATE))
        pulse = _bandpass(np.random.randn(pn), 2000, min(6000, SAMPLE_RATE * 0.45))
        pulse *= np.exp(-np.linspace(0, random.uniform(15, 35), pn))
        end = min(pos + pn, n)
        s[pos:end] += pulse[:end - pos]
        pos += int(random.uniform(0.004, 0.013) * SAMPLE_RATE)
    s = soft_saturate(s, 1.5)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.65)


def _insect_water_bug():
    """Water strider click: inharmonic pluck, random stereo pan."""
    freq = random.uniform(1800, 3800)
    dur = random.uniform(0.015, 0.045)
    t = _t(dur)
    s = (np.sin(2 * np.pi * freq * t)
         + 0.5 * np.sin(2 * np.pi * freq * 1.45 * t))
    s *= np.exp(-np.linspace(0, random.uniform(20, 50), len(t)))
    pan = random.uniform(0.15, 0.85)
    stereo = np.column_stack([s * (1 - pan), s * pan])
    return stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.65


_INSECT_SOUNDS = [
    ("cricket",     _insect_cricket,     5),
    ("cicada",      _insect_cicada,      3),
    ("grasshopper", _insect_grasshopper, 3),
    ("water_bug",   _insect_water_bug,   2),
]


def make_insects_sequence(target_duration=10.0):
    names, fns, weights = zip(*_INSECT_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.5, 0.25)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.04 * gap_scale, 0.4 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# GAMELAN MODE  Balinese/Javanese metallophones with beating pairs
# =============================================================================

_GAMELAN_ROOTS   = [110, 130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 261.63]
# Inharmonic partial ratios measured from physical gamelan bars
_GAMELAN_PARTIALS = [1.0, 2.756, 5.404, 8.933, 13.37]
_GAMELAN_AMPS     = [1.0, 0.45,  0.25,  0.15,   0.08]


def _gamelan_bar(freq=None, dur=None):
    """Single struck bar: inharmonic partials with random detuning."""
    if freq is None:
        freq = random.choice(_GAMELAN_ROOTS)
    if dur is None:
        dur = random.uniform(0.6, 2.5)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    s = np.zeros(n)
    for ratio, amp in zip(_GAMELAN_PARTIALS, _GAMELAN_AMPS):
        f = freq * ratio
        if f >= SAMPLE_RATE * 0.45:
            break
        f_det = f * (2 ** (random.uniform(-0.5, 0.5) / 1200))
        decay = random.uniform(3, 8) * ratio
        s += amp * np.sin(2 * np.pi * f_det * t) * np.exp(-decay * t / dur)
    # metallic transient
    atk = min(int(0.004 * SAMPLE_RATE), n)
    s[:atk] += _bandpass(np.random.randn(atk), 2000, min(8000, SAMPLE_RATE * 0.45)) \
               * 0.3 * np.linspace(1, 0, atk)
    s = apply_reverb(s, decay=0.4, num_echoes=4)
    pan = random.uniform(0.15, 0.85)
    stereo = np.column_stack([s * np.sqrt(1 - pan), s * np.sqrt(pan)])
    return stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.75


def _gamelan_pair():
    """Paired bars detuned 6–18 cents for ombak (acoustic beating)."""
    freq = random.choice(_GAMELAN_ROOTS)
    freq2 = freq * (2 ** (random.uniform(6, 18) / 1200))
    dur = random.uniform(0.8, 2.2)
    b1 = _gamelan_bar(freq, dur)
    b2 = _gamelan_bar(freq2, dur)
    n = min(len(b1), len(b2))
    s = b1[:n] + b2[:n]
    return s / (np.max(np.abs(s)) + 1e-9) * 0.75


def _gamelan_gong():
    """Low gong: very inharmonic partials, long resonance."""
    freq = random.uniform(55, 110)
    dur = random.uniform(1.5, 3.5)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    s = np.zeros(n)
    for ratio, amp in zip([1.0, 1.51, 2.14, 3.17, 4.56],
                          [1.0, 0.6,  0.4,  0.25, 0.15]):
        f = freq * ratio
        if f >= SAMPLE_RATE * 0.45:
            break
        f_det = f * (2 ** (random.uniform(-1.5, 1.5) / 1200))
        s += amp * np.sin(2 * np.pi * f_det * t) * np.exp(-random.uniform(0.4, 1.2) * ratio * t)
    atk = min(int(0.008 * SAMPLE_RATE), n)
    s[:atk] += _bandpass(np.random.randn(atk), 100, 800) * 0.4 * np.linspace(1, 0, atk)
    pan = random.uniform(0.3, 0.7)
    stereo = np.column_stack([s * np.sqrt(1 - pan), s * np.sqrt(pan)])
    return stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.75


_GAMELAN_SOUNDS = [
    ("bar",  _gamelan_bar,  4),
    ("pair", _gamelan_pair, 4),
    ("gong", _gamelan_gong, 2),
]


def make_gamelan_sequence(target_duration=10.0):
    names, fns, weights = zip(*_GAMELAN_SOUNDS)
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 3.0, 0.5)
    while total < target_duration:
        idx = random.choices(range(len(names)), weights=weights)[0]
        sig = fns[idx]()
        gap = random.uniform(0.1 * gap_scale, 0.7 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# GLASS / CLOCKWORK / CREATURE / ELECTRICITY / CAVE PRESETS
# =============================================================================

def _modal_strike(partials, duration, base=None, decay=8.0, noise=0.12):
    """Inharmonic struck resonator shared by glass, clockwork, and cave events."""
    base = random.uniform(300, 900) if base is None else base
    t = _t(duration)
    s = np.zeros(len(t), dtype=np.float64)
    for i, ratio in enumerate(partials):
        freq = base * ratio * random.uniform(0.996, 1.004)
        if freq >= SAMPLE_RATE * 0.45:
            continue
        s += (1.0 / (1 + i * 0.55)) * np.sin(2 * np.pi * freq * t) \
             * np.exp(-(decay * (1 + i * 0.32)) * t / duration)
    attack = min(len(s), max(2, int(0.004 * SAMPLE_RATE)))
    s[:attack] += np.random.randn(attack) * noise * np.linspace(1, 0, attack)
    return s


def _glass_ping():
    dur = random.uniform(0.35, 1.1)
    s = _modal_strike([1, 2.32, 4.25, 6.63], dur, decay=7.0, noise=0.08)
    s = apply_reverb(s, decay=0.22, num_echoes=3)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.72)


def _glass_shard():
    dur = random.uniform(0.08, 0.24)
    n = int(dur * SAMPLE_RATE)
    s = np.zeros(n)
    for _ in range(random.randint(3, 8)):
        start = random.randrange(max(1, int(n * 0.65)))
        length = min(n - start, random.randint(max(4, int(.01 * SAMPLE_RATE)), max(5, int(.05 * SAMPLE_RATE))))
        t = np.arange(length) / SAMPLE_RATE
        freq = random.uniform(1800, min(9000, SAMPLE_RATE * .42))
        s[start:start + length] += np.sin(2 * np.pi * freq * t) * np.exp(-np.linspace(0, 16, length))
    s += _bandpass(np.random.randn(n), 1800, min(10000, SAMPLE_RATE * .45)) * np.exp(-np.linspace(0, 28, n)) * .22
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * 0.7)


def _glass_rim():
    dur = random.uniform(0.5, 1.4)
    t = _t(dur)
    base = random.uniform(420, 1100)
    wobble = 1 + .002 * np.sin(2 * np.pi * random.uniform(3, 7) * t)
    phase = 2 * np.pi * np.cumsum(base * wobble) / SAMPLE_RATE
    s = (np.sin(phase) + .35 * np.sin(phase * 2.71)) * np.exp(-np.linspace(0, 5, len(t)))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .68)


_GLASS_SOUNDS = [("ping", _glass_ping, 4), ("shard", _glass_shard, 2), ("rim", _glass_rim, 3)]


def _clock_tick():
    dur = random.uniform(.018, .055)
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    s = .65 * np.sin(2 * np.pi * random.uniform(900, 2200) * t)
    s += .55 * _bandpass(np.random.randn(n), 1200, min(7000, SAMPLE_RATE * .44))
    s *= np.exp(-np.linspace(0, 24, n))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .66)


def _clock_ratchet():
    count = random.randint(4, 11)
    spacing = random.uniform(.014, .035)
    n = int((count * spacing + .05) * SAMPLE_RATE)
    s = np.zeros(n)
    for i in range(count):
        click = _clock_tick()[:, 0]
        start = int(i * spacing * SAMPLE_RATE)
        end = min(n, start + len(click))
        s[start:end] += click[:end-start] * (1 - i / (count * 1.4))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .7)


def _clock_spring():
    dur = random.uniform(.15, .55)
    t = _t(dur)
    f0, f1 = random.uniform(120, 260), random.uniform(700, 1800)
    freq = f0 + (f1 - f0) * np.sin(np.pi * t / dur) ** 2
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    s = np.sin(phase + 1.8 * np.sin(phase * .37)) * np.exp(-np.linspace(0, 7, len(t)))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .7)


_CLOCKWORK_SOUNDS = [("tick", _clock_tick, 6), ("ratchet", _clock_ratchet, 3), ("spring", _clock_spring, 2)]


def _creature_chirp():
    dur = random.uniform(.08, .32)
    t = _t(dur)
    base = random.uniform(180, 750)
    contour = 1 + random.uniform(.4, 1.5) * np.sin(np.pi * t / dur) ** random.uniform(.7, 1.8)
    vibrato = 1 + random.uniform(.01, .045) * np.sin(2 * np.pi * random.uniform(12, 35) * t)
    phase = 2 * np.pi * np.cumsum(base * contour * vibrato) / SAMPLE_RATE
    s = np.sin(phase) + .35 * np.sin(2 * phase) + .16 * np.sin(3 * phase)
    s *= np.sin(np.pi * np.linspace(0, 1, len(t))) ** .7
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .68)


def _creature_growl():
    dur = random.uniform(.25, .8)
    t = _t(dur)
    f0 = random.uniform(45, 110)
    phase = 2 * np.pi * np.cumsum(f0 * (1 + .08 * np.sin(2*np.pi*random.uniform(3, 8)*t))) / SAMPLE_RATE
    voice = np.sin(phase) + .6*np.sin(2*phase) + .35*np.sin(3*phase) + .2*np.sin(5*phase)
    breath = _bandpass(np.random.randn(len(t)), 80, 900) * .28
    s = soft_saturate((voice + breath) * np.sin(np.pi*np.linspace(0, 1, len(t))), 1.8)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .7)


def _creature_call():
    count = random.randint(2, 5)
    parts = []
    for i in range(count):
        call = _creature_chirp()
        parts.extend([call, silence(random.uniform(.025, .09) * (1 + i*.12))])
    return np.concatenate(parts[:-1])


def _creature_breath():
    dur = random.uniform(.25, .7)
    n = int(dur * SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 180, 2400)
    s *= np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .55)


_CREATURE_SOUNDS = [("chirp", _creature_chirp, 4), ("growl", _creature_growl, 2), ("call", _creature_call, 3), ("breath", _creature_breath, 2)]


# =============================================================================
# CAT PRESET — formant-shaped feline vocal gestures
# =============================================================================

def _cat_voiced(freq_curve, duration, nasal=0.7, breath=0.025):
    """Harmonic glottal source shaped by feline-like vocal resonances."""
    t = _t(duration)
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    source = np.zeros(len(t), dtype=np.float64)
    for harmonic in range(1, 13):
        source += np.sin(harmonic * phase) / (harmonic ** 1.18)

    scale = _knob(KNOB_BRIGHTNESS, 0.78, 1.22)
    formants = (
        (520 * scale, 980 * scale, 1.0),
        (1000 * scale, 1750 * scale, nasal),
        (2100 * scale, 3600 * scale, 0.32),
    )
    voice = source * 0.22
    for low, high, gain in formants:
        voice += gain * _bandpass(source, max(60, low), min(SAMPLE_RATE * 0.45, high))
    if breath > 0:
        air = _bandpass(np.random.randn(len(t)), 350, min(6500, SAMPLE_RATE * 0.45))
        voice += air * breath
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.62
    voice *= env
    peak = np.max(np.abs(voice))
    return to_stereo(voice / (peak + 1e-9) * 0.68)


def _cat_meow():
    duration = random.uniform(0.35, 0.95)
    t = _t(duration)
    x = t / duration
    base = _knob(KNOB_BRIGHTNESS, 230, 520) * random.uniform(0.88, 1.12)
    arch = np.sin(np.pi * x) ** random.uniform(0.7, 1.25)
    freq = base * (0.88 + random.uniform(0.65, 1.25) * arch)
    freq *= 1 + random.uniform(0.012, 0.035) * np.sin(2 * np.pi * random.uniform(5, 9) * t)
    return _cat_voiced(freq, duration, nasal=random.uniform(0.65, 0.95))


def _cat_mewl():
    duration = random.uniform(0.18, 0.48)
    t = _t(duration)
    x = t / duration
    base = _knob(KNOB_BRIGHTNESS, 430, 820) * random.uniform(0.92, 1.1)
    freq = base * (0.92 + 0.48 * np.sin(np.pi * x) ** 0.8 - 0.12 * x)
    freq *= 1 + 0.025 * np.sin(2 * np.pi * random.uniform(8, 13) * t)
    return _cat_voiced(freq, duration, nasal=0.92, breath=0.018)


def _cat_two_part_call():
    first = _cat_mewl()
    gap = silence(random.uniform(0.025, 0.07))
    second = _cat_meow()
    return np.concatenate([first, gap, second])


def _cat_trill():
    duration = random.uniform(0.18, 0.5)
    t = _t(duration)
    base = _knob(KNOB_BRIGHTNESS, 320, 700) * random.uniform(0.9, 1.12)
    flutter = 0.10 * np.sin(2 * np.pi * random.uniform(18, 30) * t)
    glide = np.linspace(0.92, random.uniform(1.05, 1.3), len(t))
    return _cat_voiced(base * glide * (1 + flutter), duration, nasal=0.85, breath=0.012)


def _cat_purr():
    duration = random.uniform(0.65, 1.6)
    t = _t(duration)
    fundamental = random.uniform(75, 125)
    phase = 2 * np.pi * np.cumsum(fundamental * (1 + 0.025 * np.sin(2*np.pi*1.3*t))) / SAMPLE_RATE
    voice = np.sin(phase) + 0.45*np.sin(2*phase) + 0.2*np.sin(3*phase)
    pulse = 0.28 + 0.72 * np.maximum(0, np.sin(2*np.pi*random.uniform(22, 29)*t)) ** 1.7
    rumble = _bandpass(np.random.randn(len(t)), 45, 420) * 0.3
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.7
    sound = (voice * pulse + rumble) * env
    return to_stereo(sound / (np.max(np.abs(sound)) + 1e-9) * 0.58)


def _cat_hiss():
    duration = random.uniform(0.3, 0.85)
    n = int(duration * SAMPLE_RATE)
    sound = _bandpass(np.random.randn(n), 900, min(9500, SAMPLE_RATE * 0.45))
    flutter = 0.82 + 0.18*np.sin(2*np.pi*random.uniform(7, 14)*np.arange(n)/SAMPLE_RATE)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.45
    sound *= flutter * env
    return to_stereo(sound / (np.max(np.abs(sound)) + 1e-9) * 0.56)


_CAT_SOUNDS = [
    ("meow", _cat_meow, 6),
    ("mewl", _cat_mewl, 3),
    ("two-part", _cat_two_part_call, 2),
    ("trill", _cat_trill, 2),
    ("purr", _cat_purr, 2),
    ("hiss", _cat_hiss, 1),
]


# =============================================================================
# BIRDS PRESET — melodic peeps, whistles, trills, and phrase grammar
# =============================================================================

def _bird_tone(freq_curve, duration, trill_rate=0.0, breath=0.008):
    t = _t(duration)
    if trill_rate:
        freq_curve = freq_curve * (1 + random.uniform(0.018, 0.055) * np.sin(2*np.pi*trill_rate*t))
    phase = 2 * np.pi * np.cumsum(freq_curve) / SAMPLE_RATE
    sound = np.sin(phase) + 0.22*np.sin(2*phase + 0.4) + 0.07*np.sin(3*phase)
    if breath:
        sound += breath * _bandpass(np.random.randn(len(t)), 1800, min(10000, SAMPLE_RATE * 0.45))
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.48
    sound *= env
    return to_stereo(sound / (np.max(np.abs(sound)) + 1e-9) * 0.62,
                     haas_ms=random.uniform(0.4, 3.0))


def _bird_peep():
    duration = random.uniform(0.055, 0.16)
    t = _t(duration)
    x = t / duration
    base = _knob(KNOB_BRIGHTNESS, 1700, 4300) * random.uniform(0.85, 1.18)
    contour = 0.9 + random.uniform(0.18, 0.55) * np.sin(np.pi*x) ** 0.8
    return _bird_tone(base * contour, duration)


def _bird_double_peep():
    first = _bird_peep()
    return np.concatenate([first, silence(random.uniform(0.025, 0.075)), _bird_peep()])


def _bird_whistle():
    duration = random.uniform(0.16, 0.48)
    t = _t(duration)
    x = t / duration
    start = _knob(KNOB_BRIGHTNESS, 1300, 3600) * random.uniform(0.85, 1.15)
    direction = random.choice((-1, 1))
    sweep = random.uniform(0.22, 0.7)
    curve = x*x*(3 - 2*x)
    freq = start * (1 + direction * sweep * curve)
    return _bird_tone(np.maximum(700, freq), duration, trill_rate=random.uniform(4, 8))


def _bird_trill():
    duration = random.uniform(0.18, 0.52)
    t = _t(duration)
    base = _knob(KNOB_BRIGHTNESS, 1900, 4800) * random.uniform(0.88, 1.12)
    drift = np.linspace(random.uniform(0.9, 1.0), random.uniform(1.0, 1.18), len(t))
    return _bird_tone(base * drift, duration, trill_rate=random.uniform(22, 38), breath=0.004)


def _bird_warble():
    count = random.randint(3, 6)
    parts = []
    root = _knob(KNOB_BRIGHTNESS, 1400, 3500) * random.uniform(0.9, 1.1)
    ratios = random.choice(([1, 1.28, 1.12, 1.5], [1.35, 1.08, 1.42, 0.96], [1, 1.5, 1.25, 1.68]))
    for i in range(count):
        duration = random.uniform(0.055, 0.14)
        t = _t(duration)
        ratio = ratios[i % len(ratios)]
        freq = root * ratio * (1 + 0.12*np.sin(np.pi*t/duration))
        parts.append(_bird_tone(freq, duration, trill_rate=random.uniform(7, 14), breath=0.003))
        if i < count - 1:
            parts.append(silence(random.uniform(0.018, 0.055)))
    return np.concatenate(parts)


_BIRD_SOUNDS = [
    ("peep", _bird_peep, 5),
    ("double-peep", _bird_double_peep, 4),
    ("whistle", _bird_whistle, 3),
    ("trill", _bird_trill, 3),
    ("warble", _bird_warble, 3),
]


def _electric_arc():
    dur = random.uniform(.04, .22)
    n = int(dur * SAMPLE_RATE)
    noise = _bandpass(np.random.randn(n), 500, min(10000, SAMPLE_RATE*.45))
    gate = (np.random.random(n) > random.uniform(.72, .9)).astype(float)
    s = soft_saturate(noise * gate * np.exp(-np.linspace(0, 7, n)), 2.5)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .72)


def _electric_hum():
    dur = random.uniform(.25, .8)
    t = _t(dur)
    mains = random.choice([50, 60])
    s = sum((1/i) * np.sin(2*np.pi*mains*i*t) for i in range(1, 7))
    s *= (.65 + .35*np.sin(2*np.pi*random.uniform(3, 9)*t)) * np.sin(np.pi*np.linspace(0, 1, len(t)))
    s += _bandpass(np.random.randn(len(t)), 1200, 5000) * .08
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .62)


def _electric_relay():
    click = _clock_tick()
    thud = _modal_strike([1, 1.7, 2.8], random.uniform(.04, .11), base=random.uniform(90, 180), decay=15, noise=.2)
    thud = to_stereo(thud / (np.max(np.abs(thud)) + 1e-9) * .6)
    n = max(len(click), len(thud)); out = np.zeros((n, 2)); out[:len(click)] += click; out[:len(thud)] += thud
    return out / (np.max(np.abs(out)) + 1e-9) * .7


def _electric_charge():
    dur = random.uniform(.12, .4)
    t = _t(dur)
    f0, f1 = random.uniform(80, 180), random.uniform(1800, 5000)
    freq = f0 * (f1/f0) ** (t/dur)
    phase = 2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    s = (np.sin(phase) + .25*np.sign(np.sin(phase*1.013))) * np.linspace(0, 1, len(t))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .65)


_ELECTRICITY_SOUNDS = [("arc", _electric_arc, 4), ("hum", _electric_hum, 2), ("relay", _electric_relay, 3), ("charge", _electric_charge, 2)]


def _cave_drip():
    dur = random.uniform(.3, .85)
    t = _t(dur)
    f0, f1 = random.uniform(500, 1100), random.uniform(180, 420)
    freq = f0 * (f1/f0) ** (t/dur)
    phase = 2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    s = np.sin(phase) * np.exp(-np.linspace(0, 13, len(t)))
    s = apply_reverb(s, decay=.7, num_echoes=9)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .66)


def _cave_stone():
    dur = random.uniform(.35, 1.1)
    base = random.uniform(70, 240)
    s = _modal_strike([1, 1.47, 2.09, 3.23], dur, base=base, decay=7, noise=.32)
    s = apply_reverb(s, decay=.75, num_echoes=8)
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .72)


def _cave_rumble():
    dur = random.uniform(.5, 1.8)
    n = int(dur*SAMPLE_RATE)
    s = _bandpass(np.random.randn(n), 20, 180)
    s *= np.sin(np.pi*np.linspace(0, 1, n)) ** .7
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .58)


def _cave_wind():
    dur = random.uniform(.5, 1.4)
    n = int(dur*SAMPLE_RATE); t = np.arange(n)/SAMPLE_RATE
    s = _bandpass(np.random.randn(n), 80, 850)
    s *= (.3 + .7*np.sin(np.pi*np.linspace(0, 1, n))) * (.75 + .25*np.sin(2*np.pi*random.uniform(.2,.7)*t))
    return to_stereo(s / (np.max(np.abs(s)) + 1e-9) * .5)


_CAVE_SOUNDS = [("drip", _cave_drip, 5), ("stone", _cave_stone, 3), ("rumble", _cave_rumble, 2), ("wind", _cave_wind, 2)]


def _make_sound_table_sequence(sounds, target_duration, gap_range, mode='glorb'):
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.8, .3)
    while total < target_duration:
        sig = _weighted_pick(sounds)
        gap = random.uniform(gap_range[0]*gap_scale, gap_range[1]*gap_scale)
        parts.extend([sig, silence(gap)])
        total += len(sig)/SAMPLE_RATE + gap
    return _finalise(np.concatenate(parts), target_duration, mode=mode)


def make_glass_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_GLASS_SOUNDS, target_duration, (.08, .45))


def make_clockwork_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_CLOCKWORK_SOUNDS, target_duration, (.025, .22))


def make_creature_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_CREATURE_SOUNDS, target_duration, (.1, .55))


def make_cat_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_CAT_SOUNDS, target_duration, (.25, 1.25), mode='cat')


def make_birds_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_BIRD_SOUNDS, target_duration, (.12, .8), mode='birds')


def make_electricity_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_ELECTRICITY_SOUNDS, target_duration, (.04, .3))


def make_cave_sequence(target_duration=10.0):
    return _make_sound_table_sequence(_CAVE_SOUNDS, target_duration, (.18, .85), mode='cave')


# =============================================================================
# PRESET EVENT DISPATCH — single-event generators for groove / arp playback
# =============================================================================

def _weighted_pick(sounds_table):
    """Pick and call one event from a (name, fn, weight) dispatch table."""
    weights = [w for _, _, w in sounds_table]
    total = sum(weights)
    r = random.uniform(0, total)
    cumw = 0
    for _, fn, w in sounds_table:
        cumw += w
        if r <= cumw:
            return fn()
    return sounds_table[-1][1]()


# Maps preset name → callable that returns one short audio event (mono or stereo np array).
# 'nature' and 'ui-pack' have no event table; they fall back to glorb blips.
_PRESET_EVENTS = {
    'glorb':      lambda: make_blip()[0],
    'retro':      lambda: _weighted_pick(_RETRO_SOUNDS),
    'scifi':      lambda: _weighted_pick(_SCIFI_SOUNDS),
    'haptic':     lambda: _weighted_pick(_HAPTIC_SOUNDS),
    'radio':      lambda: _weighted_pick(_RADIO_SOUNDS),
    'foley':      lambda: _weighted_pick(_FOLEY_SOUNDS),
    'underwater': lambda: _weighted_pick(_UNDERWATER_SOUNDS),
    'weather':    lambda: _weighted_pick(_WEATHER_SOUNDS),
    'bell':       lambda: _weighted_pick(_BELL_SOUNDS),
    'bass':       lambda: _weighted_pick(_BASS_SOUNDS),
    'glitch':     lambda: _weighted_pick(_GLITCH_SOUNDS),
    'pinball':    lambda: _weighted_pick(_PINBALL_SOUNDS),
    'horror':     lambda: _weighted_pick(_HORROR_SOUNDS),
    'granular':   lambda: _weighted_pick(_GRANULAR_SOUNDS),
    'lofi':       lambda: _weighted_pick(_LOFI_SOUNDS),
    'modem':      lambda: _weighted_pick(_MODEM_SOUNDS),
    'insects':    lambda: _weighted_pick(_INSECT_SOUNDS),
    'gamelan':    lambda: _weighted_pick(_GAMELAN_SOUNDS),
    'glass':      lambda: _weighted_pick(_GLASS_SOUNDS),
    'clockwork':  lambda: _weighted_pick(_CLOCKWORK_SOUNDS),
    'creature':   lambda: _weighted_pick(_CREATURE_SOUNDS),
    'cat':        lambda: _weighted_pick(_CAT_SOUNDS),
    'birds':      lambda: _weighted_pick(_BIRD_SOUNDS),
    'electricity':lambda: _weighted_pick(_ELECTRICITY_SOUNDS),
    'cave':       lambda: _weighted_pick(_CAVE_SOUNDS),
    'nature':     lambda: make_blip()[0],   # fallback — no event table
    'ui-pack':    lambda: make_blip()[0],   # fallback
}


# =============================================================================
# ARP MODE  synthesizer arpeggiator cycling chord patterns
# =============================================================================

_ARP_SCALES = {
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11],
}

_ARP_PATTERNS = [
    [0, 1, 2, 3],          # up
    [3, 2, 1, 0],          # down
    [0, 2, 1, 3, 2, 4],    # skip up
    [0, 2, 4, 2, 0, 2],    # oscillate
    [0, 1, 2, 3, 2, 1],    # up-down
    [0, 3, 1, 4, 2, 5],    # wide skip
]


def _arp_note(freq, dur, wave='tri', detune_cents=0.0):
    """One synthesized arp note with ADSR envelope."""
    f = freq * (2 ** (detune_cents / 1200))
    t = _t(dur)
    n = len(t)
    phase = np.cumsum(np.full(n, f / SAMPLE_RATE))
    if wave == 'tri':
        s = 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1
    elif wave == 'saw':
        s = 2 * (phase - np.floor(phase)) - 1
    else:  # square (soft)
        s = np.tanh(4 * np.sin(2 * np.pi * phase))
    atk = int(min(0.008, dur * 0.1) * SAMPLE_RATE)
    dec = int(min(0.03,  dur * 0.2) * SAMPLE_RATE)
    rel = int(min(0.04,  dur * 0.3) * SAMPLE_RATE)
    env = np.ones(n) * 0.7
    if atk > 0: env[:atk]        = np.linspace(0, 1, atk)
    if dec > 0: env[atk:atk+dec] = np.linspace(1, 0.7, dec)
    if rel > 0: env[-rel:]       = np.linspace(0.7, 0, rel)
    s = _bandpass(s * env, 80, min(8000, SAMPLE_RATE * 0.45))
    return s


def _arp_phrase(scale_name=None, root_midi=None, wave=None, note_dur=None):
    """One arpeggio phrase — all params optional; None → random choice."""
    scale = _ARP_SCALES.get(scale_name) if scale_name else None
    if scale is None:
        scale = random.choice(list(_ARP_SCALES.values()))
    if root_midi is None:
        root_midi = random.randint(48, 72)   # C3–C5
    # Build 3-octave scale as MIDI note numbers → frequencies
    freqs = []
    for octave in range(3):
        for interval in scale:
            midi = root_midi + interval + 12 * octave
            freqs.append(440.0 * (2 ** ((midi - 69) / 12)))
    pattern  = random.choice(_ARP_PATTERNS)
    if wave is None:
        wave = random.choice(['tri', 'saw', 'square'])
    if note_dur is None:
        note_dur = random.uniform(0.04, 0.14)
    gap_dur  = random.uniform(0.0, 0.012)
    detune   = random.uniform(-3, 3)
    parts = []
    for degree in pattern:
        idx = min(degree, len(freqs) - 1)
        note = _arp_note(freqs[idx], note_dur, wave=wave, detune_cents=detune)
        parts.append(note)
        if gap_dur > 0:
            parts.append(np.zeros(int(gap_dur * SAMPLE_RATE)))
    s = np.concatenate(parts); del parts
    s = apply_reverb(s, decay=0.15, num_echoes=3)
    pan = random.uniform(0.25, 0.75)
    stereo = np.column_stack([s * np.sqrt(1 - pan), s * np.sqrt(pan)])
    return stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.7


_ARP_ROOT_MAP = {
    'C':60,'C#':61,'D':62,'D#':63,'E':64,'F':65,
    'F#':66,'G':67,'G#':68,'A':69,'A#':70,'B':71,
}


def make_arp_sequence(target_duration=10.0, bpm=None, scale=None, root=None, wave=None, preset='synth'):
    """Generate an arp sequence.
    preset='synth': use the built-in tri/saw/square synth voice.
    Any other preset: use that preset's event sounds at arp timing positions.
    """
    # When using preset sounds (not synth), just apply arp timing pattern with preset audio
    if preset != 'synth':
        event_fn = _PRESET_EVENTS.get(preset, _PRESET_EVENTS['glorb'])
        note_dur = None
        if bpm is not None:
            bpm = max(40, min(200, int(bpm)))
            note_dur = 60.0 / bpm / 2
        parts, total = [], 0.0
        gap_scale = _knob(KNOB_ENERGY, 2.0, 0.15)
        while total < target_duration:
            sig = event_fn()
            if sig.ndim == 1:
                sig = np.column_stack([sig, sig])
            gap = random.uniform(0.05 * gap_scale, 0.35 * gap_scale)
            if note_dur is not None:
                gap = note_dur * random.uniform(0.8, 1.2)
            parts.append(sig)
            parts.append(silence(gap))
            total += len(sig) / SAMPLE_RATE + gap
        audio = np.concatenate(parts); del parts
        return _finalise(audio, target_duration)

    # Synth arp voice (original behaviour)
    note_dur = None
    if bpm is not None:
        bpm = max(40, min(200, int(bpm)))
        note_dur = 60.0 / bpm / 2          # half-beat per note
    root_midi = None
    if root and root.upper() in _ARP_ROOT_MAP:
        root_midi = _ARP_ROOT_MAP[root.upper()]
    parts, total = [], 0.0
    gap_scale = _knob(KNOB_ENERGY, 2.0, 0.15)
    while total < target_duration:
        sig = _arp_phrase(scale_name=scale, root_midi=root_midi, wave=wave, note_dur=note_dur)
        gap = random.uniform(0.05 * gap_scale, 0.35 * gap_scale)
        parts.append(sig)
        parts.append(silence(gap))
        total += len(sig) / SAMPLE_RATE + gap
    audio = np.concatenate(parts); del parts
    return _finalise(audio, target_duration)


# =============================================================================
# GROOVE MODE  rhythmically quantised blip sequencer
# =============================================================================

_GROOVE_TEMPLATES = {
    # Each entry: list of (beat_position, velocity) within a 4-beat bar
    "four-on-floor": [(0.0, 1.00), (1.0, 1.00), (2.0, 1.00), (3.0, 1.00)],
    "breakbeat":     [(0.0, 1.00), (0.75, 0.70), (1.5, 0.85), (2.0, 0.90),
                      (2.5, 0.60), (3.25, 1.00), (3.5, 0.65)],
    "swing16":       [(0.0, 1.00), (0.667, 0.70), (1.0, 0.90), (1.667, 0.70),
                      (2.0, 1.00), (2.667, 0.70), (3.0, 0.90), (3.667, 0.70)],
    "polyrhythm":    [(0.0, 1.00), (4/3, 0.80), (8/3, 0.80)],
    "sparse":        [(0.0, 1.00), (1.25, 0.60), (3.0, 0.80)],
}


def make_groove_sequence(target_duration=10.0, bpm=90, template='four-on-floor', preset='glorb'):
    """Blips stamped onto a rhythmic grid; bpm, template, and preset are user-controlled."""
    bpm      = max(40, min(200, int(bpm)))
    beat_dur = 60.0 / bpm
    bar_dur  = beat_dur * 4
    events   = _GROOVE_TEMPLATES.get(template, _GROOVE_TEMPLATES['four-on-floor'])
    event_fn = _PRESET_EVENTS.get(preset, _PRESET_EVENTS['glorb'])

    # Absolute event times (seconds) for every bar needed to fill target_duration
    all_events = []
    t = 0.0
    while t < target_duration + bar_dur:
        for pos_beats, vel in events:
            et = t + pos_beats * beat_dur
            if et < target_duration + 0.5:
                all_events.append((et, vel))
        t += bar_dur

    n_total = int((target_duration + 1.0) * SAMPLE_RATE)
    out = np.zeros((n_total, 2), dtype=np.float32)

    for i, (t_event, vel) in enumerate(all_events):
        # Max blip duration = gap to next event minus 5 ms guard
        if i + 1 < len(all_events):
            max_dur = all_events[i + 1][0] - t_event - 0.005
        else:
            max_dur = 0.30
        max_dur = max(0.02, min(max_dur, 0.35))

        blip = event_fn()
        if blip.ndim == 1:
            blip = np.column_stack([blip, blip])

        n_blip = int(max_dur * SAMPLE_RATE)
        if len(blip) > n_blip:
            fade = min(128, n_blip // 4)
            blip = blip[:n_blip].copy()
            blip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        else:
            pad = n_blip - len(blip)
            blip = np.vstack([blip, np.zeros((pad, 2), dtype=np.float32)])

        start = int(t_event * SAMPLE_RATE)
        end   = min(start + len(blip), n_total)
        out[start:end] += blip[:end - start] * np.float32(vel)
        del blip

    return _finalise(out, target_duration)
