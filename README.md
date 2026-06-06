# GLORB

**Generative Sound Synthesiser** by Matteo Bassi. Every render is unique — no samples, no loops, pure mathematics.

🌐 **Live:** https://bassimatte.github.io/glorb/

---

## What you hear

Glorb generates electronic sounds with an organic, analog feel across **21 sound worlds**. Depending on the mode, you'll hear:

- **abstract blips and blops** that chirp, sweep and pop like a living circuit board
- **8-bit chiptune melodies** — square waves stepping through arpeggios with a crisp retro bite
- **rain, fire and insects** — procedural ambient textures that breathe and shift
- **sci-fi effects** — laser shots, warp drives, energy shields and matter transporters
- **deep bass hits** — 808 kicks with falling pitch, subsonic thuds and growling harmonics
- **metallic bells** — tubular bells, marimbas, singing bowls and Balinese gamelan gongs
- **lo-fi nostalgia** — vinyl crackle, cassette hiss, tape dropout and power-line buzz
- **dial-up modem** screech — DTMF tones, FSK handshakes, the full connect sequence
- **generative arpeggios** stepping through minor, pentatonic and modal scales
- **digital glitch** — bit-crushed aliasing, grain stutter, buffer freeze and DC spikes
- **horror textures** — tremolo drones, eerie sweeps, ghost whispers and jump scares
- **underwater soundscapes** — sonar pings, bubble streams and hydrophone hiss
- **foley and haptics** — footsteps, paper rustle, clicks, taps and motor rumble

Three knobs — **⚡ Energy**, **☀ Brightness** and **🌀 Chaos** — shape the density, tone and texture of every render in real time.

---

## Sound worlds

| Mode | Description |
|---|---|
| **Glorb** | Abstract blip/bloop electronic tones |
| **Retro** | 8-bit chiptune square and pulse waves |
| **Nature** | Rain, fire, insects — procedural ambient textures |
| **Sci-Fi** | Phasers, warp drives, laser bursts |
| **Haptic** | Tactile vibration pulses and clicks |
| **Radio** | AM/FM static, morse, transmission artefacts |
| **UI Pack** | Notification pings, confirm tones, interface clicks |
| **Foley** | Footsteps, paper, keys, everyday impacts |
| **Underwater** | Sonar pings, bubble trains, deep pressure tones |
| **Weather** | Thunder, lightning crackle, rain layers |
| **Bell** | Marimba, tubular bells, resonant metal |
| **Bass** | Deep sub-bass pulses and 808-style hits |
| **Glitch** | Buffer corruption, bit-crush, digital stutter |
| **Pinball** | Flippers, bumpers, drain, mechanical clicks |
| **Horror** | Dissonant drones, breath, reversed tails |
| **Granular** | Scattered grain clouds across noise and pitch |
| **Lo-Fi** | Vinyl crackle, tape wow/flutter, telephone bandwidth |
| **Modem** | DTMF tones, FSK handshake, connect screech |
| **Insects** | Crickets, cicadas, grasshoppers, water bugs |
| **Gamelan** | Inharmonic metal bars, detuned pairs, gong resonance |
| **Arp** | Generative arpeggiator across minor, major, pentatonic, and modal scales |

---

## Synthesis engine

- **FM synthesis** — frequency modulation for metallic, bell, sci-fi timbres
- **Additive harmonics** — stacked partials with individual envelopes
- **Subtractive synthesis** — Butterworth bandpass/lowpass filtered noise
- **Granular synthesis** — stochastic grain clouds scattered in time and pitch
- **Physical modelling** — mass-spring-damper resonators for bells and impacts
- **Karplus-Strong** — delay-line feedback for plucked string sounds
- **PolyBLEP** — bandlimited square/pulse waves (no aliasing)
- **TPDF dithering** — triangular noise before quantisation
- **Lookahead limiter** — true-peak limiting with 5ms lookahead + release

---

## Parameter knobs

Three real-time knobs shape every mode:

| Knob | Effect |
|---|---|
| **⚡ Energy** | Low = sparse/slow, High = dense/rapid |
| **☀ Brightness** | Low = dark/muffled, High = bright/sharp |
| **🌀 Chaos** | Low = clean/ordered, High = saturated/glitchy |

Moving a knob auto-generates a 3s preview after 600ms.

---

## Web interface

```bash
pip install -r requirements-server.txt
python server.py
```

Open **http://localhost:5000**

Features: 6 colour themes, live waveform, 7 audio-reactive background visualisers (Dots, Wave, Particles, Rings, Matrix, Blobs, None), loop playback, WAV download, shareable URL (🔗 button encodes mode + knobs + settings into the link).

---

## CLI

```bash
python main.py              # 10s, high quality → blipblop.wav
python main.py -d 30        # 30s output
python main.py -q studio    # 48kHz / 24-bit
python main.py --play       # render and play immediately
```

### Quality presets

| Flag | Sample rate | Bit depth |
|---|---|---|
| `standard` | 44 100 Hz | 16-bit PCM |
| `high` *(default)* | 44 100 Hz | 24-bit PCM |
| `studio` | 48 000 Hz | 24-bit PCM |
| `float` | 44 100 Hz | 32-bit float |

---

## Batch renderer (`generate.py`)

Generate audio files for one, several, or all 21 modes — with optional Freesound bulk-upload XLSX:

```bash
# Single mode
python generate.py --mode glorb

# Multiple modes
python generate.py --mode glorb retro insects modem

# All modes, 60 s each, studio quality
python generate.py --all --duration 60 --quality studio

# All modes + generate Freesound bulk XLSX
python generate.py --all --duration 120 --freesound

# Regenerate XLSX only (no re-render)
python generate.py --all --metadata-only --freesound

# Custom knobs and reproducible seed
python generate.py --all --energy 80 --brightness 30 --chaos 70 --seed 42
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode MODE [...]` | — | Mode(s) to render |
| `--all` | — | Render all 21 modes |
| `--duration SECS` | 30 | Duration in seconds |
| `--quality PRESET` | high | `standard` / `high` / `studio` / `float` |
| `--energy 0-100` | 50 | Event density knob |
| `--brightness 0-100` | 50 | Frequency balance knob |
| `--chaos 0-100` | 50 | Saturation/wobble knob |
| `--output-dir DIR` | `exports/glorb` | Output folder |
| `--freesound` | off | Generate `freesound_bulk.xlsx` |
| `--metadata-only` | off | XLSX only, skip rendering |
| `--seed N` | random | Fix seed for reproducible output |

Output files are named `Glorb_<mode>.wav`. The XLSX matches the [Freesound bulk describe](https://freesound.org/home/describe/) template.

---

## Screenshots

![Glorb UI with yellow particles background](screenshots/main%20ui%20-%20yellow%20particles.png)

---

## Installation

```bash
git clone https://github.com/bassimatte/glorb.git
cd glorb
pip install numpy sounddevice soundfile scipy flask flask-cors
```

---

## Deployment

- **Frontend** (GitHub Pages): `docs/index.html` — served statically
- **Backend** (Railway): Flask + gunicorn via Docker — `https://glorb-production.up.railway.app/`
- Set `GLORB_MAX_DURATION=120` on the Railway service to match the online UI limit.

---

## License

MIT
