# GLORB

**Organic sound generator** by Matteo Bassi. Synthesizes random sequences of blips, blops, zaps, clicks and bloops — each one unique, each one alive.

🌐 **Live:** https://bassimatte.github.io/glorb/

---

## What it does

Glorb generates electronic sounds with an organic, analog feel across **17 sound worlds**:

| Mode | Description |
|---|---|
| **Glorb** | Abstract blip/bloop electronic tones |
| **Retro** | 8-bit chiptune square and pulse waves |
| **Nature** | Rain, fire, insects — procedural ambient textures |
| **Sci-Fi** | Phasers, warp drives, laser bursts |
| **Haptic** | Tactile vibration pulses and clicks |
| **Radio** | AM/FM static, morse, transmission artefacts |
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

Features: 6 colour themes, live waveform, audio-reactive dot-grid background, loop playback, WAV download.

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

## Installation

```bash
git clone https://github.com/bassimatte/glorb.git
cd glorb
pip install numpy sounddevice soundfile scipy flask flask-cors
```

---

## Deployment

- **Frontend** (GitHub Pages): `docs/index.html` — served statically
- **Backend** (Render.com): Flask + gunicorn via Docker — `render.yaml` included

---

## License

MIT
