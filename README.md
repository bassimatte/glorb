# GLORB

**Organic sound generator.** Synthesizes random sequences of blips, blops, zaps, clicks and bloops — each one unique, each one alive.

---

## What it does

Glorb generates short electronic sounds with an organic, analog feel using a layered synthesis engine:

- **FM synthesis** — frequency modulation for metallic, complex tones
- **Additive harmonics** — stacked partials for bright, bell-like textures
- **Chirps** — frequency sweeps for classic blip/zap shapes
- **Pitch wobble** — subtle LFO vibrato that removes the frozen, robotic feel
- **Tanh saturation** — analog warmth and soft clipping
- **Noise floor** — tiny circuit hiss for air and texture
- **Comb reverb + echo** — space and depth
- **Haas stereo widening** — natural stereo image via L/R micro-delay

Every run produces a different sequence. Output is normalized to −0.1 dBFS.

---

## Web interface

```bash
python server.py
```

Open **http://localhost:5000** — set duration, pick quality, hit Generate.

The UI shows a live waveform, plays audio in the browser, and lets you download the WAV.

![Glorb UI](https://raw.githubusercontent.com/bassimatte/glorb/main/docs/screenshot.png)

---

## CLI

```bash
# Default: 10s, high quality → blipblop.wav
python main.py

# Set duration
python main.py -d 30

# Custom output file
python main.py -d 60 -o my_glorb.wav

# Set quality
python main.py -q studio

# Render and play immediately
python main.py -d 10 --play
```

### Quality presets

| Flag | Sample rate | Bit depth | Use case |
|---|---|---|---|
| `standard` | 44 100 Hz | 16-bit PCM | Small file size |
| `high` *(default)* | 44 100 Hz | 24-bit PCM | CD+ quality |
| `studio` | 48 000 Hz | 24-bit PCM | Pro audio / video |
| `float` | 44 100 Hz | 32-bit float | Maximum precision |

---

## Installation

```bash
git clone https://github.com/bassimatte/glorb.git
cd glorb
pip install numpy sounddevice soundfile scipy flask
```

### Requirements

- Python 3.10+
- `numpy`
- `sounddevice`
- `soundfile`
- `scipy`
- `flask` *(web interface only)*

---

## Project structure

```
glorb/
├── main.py              # Synthesis engine + CLI
├── server.py            # Flask web server
└── templates/
    └── index.html       # Minimal dark web UI
```

---

## License

MIT
