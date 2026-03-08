<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/3a0f13d8-1c60-4fdb-9a29-208d25ed982a" />


<p align="center">
  <p align="center"><strong>Real-time AI memory aid for people with dementia, face blindness, and memory impairment.</strong></p>
  <p align="center">
    <a href="#demo">Demo</a> •
    <a href="#features">Features</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#roadmap">Roadmap</a>
  </p>
</p>

---

> *"Because no one should forget who they love."*

Faces watches your camera and listens to your conversations in real time. When someone walks up to you, it recognizes their face and shows you their name, your relationship, and every important detail from past conversations — instantly, with zero effort on your part.

## The Problem

Over **55 million** people worldwide live with dementia, projected to hit **139 million by 2050**. Another **10 million** have prosopagnosia (face blindness), **2 million** live with aphasia, and nearly **40%** of people over 65 experience age-associated memory impairment. There is no real-time, frictionless tool that helps someone know *who is standing in front of them*.

## Demo

<a name="demo"></a>

>  *Demo video coming soon*

## Features

<a name="features"></a>

### Face Recognition — Always On
- Continuous real-time face detection and identification at full frame rate
- Handles **multiple faces simultaneously** in a single frame, each with independent labels
- Tracks people seamlessly as they move, turn, step out, and come back
- Persistent identity — recognizes the same person across sessions

### Conversational Learning
- Say **"My name is Sarah"** → name updates on screen instantly
- Say **"I'm your daughter"** → relationship badge changes live
- Mention **"doctor appointment on Tuesday"** → captured as a key detail
- No hardcoded phrases — the AI understands natural introductions and relationship declarations

### Smart Key Info Extraction
- Extracts dates, appointments, events, people, places, health info from natural speech
- Maintains a **living memory list** — corrects outdated info instead of duplicating
- Filters out small talk and filler — only stores what actually matters
- All details persist across sessions and server restarts

### Cross-Platform, Zero Setup
- Runs entirely in the browser — no install, no native dependencies
- Works on any device with a camera, microphone, and a browser
- Open a URL, allow permissions, done

## How It Works

<a name="how-it-works"></a>

1. **Camera** captures video → **face-api.js** detects and identifies faces client-side using 128-dim descriptors
2. **Microphone** streams raw PCM audio over WebSocket → custom audio gate filters silence → **Whisper** transcribes
3. **Zen API** (LLM) runs two parallel extraction tasks per transcription:
   - **Identity extraction** — names and relationships from natural speech
   - **Key info extraction** — dates, events, appointments, people, health details
4. Results are stored in **SQLite** and pushed to the frontend in real-time via WebSocket

## Getting Started

<a name="getting-started"></a>

### Prerequisites

- Python 3.9+
- pip
- A modern browser (Chrome recommended)
- AUTH API key ([get one here](https://opencode.ai))

### Installation

```bash
# Clone the repository
git clone https://github.com/abacushacks/adahacks-2026.git
cd adahacks-2026

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Run migrations
python3 manage.py migrate

# Set your AUTH API key
export _KEY="your--api-key"

# Start the server
python3 manage.py runserver
```

### Usage

1. Allow camera and microphone access
2. Click anywhere on the screen to start the audio pipeline
3. Faces are detected automatically — start talking and watch the magic happen

## Architecture

<a name="architecture"></a>

### Frontend
| File | Purpose |
|------|---------|
| `frontend/js/app.js` | Main app — camera setup, render loop, popup positioning, active face tracking |
| `frontend/js/face-tracker.js` | Face detection, recognition, descriptor matching, popup management |
| `frontend/js/audio-manager.js` | WebSocket connection, audio streaming, speech gate |
| `frontend/css/style.css` | Popup overlay styling, key info bullet list |

### Backend
| File | Purpose |
|------|---------|
| `backend/audio_stream/consumers.py` | WebSocket consumer — handles audio, face descriptors, transcription, extraction |
| `backend/audio_stream/zen_service.py` | Zen API (LLM) — name/relationship parsing, key info extraction |
| `backend/audio_stream/services.py` | Face matching service — Euclidean distance on 128-dim descriptors |
| `backend/audio_stream/models.py` | Face model — label, descriptor, name, relationship, metadata |
| `backend/audio_stream/constants.py` | Tunable parameters — buffer sizes, thresholds, Whisper config |

### AI Pipeline

The AI context engine is the core intelligence of Faces. Every transcription chunk triggers **two parallel LLM calls** via `asyncio.gather()`:

1. **Identity extraction** — detects name/relationship introductions from natural speech. No regex, no hardcoded patterns — the model understands conversational phrasing.

2. **Key info extraction** — receives the full current memory state and returns an updated, curated list. Replaces outdated entries, filters noise, only keeps what matters.

Both are race-condition-safe: the face recognition pipeline re-reads the database after its API calls return to avoid overwriting speech-derived updates.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Face Detection | face-api.js (SSD MobileNet + 128-dim descriptors) |
| Speech-to-Text | OpenAI Whisper (local, base.en model) |
| LLM | Zen API (OpenAI-compatible, kimi-k2.5) |
| Backend | Django + Django Channels + Daphne |
| Transport | WebSockets (raw PCM audio + JSON) |
| Database | SQLite |
| Frontend | Vanilla JavaScript + HTML5 Canvas + WebAudio API |

## Roadmap for the future

<a name="roadmap"></a>

-  AR glasses integration — context in your actual field of view
-  Lip tracking — supplement audio with visual speech recognition
-  Multi-language support
-  Caregiver dashboard — family members can add context before visits
-  Emotion & mood detection
-  On-device LLM inference — fully offline, no API dependency

## For our Community

This project was built for people who need it most:

| Condition | Affected Population |
|-----------|-------------------|
| Dementia | 55 million worldwide (139M by 2050) |
| Prosopagnosia (face blindness) | 10 million |
| Aphasia | 2 million (US alone) |
| Age-related memory impairment | ~40% of people over 65 |
| Unpaid dementia caregivers | 11 million (US alone) |

## License

See [LICENSE](LICENSE) for details.

---

**Built by [Abacus](https://github.com/abacushacks) at AdaHacks 2026**
