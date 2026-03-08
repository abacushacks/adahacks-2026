# Implementation Plan

## Step-by-step plan followed
1. Created a clean monorepo layout with separate `backend/` and `frontend/` applications plus a shared root `implementation.md`.
2. Configured a Django project with SQLite defaults, ASGI support, and Django Channels so the backend can serve HTTP health checks and WebSocket sessions from one runtime.
3. Implemented the required SQLite schema with `Person` and `Memory` models and added the initial migration.
4. Added a websocket consumer that accepts live media messages, speaker-reference calibration, and session lifecycle events.
5. Implemented a DeepFace-based frame pipeline that:
   - decodes incoming JPEG frames,
   - detects faces,
   - enforces single-face processing by selecting only the largest face,
   - extracts the facial embedding,
   - estimates mouth motion from the lower facial region.
6. Implemented audio processing around Whisper that:
   - decodes streamed PCM chunks,
   - resamples audio for Whisper,
   - filters out the patient voice through speaker fingerprint comparison,
   - only buffers audio when recent mouth motion indicates that the tracked subject is speaking,
   - transcribes approved chunks.
7. Implemented transcript parsing logic for:
   - subject name extraction for new-person enrollment,
   - birthday facts,
   - relationship facts,
   - life-event facts.
8. Implemented the realtime session state machine that handles:
   - unknown-face temporary embedding archiving,
   - person creation after name confirmation,
   - returning-face recognition and metadata packaging,
   - immediate `Memory` writes for newly detected conversational facts,
   - tracking-loss notifications that remove the UI popup.
9. Built a React + TypeScript frontend that:
   - requests webcam and microphone access,
   - records a patient voice calibration sample,
   - streams video frames and raw audio chunks over WebSockets,
   - renders the live video feed,
   - mounts a metadata popup to the right side of a recognized face and snaps it to the left when the right edge lacks space.
10. Added dependency manifests, styling, and local validation steps so the codebase is ready for dependency installation and runtime startup.

## Folder and file structure
```text
adahacks-2026/
├── .gitignore
├── LICENSE
├── implementation.md
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── config/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── routing.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── memory_aid/
│       ├── __init__.py
│       ├── admin.py
│       ├── apps.py
│       ├── consumers.py
│       ├── models.py
│       ├── urls.py
│       ├── views.py
│       ├── migrations/
│       │   ├── 0001_initial.py
│       │   └── __init__.py
│       └── services/
│           ├── __init__.py
│           ├── audio_service.py
│           ├── face_service.py
│           ├── session_state.py
│           ├── speaker_id.py
│           └── speech_parser.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    ├── public/
    │   └── audio-processor.js
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── styles.css
        ├── vite-env.d.ts
        ├── components/
        │   ├── MetadataPopup.tsx
        │   └── VideoStage.tsx
        ├── hooks/
        │   └── useStreamingSession.ts
        ├── types/
        │   └── protocol.ts
        └── utils/
            ├── audio.ts
            └── video.ts
```

## Validation performed
- Python backend syntax validation completed with `python3 -m compileall backend`.
- Frontend TypeScript validation is blocked until `frontend` dependencies are installed locally; the remaining compiler output is unresolved React/Vite package typing rather than project syntax failures.
