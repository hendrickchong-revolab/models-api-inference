WhisperX API Server (CUDA only)

This folder is self-contained. It includes the WhisperX API server source and
builds the CUDA image locally.

Run

- From this folder:
  - docker compose up --build

Dynamic batching (optional)

Set these env vars to enable request batching:
- BATCH_MAX_SIZE (default: 10)
- BATCH_WINDOW_MS (default: 100)

Request examples (localhost:8090)

Audio/transcriptions:
```bash
curl -s http://localhost:8090/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F model=whisper-1 \
  -F response_format=json
```

Audio/translations:
```bash
curl -s http://localhost:8090/v1/audio/translations \
  -F file=@/path/to/audio.wav \
  -F model=whisper-1 \
  -F response_format=json
```

