vLLM inference model containers
================================

This workspace serves multiple ASR models behind a single LiteLLM gateway.
Your data pipeline should only talk to the LiteLLM gateway and pass the model
alias defined in config.json. The alias is how you select the model.

Quick mental model
------------------
- config.json defines which models run, on which GPU/port, and their alias.
- start_models.py reads config.json, generates config.yaml, and writes
	docker-compose.yml with all enabled services.
- LiteLLM exposes a single OpenAI‑compatible API at http://localhost:4000.
- You choose the model using the alias (the "name" field in config.json).

How model selection works
-------------------------
If both whisper and whisper_ms_precise are enabled, you select them by passing
the alias:
- model=whisper
- model=whisper_ms_precise

Model config template (config.json)
-----------------------------------
Each enabled model entry looks like this:
```json
{
	"name": "whisper",
	"enabled": true,
	"compose_dir": "vllm_inference_models/whisper",
	"model_name": "openai/whisper-large-v3",
	"port": 7085,
	"gpu_id": 1
}
```

Required fields
---------------
- name: alias you will pass in requests (model=...)
- enabled: true/false
- compose_dir: model container directory
- model_name: actual HF model id
- port: container port
- gpu_id: must be one of [1, 2, 3]

Gateway endpoints
-----------------
Base URL: http://localhost:4000

List available models
```bash
curl -s -H "Authorization: Bearer sk-1234" http://localhost:4000/v1/models | cat
```

Audio transcription (multipart)
```bash
curl -s http://localhost:4000/v1/audio/transcriptions \
	-H "Authorization: Bearer sk-1234" \
	-F file=@/path/to/audio.wav \
	-F model=whisper \
	-F response_format=json
```

Minimal request template (multipart)
------------------------------------
Fields required for /v1/audio/transcriptions:
```
file: audio file (wav/mp3/etc.)
model: alias from config.json (e.g. whisper, qwen_asr, glm_asr)
```

Common optional fields (if supported by the model)
--------------------------------------------------
- response_format: json | verbose_json | text
- language: e.g. "en"
- prompt: transcription hint
- temperature: float
- timestamp_granularities[]: "segment" or "word"

Python request template (requests)
----------------------------------
```python
import requests

url = "http://localhost:4000/v1/audio/transcriptions"
headers = {"Authorization": "Bearer sk-1234"}

with open("/path/to/audio.wav", "rb") as f:
		files = {"file": ("audio.wav", f)}
		data = {
				"model": "whisper",
				"response_format": "json",
				# Optional fields:
				# "language": "en",
				# "prompt": "Context hint",
				# "temperature": 0.2,
		}
		r = requests.post(url, headers=headers, files=files, data=data, timeout=300)

print(r.status_code)
print(r.text)
```

Per-model notes
---------------
Whisper (vllm_inference_models/whisper)
- Uses python -m app.main
- Supports OpenAI /v1/audio/transcriptions

Qwen ASR (vllm_inference_models/qwen_asr)
- Uses vllm serve
- Supports OpenAI /v1/audio/transcriptions

GLM ASR (vllm_inference_models/glm_asr)
- Uses vllm serve
- Supports OpenAI /v1/audio/transcriptions

WhisperX (vllm_inference_models/whisperx)
- CUDA only
- Runs a separate FastAPI server
- Uses /v1/audio/transcriptions endpoint compatible with LiteLLM routing

FireRedASR2S AED (vllm_inference_models/fireredasr2s_aed)
- Runs a FastAPI OpenAI-compatible wrapper (non-vLLM backend)
- Uses FireRedASR model code from fireredasr-api/fireredasr
- Requires model files mounted from fireredasr-api/pretrained_models
- Alias example: fireredasr2s_aed

How to pass extra options
-------------------------
Pass any additional OpenAI transcription fields in the multipart form. If a
model ignores a field, the server will log a warning and continue. The safest
set of fields is: file, model, response_format.

If you need per‑model overrides (dtype, gpu memory utilization, etc.), add
extra keys in config.json and have start_models.py map them into the generated
docker-compose.yml.
