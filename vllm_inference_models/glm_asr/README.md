GLM-ASR vLLM container

Change MODEL_NAME in .env to switch models.
Exposes OpenAI-compatible endpoints (chat/completions and audio/transcriptions)
via vLLM serve.

API request examples (localhost:8090)

Chat/completions:
```bash
curl -s http://localhost:8090/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "YOUR_MODEL_NAME",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Transcribe the audio and summarize the key points."}
		],
		"temperature": 0.2
	}'
```

Audio/transcriptions:
```bash
curl -s http://localhost:8090/v1/audio/transcriptions \
	-F file=@/path/to/audio.wav \
	-F model=YOUR_MODEL_NAME \
	-F response_format=json
```

Request message format (chat/completions):
```json
{
	"model": "YOUR_MODEL_NAME",
	"messages": [
		{"role": "system", "content": "..."},
		{"role": "user", "content": "..."}
	]
}
```
