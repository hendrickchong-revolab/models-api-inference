import os
import sys
import requests


def main() -> int:
    api_base = os.getenv("LITELLM_URL", "http://localhost:4000")
    model = os.getenv("LITELLM_MODEL", "fireredasr2s_aed")
    api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
    audio_path = "./sample.wav"

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return 1

    url = f"{api_base}/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        data = {
            "model": model,
            "response_format": "json",
        }
        headers = {"Authorization": f"Bearer {api_key}"}  # API key passed in headers
        response = requests.post(url, files=files, data=data, headers=headers, timeout=300)

    print(response.status_code)
    print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
