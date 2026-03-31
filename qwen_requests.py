import argparse
import base64
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description="Send one Qwen ASR request.")
parser.add_argument("--audio", required=True, help="Path to an audio file.")
parser.add_argument("--url", default="http://localhost:8061/v1/chat/completions")
args = parser.parse_args()

audio_path = Path(args.audio).expanduser().resolve()
if not audio_path.exists():
    raise SystemExit(f"Audio not found: {audio_path}")

with audio_path.open("rb") as f:
    encoded_data = base64.b64encode(f.read()).decode("utf-8")

audio_data_url = f"data:audio/mp3;base64,{encoded_data}"

url = args.url
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio_data_url
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=300)
response.raise_for_status()
content = response.json()['choices'][0]['message']['content']
print(content)

# parse ASR output if you want
from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
print(language)
print(text)
