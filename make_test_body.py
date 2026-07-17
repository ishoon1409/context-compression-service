import json
import sys

stream = "--stream" in sys.argv

body = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "user", "content": "Explain quantum computing. " * 600}
    ],
    "stream": stream
}

filename = "body_long_stream.json" if stream else "body_long.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(body, f)

print(f"Wrote {filename}")