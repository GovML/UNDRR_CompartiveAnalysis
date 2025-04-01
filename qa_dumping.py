import requests
import json

url = "http://localhost:11434/api/generate"  # Ollama API endpoint
model = "deepseek-r1:7b"  # Replace with your model name
prompt = "What is the capital of France?"
context = "The capital of France is"

payload = {
    "model": model,
    "prompt": context + prompt,  # Prepend context
    "stream": False
}

response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
print(response.json()['response'])