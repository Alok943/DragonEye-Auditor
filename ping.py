import os
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("AGENT_URL")  # should be https://xxxx.ngrok-free.app/api/generate

print(f"📡 Pinging Archit's Laptop at: {url}")

headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "DragonEye-Auditor/1.0"
}

payload = {
    "model": "llama3.1:latest",
    "prompt": "Reply with exactly: Connection Successful",
    "stream": False
}

try:
    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=60,        # LLM inference takes time, 15s was too short
        verify=False
    )

    if "text/html" in response.headers.get("Content-Type", ""):
        print("❌ Ngrok still blocking. Raw HTML:")
        print(response.text[:300])

    elif response.status_code != 200:
        print(f"❌ HTTP {response.status_code}: {response.text[:300]}")

    else:
        data = response.json()
        print("✅ Success! Archit's GPU responded:", data.get("response", "No response key"))

except requests.exceptions.Timeout:
    print("❌ Timeout — model took too long. Try increasing timeout or check if model is loaded.")
except Exception as e:
    print("❌ Python Crash:", e)