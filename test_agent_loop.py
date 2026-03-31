import requests
import json
import time

# Configuration
ENV_URL = "http://localhost:8000"  # Archit's FastAPI Environment
AGENT_MODEL_URL = "http://localhost:11434/api/generate" # Local Ollama

def call_local_agent(review_text):
    """Simulates the Agent (Sarvam 2B) making a decision."""
    prompt = (
        f"Analyze this Hinglish review. Respond ONLY in JSON format: "
        f"{{\"label\": \"SAFE\" or \"ABUSE\", \"lang\": \"hindi\", \"english\", or \"hinglish\"}}. "
        f"Review: {review_text}"
    )
    
    try:
        response = requests.post(AGENT_MODEL_URL, json={
            "model": "sarvam:2b",
            "prompt": prompt,
            "stream": False
        }, timeout=10)
        
        # Quick and dirty JSON extraction for testing
        result = response.json().get('response', '{}')
        # In production, use the 'clean_json_response' utility we built!
        return json.loads(result)
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        return {"label": "SAFE", "lang": "hinglish"}

def run_test_session(episodes=3):
    """The standard OpenEnv reset -> step loop."""
    print(f"🚀 Starting DragonEye-Auditor Test ({episodes} episodes)\n" + "="*50)
    
    for i in range(episodes):
        print(f"\n--- Episode {i+1} ---")
        
        # 1. RESET the environment to get a new review
        reset_resp = requests.post(f"{ENV_URL}/reset")
        state = reset_resp.json()
        review = state.get("review_text")
        print(f"📥 Env Review: {review}")

        # 2. AGENT processes the review
        start_time = time.time()
        action = call_local_agent(review)
        latency = round(time.time() - start_time, 2)
        print(f"🧠 Agent Decision: {action['label']} ({action['lang']}) | Latency: {latency}s")

        # 3. STEP the environment to get the reward
        step_resp = requests.post(f"{ENV_URL}/step", json=action)
        result = step_resp.json()

        # 4. RESULTS
        reward = result.get("reward")
        info = result.get("info", "No info")
        
        if reward >= 0.8:
            print(f"✅ Reward: {reward} - Great job!")
        elif reward > 0.0:
            print(f"⚠️ Reward: {reward} - Partial Credit.")
        else:
            print(f"❌ Reward: {reward} - Failed.")
            
        print(f"📝 Info: {info}")

if __name__ == "__main__":
    run_test_session()