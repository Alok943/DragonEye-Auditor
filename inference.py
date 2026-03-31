import os
import json
import httpx
from openai import OpenAI

# ==========================================
# 1. Hackathon Required Variables
# ==========================================
# The judges will inject these automatically during evaluation.
# We provide defaults so you can test it locally.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1") # Defaulting to local Ollama's OpenAI-compatible endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:7b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "sk-local-dummy-key")

# The environment URL (Local for now, HF Space URL later)
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Initialize the OpenAI Client exactly as the rules dictate
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ==========================================
# 2. AI Brain Logic
# ==========================================
def get_action(observation_text: str) -> dict:
    """Calls the LLM using the OpenAI spec to get the next action."""
    prompt = f"""
    Analyze this Indian e-commerce review: "{observation_text}"
    
    Return ONLY a JSON object:
    {{
        "label": "SAFE", "SPAM", or "TOXIC",
        "lang": "en", "hi", or "hinglish",
        "reasoning": "Brief explanation"
    }}
    
    Rules: Sarcastic frustration is SAFE. 'Chor' or links are TOXIC/SPAM.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Hinglish e-commerce auditor outputting strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            # If the model supports JSON mode natively, uncomment the next line:
            # response_format={ "type": "json_object" }
        )
        
        # Parse the string response into a Python dictionary
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        # Fallback to prevent crash during automated grading
        return {"label": "SAFE", "lang": "en", "reasoning": "Fallback action"}

# ==========================================
# 3. The OpenEnv Evaluation Loop
# ==========================================
def run_baseline(episodes=5):
    print(f"🚀 Starting Baseline Inference with {MODEL_NAME}...")
    total_reward = 0.0
    
    with httpx.Client(timeout=30.0) as http:
        # Reset the environment to start
        resp = http.post(f"{ENV_URL}/reset")
        resp.raise_for_status()
        obs = resp.json()
        
        for i in range(episodes):
            print(f"\n📝 Review {i+1}: {obs['review_text']}")
            
            # Agent thinks
            action = get_action(obs['review_text'])
            print(f"🤖 Action: {action['label']} | {action['lang']}")
            
            # Agent steps in environment
            step_resp = http.post(f"{ENV_URL}/step", json=action)
            result = step_resp.json()
            
            # Calculate metrics
            reward = result["reward"]
            total_reward += reward
            print(f"💰 Reward: {reward} | Actual Label: {result['info'].get('actual_label')}")
            
            if result["done"]:
                obs = result["observation"]
                
    avg_reward = total_reward / episodes
    print("\n" + "="*40)
    print(f"🏆 BASELINE COMPLETE")
    print(f"📊 Average Reward Score: {avg_reward:.2f} / 1.00")
    print("="*40)

if __name__ == "__main__":
    run_baseline(episodes=10)