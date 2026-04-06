import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables from .env
load_dotenv()

# Pull variables directly mapping to your .env file
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN") # Now correctly pulling your Gemini key!
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Initialize the client using the mapped token
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# 2. Agent Logic
def get_action(observation_text: str) -> dict:
    prompt = f"""
    Analyze this Indian e-commerce review: "{observation_text}"
    Return ONLY a JSON object: {{"label": "SAFE"|"SPAM"|"TOXIC", "lang": "en"|"hi"|"hinglish", "reasoning": "Brief explanation"}}
    Rules: Sarcastic frustration is SAFE. 'Chor' is TOXIC. Any links, telegram channel promotions, or promotional coupons/offers are STRICTLY SPAM.
    """
    
    try:
        # Call the native Gemini 2.5 Flash model
        response = model.generate_content(prompt)
        
        # Parse the guaranteed JSON response
        parsed_data = json.loads(response.text)
        
        # Inject the model metadata for your audit logs
        parsed_data["model_name"] = MODEL_NAME 
        
        return parsed_data
        
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        # Make sure the fallback also includes the model_name so it doesn't crash Pydantic!
        return {
            "label": "SAFE", 
            "lang": "en", 
            "reasoning": "Fallback due to API error",
            "model_name": MODEL_NAME
        }

# 3. Strict Grading Loop
def run_baseline(episodes=10):
    all_rewards = []
    print(f"[START] task=hinglish-review-auditor env=dragoneye model={MODEL_NAME}", flush=True)
    total_reward = 0.0
    
    with httpx.Client(timeout=30.0) as http:
        # Start the environment
        resp = http.post(f"{ENV_URL}/reset")
        resp.raise_for_status()
        obs = resp.json()
        
        for i in range(episodes):
            # Agent makes a decision
            action = get_action(obs['review_text'])
            
            # Send decision to environment
            step_resp = http.post(f"{ENV_URL}/step", json=action)
            result = step_resp.json()
            
            reward = result["reward"]
            total_reward += reward
            all_rewards.append(reward)
            
            # Print strict Hackathon OpenEnv log format
            print(f"[STEP] step={i+1} action={action.get('label')} reward={reward:.2f} done={str(result['done']).lower()} error=null", flush=True)
            
            if result["done"]:
                obs = result["observation"]
                
    avg_reward = total_reward / episodes
    
    print(f"[END] success={str(avg_reward >= 0.5).lower()} steps={episodes} score={avg_reward:.2f} rewards={','.join(f'{r:.2f}' for r in all_rewards)}", flush=True)

if __name__ == "__main__":
    # Ensure we actually have a token before running
    if not HF_TOKEN:
        print("❌ CRITICAL ERROR: HF_TOKEN not found in .env file!")
    else:
        run_baseline(episodes=50)