import os
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from .rewards import calculate_reward

# Load DragonEye .env configuration
load_dotenv()

AGENT_API_URL = os.getenv("AGENT_URL", "http://localhost:11434/api/generate")
AGENT_MODEL = os.getenv("AGENT_MODEL", "sarvam:2b")

# --- 1. Let Pydantic do the heavy lifting ---
class AgentDecision(BaseModel):
    label: str
    lang: str

def get_agent_decision(review_text: str) -> dict:
    """Calls the active model to moderate the review."""
    prompt = (
        f"Moderate this review. Return JSON ONLY with exactly these keys: "
        f"'label' (MUST be SAFE, SPAM, or TOXIC) and 'lang' (MUST be en, hi, or hinglish). "
        f"Review: {review_text}"
    )
    
    payload = {
        "model": AGENT_MODEL, 
        "prompt": prompt, 
        "stream": False,
        "format": "json" # <-- THE MAGIC BULLET: Forces pure JSON from Ollama
    }
    
    try:
        # httpx is the modern standard over 'requests'
        with httpx.Client(verify=False, timeout=15.0) as client:
            response = client.post(AGENT_API_URL, json=payload)
            response.raise_for_status()
            
            raw_output = response.json().get('response', '{}')
            
            # Pydantic validates it instantly. If the LLM messed up the keys, this catches it.
            valid_decision = AgentDecision.model_validate_json(raw_output)
            return valid_decision.model_dump()
            
    except ValidationError as ve:
        print(f"⚠️ LLM hallucinated bad keys: {ve}")
        return {"label": "SAFE", "lang": "error"}
    except Exception as e:
        print(f"⚠️ Agent Request Failed: {e}")
        return {"label": "SAFE", "lang": "error"}

def get_auditor_grade(review_text: str, agent_decision: dict, expected: dict) -> float:
    """Calculates the reward using the expected ground truth."""
    reward = calculate_reward(
        expected_label=expected.get('label', 'SAFE'),
        expected_lang=expected.get('lang', 'hinglish'),
        action_label=agent_decision.get('label'),
        action_lang=agent_decision.get('lang'),
        difficulty=expected.get('difficulty', 'easy')
    )
    return reward