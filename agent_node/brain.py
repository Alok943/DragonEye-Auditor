import os
import re
import requests
import json
from .rewards import calculate_reward

# Pull from Environment Variables (Crucial for the automated Judge pipeline)
# It defaults to Archit's local Ollama if the env var isn't found.
AGENT_API_URL = os.getenv("AGENT_URL", "http://localhost:11434/api/generate")
AGENT_MODEL = os.getenv("AGENT_MODEL", "sarvam:2b")

def clean_and_parse_json(raw_text: str) -> dict:
    """Strips Markdown formatting and safely parses LLM JSON outputs."""
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        clean_json = match.group(1)
    else:
        clean_json = raw_text.strip().strip("`").replace("json", "", 1).strip()
    
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError:
        # Fallback to prevent environment crashes during automated eval
        return {"label": "SAFE", "lang": "unknown"}

def get_agent_decision(review_text: str) -> dict:
    """Calls the active model to moderate the review."""
    prompt = (
        f"Moderate this Hinglish review. Return JSON ONLY: "
        f"{{'label': 'SAFE'|'ABUSE', 'lang': 'hindi'|'english'|'hinglish'}}. "
        f"Review: {review_text}"
    )
    
    try:
        response = requests.post(
            AGENT_API_URL, 
            json={"model": AGENT_MODEL, "prompt": prompt, "stream": False},
            timeout=15
        )
        response.raise_for_status()
        raw_output = response.json().get('response', '{}')
        return clean_and_parse_json(raw_output)
    except Exception as e:
        print(f"Agent Request Failed: {e}")
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