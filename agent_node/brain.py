import requests
import json
from .rewards import calculate_reward # Import your new shaped reward logic

def get_agent_decision(review_text: str):
    """Calls the local Sarvam 2B model."""
    prompt = f"Moderate this Hinglish review. Return JSON: {{'label': 'SAFE'|'ABUSE', 'lang': 'hindi'|'english'|'hinglish'}}. Review: {review_text}"
    
    response = requests.post("http://localhost:11434/api/generate", 
        json={"model": "sarvam:2b", "prompt": prompt, "stream": False})
    
    # Use the JSON cleaner we discussed
    return clean_and_parse_json(response.json()['response'])

def get_auditor_grade(review_text: str, agent_decision: dict, expected: dict):
    """Calculates the reward using Llama 8B as the referee."""
    # This calls your rewards.py logic
    reward = calculate_reward(
        expected_label=expected['label'],
        expected_lang=expected['lang'],
        action_label=agent_decision.get('label'),
        action_lang=agent_decision.get('lang'),
        difficulty=expected.get('difficulty', 'easy')
    )
    return reward