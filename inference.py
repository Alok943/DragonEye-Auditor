import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# OpenAI client — used for ALL LLM calls as required by spec
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASK_CONFIGS = [
    {"task_id": "task_1_language_id",    "difficulty": "easy",   "episodes": 5},
    {"task_id": "task_2_basic_moderation", "difficulty": "medium", "episodes": 5},
    {"task_id": "task_3_sarcasm_slang",  "difficulty": "hard",   "episodes": 5},
]

def get_action(observation_text: str) -> dict:
    prompt = (
        f'Analyze this Indian e-commerce review: "{observation_text}"\n'
        'Return ONLY a JSON object with exactly these keys:\n'
        '{"label": "SAFE"|"SPAM"|"TOXIC", "lang": "en"|"hi"|"hinglish", "reasoning": "Brief explanation"}\n'
        'Rules: Sarcastic frustration = SAFE. "Chor" = TOXIC. Promo links/coupon codes = SPAM.'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        parsed["model_name"] = MODEL_NAME
        return parsed
    except Exception as e:
        print(f"API Error: {e}", flush=True)
        return {
            "label": "SAFE",
            "lang": "en",
            "reasoning": "Fallback due to API error",
            "model_name": MODEL_NAME
        }

def run_task(task_id: str, episodes: int, http: httpx.Client):
    all_rewards = []
    total_reward = 0.0

    print(f"[START] task={task_id} env=dragoneye model={MODEL_NAME}", flush=True)

    # Reset env before each task
    resp = http.post(f"{ENV_URL}/reset")
    resp.raise_for_status()
    obs = resp.json()

    for i in range(episodes):
        action = get_action(obs["review_text"])

        step_resp = http.post(f"{ENV_URL}/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result["reward"]
        total_reward += reward
        all_rewards.append(reward)

        print(
            f"[STEP] step={i+1} action={action.get('label')} "
            f"reward={reward:.2f} done={str(result['done']).lower()} error=null",
            flush=True
        )

        # Always grab next observation from result
        obs = result["observation"]

    avg_reward = total_reward / episodes
    print(
        f"[END] success={str(avg_reward >= 0.5).lower()} steps={episodes} "
        f"score={avg_reward:.2f} rewards={','.join(f'{r:.2f}' for r in all_rewards)}",
        flush=True
    )
    return avg_reward

if __name__ == "__main__":
    if not HF_TOKEN:
        print("CRITICAL ERROR: HF_TOKEN not found in .env file!", flush=True)
        exit(1)

    with httpx.Client(timeout=60.0) as http:
        scores = {}
        for task_cfg in TASK_CONFIGS:
            score = run_task(task_cfg["task_id"], task_cfg["episodes"], http)
            scores[task_cfg["task_id"]] = score

    overall = sum(scores.values()) / len(scores)
    print(f"\nOverall average score: {overall:.2f}", flush=True)