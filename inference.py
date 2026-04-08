import os
import json
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemini-2.5-flash-lite")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://alok8732-dragoneye-auditor.hf.space")
BENCHMARK = "DragonEye-Auditor"

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    print("CRITICAL ERROR: Missing required environment variables!", flush=True)
    exit(1)
    
# OpenAI client — used for ALL LLM calls as required by spec
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)
from typing import TypedDict


class TaskConfig(TypedDict):
    task_id: str
    episodes: int
    
TASK_CONFIGS: list[TaskConfig] = [
    {"task_id": "task_1_language_id",         "episodes": 30},
    {"task_id": "task_2_basic_moderation",  "episodes": 30},
    {"task_id": "task_3_sarcasm_slang",     "episodes": 30},
]

def get_action(observation_text: str) -> dict:
    prompt = (
    f'Analyze this Indian e-commerce review: "{observation_text}"\n\n'

    'Return ONLY a valid JSON object with exactly these keys:\n'
    '{"label": "SAFE"|"SPAM"|"TOXIC", "lang": "en"|"hi"|"hinglish", "nuance_detected": true|false, "reasoning": "Brief explanation"}\n\n'

    '====================\n'
    'CLASSIFICATION RULES:\n'
    '====================\n'

    '1. SAFE:\n'
    '- All normal reviews (positive OR negative)\n'
    '- Complaints, frustration, bad experiences\n'
    '- Sarcasm or indirect criticism WITHOUT attacking people\n\n'

    '2. TOXIC:\n'
    '- Direct insults, abusive language, harassment, or threats\n'
    '- Includes strong accusations targeting people, sellers, or companies\n'
    '  (e.g., "tum log chor ho", "fraud company", "you are scammers")\n'
    '- Includes demeaning language (e.g., "idiot seller", "useless people")\n'
    '- Includes threats (e.g., "refund do warna dekh lena")\n'
    '- If the statement attacks a person/company → TOXIC\n'
    '- Complaints, frustration, or slang (e.g., "ghanta", "bakwaas experience") are NOT TOXIC unless directly targeting a person/company with abuse or threats\n'
    '- If it only describes a bad experience → SAFE\n\n'

    '3. SPAM:\n'
    '- Promotions, ads, referral links, coupons, contact numbers\n'
    '- If no clear promotional intent → DO NOT mark as SPAM\n\n'

    '====================\n'
    'STRICT DECISION RULES:\n'
    '====================\n'

    '- Sarcasm, jokes, or indirect criticism are ALWAYS SAFE unless targeting people\n'
    '- Negative tone alone does NOT mean TOXIC\n'
    '- Complaint about product/service ≠ TOXIC\n'
    '- Attack on people/company → TOXIC\n'
    '- If unsure between SAFE and TOXIC → ALWAYS choose SAFE\n'
    '- If unsure between SAFE and SPAM → choose SAFE\n\n'

    '====================\n'
    'LANGUAGE RULES:\n'
    '====================\n'
    '- "en" = Pure English\n'
    '- "hi" = Pure Hindi (even if written in English script)\n'
    '- If the sentence is mostly English with minor Hindi words → classify as "en", NOT "hinglish"\n'
    '- "hinglish" = mix of Hindi + English\n\n'

    '====================\n'
    'NUANCE DETECTION:\n'
    '====================\n'
    '- true → sarcasm, irony, slang, or indirect tone\n'
    '- false → direct, literal statement\n'
    '- If positive words are used to express a negative experience → nuance_detected MUST be true\n'
    '- SPAM content should always have nuance_detected = false\n'
    '- Look for contradiction between tone and outcome\n\n'

    '====================\n'
    'CONTRAST EXAMPLES:\n'
    '====================\n'

    '# SARCASTIC (nuance_detected = true)\n'
    'Input: "वाह क्या service hai, bina bataye order cancel kar diya"\n'
    'Output: {"label": "SAFE", "lang": "hinglish", "nuance_detected": true, "reasoning": "Sarcastic praise expressing frustration"}\n\n'

    'Input: "Amazing quality, ek use mein hi toot gaya"\n'
    'Output: {"label": "SAFE", "lang": "hinglish", "nuance_detected": true, "reasoning": "Positive wording contradicts outcome"}\n\n'

    '# SAFE (strong complaints)\n'
    'Input: "Fake product mila, bilkul expected jaisa nahi tha"\n'
    'Output: {"label": "SAFE", "lang": "hi", "nuance_detected": false, "reasoning": "Complaint about product quality"}\n\n'

    'Input: "I feel cheated after buying this, total waste of money"\n'
    'Output: {"label": "SAFE", "lang": "en", "nuance_detected": false, "reasoning": "Expression of dissatisfaction, not attacking people"}\n\n'

    '# TOXIC\n'
    'Input: "Tum log chor ho, public ko loot rahe hain saale"\n'
    
    'Output: {"label": "TOXIC", "lang": "hinglish", "nuance_detected": false, "reasoning": "Direct accusation and abusive language targeting people"}\n\n'

    'Input: "This company is a scam, they are cheating customers"\n'
    'Output: {"label": "TOXIC", "lang": "en", "nuance_detected": false, "reasoning": "Direct accusation targeting company"}\n\n'

    '# SPAM\n'
    'Input: "Earn money fast, call now 9876543210"\n'
    'Output: {"label": "SPAM", "lang": "en", "nuance_detected": false, "reasoning": "Promotional content"}\n\n'

    '====================\n'
    'IMPORTANT:\n'
    '====================\n'
    '- Return ONLY valid JSON starting with { and ending with }\n'
    '- No extra text, no markdown, no explanation outside JSON\n'
)
    import time
    for attempt in range(3):  # 3 attempts total
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            parsed["model_name"] = MODEL_NAME
            return parsed
        except Exception as e:
            print(f"API Error (attempt {attempt+1}/3): {e}", flush=True)
            if attempt < 2:
                time.sleep(15)
    
    return {
            "label": "SAFE",
            "lang": "en",
            "nuance_detected": False,
            "reasoning": "Fallback due to API error",
            "model_name": MODEL_NAME
        }

def run_task(task_id: str, episodes: int, http: httpx.Client):
    all_rewards = []
    total_reward = 0.0

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    # Reset env before each task
    resp = http.post(f"{ENV_URL}/reset", json={"task_id": task_id})
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
            f"[STEP] step={i+1} action={json.dumps(action)} "
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
    

    with httpx.Client(timeout=60.0) as http:
        scores = {}
        for task_cfg in TASK_CONFIGS:
            score = run_task(task_cfg["task_id"], task_cfg["episodes"], http)
            scores[task_cfg["task_id"]] = score

    