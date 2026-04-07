import json
import os
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel

from env_server.core.evaluator import get_auditor_grade
from env_server.core.models import Observation, Action, StepResult

app = FastAPI(title="DragonEye-Auditor: OpenEnv")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure this path correctly points to where your data folder is!
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews_v1.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    REAL_REVIEWS = json.load(f)

current_review_index = 0

# --- FIX 1: Map dataset difficulty to the YAML Task IDs ---
TASK_MAPPING = {
    "easy": "task_1_language_id",
    "medium": "task_2_basic_moderation",
    "hard": "task_3_sarcasm_slang"
}

@app.get("/state", response_model=Observation)
async def get_state():
    review = REAL_REVIEWS[current_review_index]
    # Grab the correct ID based on the current review's difficulty
    current_task_id = TASK_MAPPING.get(review.get("difficulty", "easy"), "task_2_basic_moderation")
    
    return Observation(
        session_id=str(review.get("id", current_review_index)),
        review_text=review["text"],
        task_id=current_task_id # <-- FIX 1: Added mandatory task_id
    )

import random

@app.post("/reset", response_model=Observation)
async def reset_environment():
    global current_review_index
    # FIX: Drop the agent at a random point in the dataset so it sees 
    # different difficulty levels, languages, and labels across tasks!
    current_review_index = random.randint(0, len(REAL_REVIEWS) - 1)
    
    return await get_state()
    
@app.post("/step", response_model=StepResult)
async def step_environment(action: Action):
    global current_review_index 
    
    expected_truth = REAL_REVIEWS[current_review_index]
    
    reward = get_auditor_grade(
        review_text=expected_truth["text"],
        agent_decision=action.model_dump(),
        expected=expected_truth
    ) 
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": action.model_name,
        "review_id": expected_truth.get("id", current_review_index),
        "expected_label": expected_truth.get("label"),
        "agent_label": action.label,
        "reward": reward,
        "reasoning": action.reasoning
    }
    
    log_path = "/tmp/audit_results.jsonl" if os.getenv("SPACE_ID") else "audit_results.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    # --- FIX 3: Removed the bad print statement here ---

    # --- FIX 2: Made progression sequential instead of random ---
    current_review_index = (current_review_index + 1) % len(REAL_REVIEWS)
    
    return StepResult(
        observation=await get_state(), # <-- Returns the new state cleanly
        reward=reward,
        done=True, 
        info={"message": f"Agent scored: {reward:.2f}/1.0", "gt_label": expected_truth.get("label")}
    )
def main():
    import uvicorn
    # This matches the HF Spaces default port
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

# The validator is explicitly looking for this standard Python block!
if __name__ == "__main__":
    main()