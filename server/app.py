import json
import os
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel

from env_server.core.evaluator import get_auditor_grade
from env_server.core.models import Observation, Action, StepResult

app = FastAPI(title="DragonEye-Auditor: OpenEnv")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data", "reviews_v1.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    REAL_REVIEWS = json.load(f)

import random
random.seed(42)
random.shuffle(REAL_REVIEWS)

current_review_index = int(os.getenv("START_INDEX", 0))

def get_task_id(review):
    if review.get("nuance", False):
        return "task_3_sarcasm_slang"
    elif review.get("label") in ["TOXIC", "SPAM"]:
        return "task_2_basic_moderation"
    else:
        return "task_1_language_id"


@app.get("/state", response_model=Observation)
async def get_state():
    review = REAL_REVIEWS[current_review_index]
    # Grab the correct ID based on the current review's difficulty
   
    
    return Observation(
        session_id=str(review.get("id", current_review_index)),
        review_text=review["text"],
        task_id = get_task_id(review)
    )

import random

from typing import Optional

class ResetRequest(BaseModel):
    task_id: str = "task_1_language_id"

@app.post("/reset", response_model=Observation)


async def reset_environment(req: Optional[ResetRequest] = None):
    global current_review_index
    
    task_id = req.task_id if req else "task_1_language_id"
    
    # Filter dataset based on task
    filtered_indices = [
        i for i, r in enumerate(REAL_REVIEWS)
        if get_task_id(r) == task_id
    ]
    
    # Randomly pick a starting index
    if filtered_indices:
        current_review_index = random.choice(filtered_indices)
    else:
        current_review_index = random.randint(0, len(REAL_REVIEWS) - 1)
    
    return await get_state()
    
@app.post("/step", response_model=StepResult)
async def step_environment(action: Action):
    global current_review_index 
    
    expected_truth = REAL_REVIEWS[current_review_index]
    
    # Determine task
    current_task_id = get_task_id(expected_truth)
    
    # Compute reward
    reward = get_auditor_grade(
        task_id=current_task_id,
        review_text=expected_truth["text"],
        agent_decision=action.model_dump(),
        expected=expected_truth
    )
    
    # Logging (keep as is)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": action.model_name,
        "review_id": expected_truth.get("id", current_review_index),
        "expected_label": expected_truth.get("label"),
        "agent_label": action.label,
        "expected_lang": expected_truth.get("lang", "unknown"),
        "agent_lang": action.lang,
        "expected_nuance": expected_truth.get("nuance", False),
        "agent_nuance": action.nuance_detected,
        "reward": reward,
        "reasoning": action.reasoning
    }
    
    log_path = "/tmp/audit_results.jsonl" if os.getenv("SPACE_ID") else "audit_results.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ✅ RANDOM TASK-CONDITIONED SAMPLING (CORRECT)
    filtered_indices = [
        i for i, r in enumerate(REAL_REVIEWS)
       if get_task_id(r) == current_task_id
    ]

    if filtered_indices:
        current_review_index = random.choice(filtered_indices)
    else:
        current_review_index = random.randint(0, len(REAL_REVIEWS) - 1)

    # Next observation
    next_obs = await get_state()

    return StepResult(
        observation=next_obs,
        reward=reward,
        done=True, 
        info={
            "message": f"Agent scored: {reward:.2f}/1.0",
            "gt_label": expected_truth.get("label")
        }
    )
def main():
    import uvicorn
    # This matches the HF Spaces default port
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

# The validator is explicitly looking for this standard Python block!
if __name__ == "__main__":
    main()