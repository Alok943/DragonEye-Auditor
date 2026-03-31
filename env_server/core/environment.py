import json
import random
from pathlib import Path
from .models import Action, Observation, StepResult
from .rewards import calculate_reward

# Dynamically locate the data file
DATA_PATH = Path(__file__).parent.parent / "data" / "reviews_v1.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

class AuditorEnv:
    def __init__(self):
        self.dataset = DATASET
        self.current_item = None
        self.session_count = 0

    def reset(self) -> Observation:
        """Pulls a random review to start a new audit task."""
        self.current_item = random.choice(self.dataset)
        self.session_count += 1
        
        return Observation(
            review_text=self.current_item["text"],
            session_id=f"audit_{self.session_count}"
        )

    def step(self, action: Action) -> StepResult:
        """Evaluates the AI's action and returns the reward and next state."""
        if not self.current_item:
            self.reset()

        # Calculate shaped reward using your custom logic
        reward = calculate_reward(
            expected_label=self.current_item["label"],
            expected_lang=self.current_item["lang"],
            action_label=action.label,
            action_lang=action.lang,
            difficulty=self.current_item["difficulty"]
        )

        # Ground truth data (Agent only sees this AFTER making a decision)
        info = {
            "actual_label": self.current_item["label"],
            "actual_lang": self.current_item["lang"],
            "difficulty": self.current_item["difficulty"],
            "ai_reasoning": action.reasoning
        }

        # In this simple audit task, 1 decision = 1 completed episode
        done = True

        # Auto-reset to provide the next review seamlessly
        next_obs = self.reset()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info
        )