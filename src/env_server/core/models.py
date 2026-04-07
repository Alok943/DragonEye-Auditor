from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# ==========================================
# 1. The Agent's Output (What the AI sends)
# ==========================================
class Action(BaseModel):
    label: Literal["SAFE", "SPAM", "TOXIC", "INVALID"] = Field(
        ..., description="The final audit decision for the review."
    )
    lang: Literal["en", "hi", "hinglish", "unknown"] = Field(
        ..., description="The detected primary language of the review."
    )
    nuance_detected: bool = Field(
        ..., 
        description="True if the review contains sarcasm, irony, slang, or indirect tone (e.g., positive words used to express negative experience). False otherwise."
    )
    reasoning: str = Field(
        ..., description="A brief explanation of why the AI chose this label."
    )
    model_name: str = Field(default="unknown", description="The LLM used for inference.")
# ==========================================
# 2. The Environment's State (What the AI sees)
# ==========================================
class Observation(BaseModel):
    review_text: str = Field(
        ..., description="The e-commerce review text to be audited."
    )
    session_id: str = Field(
        ..., description="Unique ID for tracking the current audit session."
    )
    task_id: str = Field(..., description="The OpenEnv task ID for this observation.")

# ==========================================
# 3. The OpenEnv Loop Standard (The Response)
# ==========================================
class StepResult(BaseModel):
    observation: Observation = Field(
        ..., description="The next observation to process."
    )
    reward: float = Field(
        ..., description="The shaped reward earned for the previous action (-1.0 to 1.0)."
    )
    done: bool = Field(
        ..., description="True if the current review is fully audited."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Extra metadata for debugging (e.g., actual ground truth)."
    )
class ResetRequest(BaseModel):
    start_index: int = 0