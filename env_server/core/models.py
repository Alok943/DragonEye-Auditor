from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# ==========================================
# 1. The Agent's Output (What the AI sends)
# ==========================================
class Action(BaseModel):
    # Literal enforces strict type-checking. 
    # If the AI predicts "SPAMMY" instead of "SPAM", Pydantic rejects it.
    label: Literal["SAFE", "SPAM", "TOXIC"] = Field(
        ..., description="The final audit decision for the review."
    )
    lang: Literal["en", "hi", "hinglish"] = Field(
        ..., description="The detected primary language of the review."
    )
    reasoning: str = Field(
        ..., description="A brief explanation of why the AI chose this label."
    )

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