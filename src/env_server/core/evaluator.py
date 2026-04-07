from .rewards import calculate_reward

def get_auditor_grade(task_id: str, review_text: str, agent_decision: dict, expected: dict) -> float:
    return calculate_reward(
        task_id=task_id,
        expected_label=expected.get("label", "SAFE"),
        expected_lang=expected.get("lang", "hinglish"),
        expected_nuance=expected.get("nuance", False),
        action_label=agent_decision.get("label"),
        action_lang=agent_decision.get("lang"),
        action_nuance=agent_decision.get("nuance_detected", False),
        reasoning=agent_decision.get("reasoning", "")
    )