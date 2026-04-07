from .rewards import calculate_reward


def get_auditor_grade(review_text: str, agent_decision: dict, expected: dict) -> float:
    """Calculates the reward using the expected ground truth."""
    return calculate_reward(
        expected_label=expected.get("label", "SAFE"),
        expected_lang=expected.get("lang", "hinglish"),
        expected_nuance=expected.get("nuance", False),
        action_label=agent_decision.get("label"),
        action_lang=agent_decision.get("lang"),
    )