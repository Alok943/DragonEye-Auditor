def calculate_reward(
    task_id: str,
    expected_label: str,
    expected_lang: str,
    expected_nuance: bool,
    action_label: str,
    action_lang: str,
    action_nuance: bool,
    reasoning: str = ""
) -> float:
    """
    Final reward function for DragonEye-Auditor OpenEnv.
    Deterministic, bounded [0,1], task-specific, and robust.
    """

    # 🚫 Hard fail: invalid output
    if action_label == "INVALID":
        return 0.0

    reward = 0.0

    # =========================
    # TASK 1: Language ID
    # =========================
    if task_id == "task_1_language_id":
        if action_lang == expected_lang:
            reward = 1.0

    # =========================
    # TASK 2: Moderation
    # =========================
    elif task_id == "task_2_basic_moderation":
        if action_label == expected_label:
            reward += 0.8
        if action_lang == expected_lang:
            reward += 0.2

    # =========================
    # TASK 3: Sarcasm + Nuance
    # =========================
    elif task_id == "task_3_sarcasm_slang":

        # 1. Language (0.2)
        if action_lang == expected_lang:
            reward += 0.2

        # 2. Nuance (0.3 with balanced penalties)
        if action_nuance == expected_nuance:
            reward += 0.3
        #elif expected_nuance and not action_nuance:
        #    reward -= 0.1  # missed sarcasm
        #elif not expected_nuance and action_nuance:
        #    reward -= 0.05  # hallucinated nuance (lighter penalty)

        # 3. Label (0.5 with penalty)
        if action_label == expected_label:
            reward += 0.5
        #else:
         #   reward -= 0.2  # wrong classification penalty

    # Clamp reward safely
    return round(max(0.0, min(1.0, reward)), 2)