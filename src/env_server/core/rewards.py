def calculate_reward(
    expected_label: str,
    expected_lang: str,
    expected_nuance: bool,
    action_label: str,
    action_lang: str,
) -> float:
    """
    Weighted Reward System (total = 1.0):
      0.2 — Linguistic ID:     correct language detection (en / hi / hinglish)
      0.3 — Contextual Nuance: correct handling of sarcasm / slang / irony
      0.5 — Audit Label:       correct final classification (SAFE / SPAM / TOXIC)
    """
    reward = 0.0

    # 1. Linguistic Identification (0.2)
    if action_lang == expected_lang:
        reward += 0.2

    # 2. Contextual Nuance (0.3)
    # Ground truth says this review has nuance (sarcasm/slang/irony).
    # The agent gets credit if it still got the label right — meaning it
    # correctly read through the subtext instead of being misled by surface tone.
    if expected_nuance:
        if action_label == expected_label:
            reward += 0.3
        # Wrong label on a nuanced review = 0 for this component.
        # No pity bonus — the signal should be clear.
    else:
        # Non-nuanced review: nuance component is awarded freely
        # since there was no subtext to navigate.
        reward += 0.3

    # 3. Audit Label (0.5)
    if action_label == expected_label:
        reward += 0.5

    # Clamp strictly to [0.0, 1.0]
    return round(max(0.0, min(1.0, reward)), 2)