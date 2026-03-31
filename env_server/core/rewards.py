def calculate_reward(
    expected_label: str, 
    expected_lang: str, 
    action_label: str, 
    action_lang: str, 
    difficulty: str
) -> float:
    """Calculates the shaped reward strictly bounded between 0.0 and 1.0."""
    reward = 0.0
    
    # 1. Partial Credit: Language Identification (20%)
    if action_lang == expected_lang:
        reward += 0.2
        
    # 2. Core Goal: Audit Label Accuracy (80%)
    if action_label == expected_label:
        reward += 0.8
    else:
        # If they fail the main task, they get 0 for this section.
        # But if it was a "Hard" sarcastic task, we give a tiny 0.1 pity bonus 
        # so the agent knows it was dealing with a tough edge case.
        if difficulty == "hard":
            reward += 0.1

    # STRICT HACKATHON RULE: Clamp final score between 0.0 and 1.0
    final_score = max(0.0, min(1.0, reward))
    
    return round(final_score, 2)