🐉 DragonEye-Auditor: The Hinglish OpenEnv Benchmark

DragonEye-Auditor is a standardized reinforcement learning environment built on the OpenEnv framework. It is designed to evaluate and audit the ability of Large Language Models (LLMs) to moderate Indian e-commerce reviews—specifically focusing on the linguistic "blind spots" of Hinglish (Hindi + English) code-switching, regional slang, and sarcastic frustration.
🎯 The Problem: The "Hinglish" Gap

In 2026, Indian e-commerce is dominated by regional linguistic nuances. Traditional moderation systems often fail to distinguish between:

    True Toxicity: Abusive language and threats.

    Sarcastic Frustration: High-value feedback disguised as praise (e.g., "Truly a world-class paperweight!").

    Code-Switched Spam: Promotional links hidden in Hinglish praise.

DragonEye-Auditor provides the "Driver's License Test" for any LLM intended for the Indian market.
🏗️ Architecture: The OpenEnv Standard

We have strictly decoupled the Environment (The Referee) from the Agent (The Player) using a containerized microservice architecture.

    The Environment (Docker/FastAPI): An immutable server that holds the ground-truth dataset and grading logic. It exposes two core endpoints: /reset and /step.

    The Agent (Python SDK): An LLM-driven auditor (Gemini 2.5 Flash / Llama 8B) that interacts with the environment purely via REST APIs. No "cheating" or direct data access is possible.

⚖️ Multi-Objective Reward Shaping

To provide a granular learning signal, we implemented a Weighted Reward System (0.2 / 0.3 / 0.5):
Component	Weight	Logic
Linguistic ID	0.2	Rewards correct identification of en, hi, or hinglish.
Contextual Nuance	0.3	Rewards detection of sarcasm, irony, or cultural slang.
Audit Label	0.5	The final categorical decision (SAFE, SPAM, TOXIC).

    Hard-Case Bonus: We provide a +0.1 pity bonus for agents that fail the label but correctly identify "Hard" difficulty sarcasm—this encourages the agent to prioritize reasoning over lucky guesses.

📊 Benchmark Results (50-Episode Stress Test)

We ran a deterministic sequential audit starting from Index 30 (the "Hard" sarcasm block) to compare a frontier model against a local edge model.
Model	Avg. Reward	Key Finding
Gemini 2.5 Flash	0.84	Mastered Hinglish slang; occasionally missed subtle coupon spam.
Llama 3.1 8B	0.62	Struggles with sarcasm; frequently flags regional praise as TOXIC.
🚀 Getting Started
1. Launch the Environment (Docker)
Bash

docker build -t hinglish-auditor-env .
docker run -p 7860:7860 hinglish-auditor-env

2. Run the Auditor Agent

Ensure your .env file contains your HF_TOKEN (Gemini Key) and MODEL_NAME.
Bash

python inference.py

🛡️ Audit Observability

Every decision made by the agent is logged to audit_results.jsonl with full metadata, including:

    Timestamp (ISO 8601)

    Model ID

    Agent Reasoning (The "Why" behind the score)

    Reward Breakdown

This transparency is critical for MLOps monitoring and bias detection in automated moderation systems.
👨‍💻 The Team: DragonEye

    Core Logic: [Your Name/Handle]

    Theme: RL Infrastructure & Hinglish NLP

    Submission Round: Round 1 - OpenEnv Mini-RL Environment