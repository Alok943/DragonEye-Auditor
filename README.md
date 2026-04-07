title: DragonEye-Auditor
emoji: 🐉
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
# 🐉 DragonEye-Auditor: Hinglish Review Intelligence Benchmark

DragonEye-Auditor is a standardized reinforcement learning environment built on the OpenEnv framework. It evaluates how well Large Language Models (LLMs) can understand and moderate Indian e-commerce reviews, particularly focusing on Hinglish (Hindi + English), sarcasm, and real-world user frustration.

---

## 🎯 Motivation

Modern Indian e-commerce platforms operate in a linguistically complex environment where users frequently mix Hindi and English, use regional slang, and express dissatisfaction indirectly through sarcasm.

Traditional moderation systems struggle to distinguish between:

* **True Toxicity** → Direct abuse, threats, harassment
* **Sarcastic Frustration** → Negative experience expressed via positive tone
* **Code-Switched Spam** → Promotional content hidden in Hinglish

DragonEye-Auditor acts as a **benchmark environment** to evaluate whether LLMs can handle these real-world nuances effectively.

---

## 🏗️ Architecture

This project follows the OpenEnv standard by separating:

### 🧱 Environment (Server)

* Built using FastAPI and deployed as a containerized service
* Holds dataset, ground truth, and reward logic
* Exposes:

  * `/reset`
  * `/step`

### 🤖 Agent (Inference Script)

* LLM-driven auditor using OpenAI-compatible API
* Interacts via REST API only
* Tested using:

  * **Gemini 2.5 Flash**
  * **Llama 3.1 8B (baseline comparison)**

---

## 🔍 Observation Space

Each step returns:

* `review_text` *(string)* → Review to analyze
* `task_id` *(string)* → Current task context
* `session_id` *(string)* → Unique episode identifier

---

## 🎮 Action Space

The agent must return a JSON object:

* `label` → SAFE | SPAM | TOXIC
* `lang` → en | hi | hinglish
* `nuance_detected` → boolean (sarcasm/slang detection)
* `reasoning` → short explanation
* `model_name` → model identifier

---

## 🧩 Tasks

### Task 1: Language Identification (Easy)

Identify whether the review is English, Hindi, or Hinglish.

### Task 2: Basic Moderation (Medium)

Classify reviews as SAFE, SPAM, or TOXIC.

### Task 3: Sarcasm & Slang Detection (Hard)

Handle indirect frustration, sarcasm, and cultural nuances.

---

## ⚖️ Reward Design

We use a weighted multi-objective reward system:

| Component            | Weight | Description                     |
| -------------------- | ------ | ------------------------------- |
| Language Detection   | 0.2    | Correct language identification |
| Nuance Detection     | 0.3    | Sarcasm/slang understanding     |
| Label Classification | 0.5    | Final moderation decision       |

This ensures the agent is rewarded not just for correctness, but for **understanding intent**.

---

## 🧠 Key Insight

We observed that most models fail not in classification, but in **intent understanding**.

By separating:

* **Nuance (intent)**
* **Label (decision)**

DragonEye-Auditor provides a more realistic evaluation of moderation systems in multilingual settings.

---

## 📊 Baseline Results

| Model            | Avg Reward | Key Observations                          |
| ---------------- | ---------- | ----------------------------------------- |
| Gemini 2.5 Flash | 0.84       | Strong Hinglish and sarcasm understanding |
| Llama 3.1 8B     | 0.62       | Struggles with nuance and sarcasm         |

---

## ⚙️ Setup Instructions

### 1. Run Environment

```bash
docker build -t hinglish-auditor-env .
docker run -p 7860:7860 hinglish-auditor-env
```

---

### 2. Configure `.env`

```env
HF_TOKEN=your_api_key
MODEL_NAME=gemini-2.5-flash
ENV_URL=http://localhost:7860
BENCHMARK=DragonEye-Auditor
```

---

### 3. Run Inference

```bash
python inference.py
```

---

## 🌐 Deployment

The environment is deployed as a containerized Hugging Face Space with OpenEnv compatibility.

For development and testing, we also used:

* **ngrok tunneling**
* Remote GPU setup (RTX 4080 system)

This allowed efficient experimentation without local compute constraints.

---

## 🛡️ Observability

All agent decisions are logged with metadata:

* Timestamp
* Model name
* Predicted vs expected labels
* Reward values
* Reasoning

This enables:

* debugging
* bias detection
* performance tracking

---

## 🧪 OpenEnv Compliance

✔ Dockerized environment
✔ `/reset` and `/step` endpoints
✔ Standardized observation/action space
✔ OpenAI-compatible inference
✔ Multi-task evaluation

---

## 👨‍💻 Author

**DragonEye Team**
ALOK SINGH [LEADER]
Focus: RL Infrastructure × Hinglish NLP

---

## 🚀 Summary

DragonEye-Auditor is not just a benchmark — it is a **real-world stress test** for LLMs operating in multilingual, culturally nuanced environments.

It highlights a critical gap in current AI systems:

> Understanding language is not enough — models must understand intent.
