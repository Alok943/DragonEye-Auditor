---
title: DragonEye-Auditor
emoji: 🐉
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

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

## 📚 Data Source

The dataset used in DragonEye-Auditor is **synthetically generated** to simulate realistic Indian e-commerce reviews.

Data was created using a combination of:

- Sarvam AI
- ChatGPT
- Google AI Studio (Gemini)

The generation process focused on replicating real-world patterns such as:

- Hinglish (Hindi + English code-switching)
- Sarcasm and indirect frustration
- Informal slang and culturally contextual expressions

All samples were:

- Carefully **curated and filtered**
- **Manually structured and annotated** for:
  - Language identification
  - Moderation (SAFE / SPAM / TOXIC)
  - Nuance detection (sarcasm/slang)

No real user data or personally identifiable information was used.

This synthetic approach allows controlled generation of **edge cases and nuanced scenarios** that are often underrepresented in real datasets.

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

  * **Gemini 2.5 Flash** (high accuracy, but unstable due to API latency and 503 errors)
  * **Gemini 2.5 Flash Lite (final model)** (slightly lower raw capability but significantly more stable)
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

## ⚠️ Practical Challenges

During evaluation, we observed that model performance is not only dependent on reasoning capability but also on **API reliability and latency**.

- Gemini 2.5 Flash showed strong reasoning ability but suffered from:
  - Frequent **503 errors**
  - High latency under load
  - Increased fallback responses

- This led to unstable scores despite good predictions.

To address this, we switched to **Gemini 2.5 Flash Lite**, which provided:

- Lower latency
- Higher availability
- More consistent end-to-end performance

This highlights an important real-world insight:

> In production-like environments, **stability often outweighs theoretical model accuracy**.

## 📊 Baseline Results

| Model                    | Avg Reward | Key Observations                                      |
| ------------------------ | ---------- | ----------------------------------------------------- |
| Gemini 2.5 Flash         | ~0.70–0.80 | Strong reasoning but affected by API timeouts/failures |
| Gemini 2.5 Flash Lite    | 0.85       | Stable, fast, and best overall performance            |
| Llama 3.1 8B             | 0.62       | Struggles with nuance and sarcasm                     |

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
MODEL_NAME=gemini-2.5-flash-lite
ENV_URL=https://alok8732-dragoneye-auditor.hf.space
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
