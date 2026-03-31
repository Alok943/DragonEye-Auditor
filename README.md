Part 1: The Teammate's Playbook (RTX 4080 / i9)

Since your teammate has the hardware to run the entire OpenEnv architecture locally, their setup is going to be incredibly fast with zero network latency. Send them these exact steps:

Step 1: Get the Code & Environment Ready
Bash

# 1. Pull the latest code from GitHub
git pull origin main

# 2. Create and activate a fresh virtual environment
python -m venv venv
venv\Scripts\activate  # (Windows) or `source venv/bin/activate` (Mac/Linux)

# 3. Install the specific Hackathon dependencies
pip install -r requirements.txt

Step 2: Start the Local "Brain" (Ollama)
They need to make sure Ollama is running in the background and that they have the required model downloaded.
Bash

ollama pull qwen3:7b-instruct

Step 3: Run the Simulation (Two Terminals)
They need to open two separate terminals in VS Code (both with the venv activated).

    Terminal 1 (Start the Environment):
    Bash

    uvicorn env_server.main:app --host 0.0.0.0 --port 8000

    Terminal 2 (Run the Inference Baseline):
    Bash

    python inference.py

They will instantly see the AI evaluating the reviews and calculating the final baseline score.