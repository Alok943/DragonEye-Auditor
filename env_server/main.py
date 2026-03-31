from fastapi import FastAPI, HTTPException
from env_server.core.models import Action, Observation, StepResult
from env_server.core.environment import AuditorEnv
from typing import Optional, Dict, Any
# Initialize the FastAPI application
app = FastAPI(
    title="OpenEnv: Hinglish Review Auditor",
    description="A reinforcement learning environment for auditing Indian e-commerce reviews.",
    version="1.0.0"
)

# Instantiate our environment instance
# In a production app, we'd use database sessions, but for a hackathon, memory is perfect.
env = AuditorEnv()

@app.get("/")
async def health_check():
    """Simple health check endpoint for Ngrok/Docker pinging."""
    return {"status": "online", "environment": "Hinglish Auditor v1"}

@app.post("/reset", response_model=Observation)
async def reset_environment():
    """
    Resets the environment and provides the first review to the agent.
    """
    try:
        return env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=Observation)
async def get_current_state():
    """
    Returns the current observation without advancing the environment.
    Required by OpenEnv Spec.
    """
    try:
        # If the environment hasn't been reset yet, start it.
        if not env.current_item:
            return env.reset()
            
        return Observation(
            review_text=env.current_item["text"],
            session_id=f"audit_{env.session_count}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResult)
async def step_environment(action: Action):
    """
    Takes an action from the AI agent, calculates the reward, and advances the state.
    """
    try:
        return env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ==========================================
# OpenEnv Administrative Endpoints
# ==========================================

@app.get("/health")
async def health_endpoint():
    """Required by HF Spaces to ensure the container is alive."""
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata_endpoint():
    """Provides the validator with your environment's identity."""
    return {
        "name": "hinglish-review-auditor",
        "description": "A real-world environment for auditing Indian e-commerce reviews in mixed-language (Hinglish)."
    }

@app.get("/schema")
async def schema_endpoint():
    """Dynamically exports your Pydantic v2 schemas for the AI agent."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": Observation.model_json_schema() # State and Observation are the same here
    }

@app.post("/mcp")
async def mcp_endpoint(payload: Optional[Dict[str, Any]] = None):
    """Model Context Protocol (MCP) required endpoint."""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id", 1) if payload else 1,
        "result": {"status": "mcp_ready"}
    }