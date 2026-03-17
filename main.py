"""
SmartSupport AI — FastAPI backend

Four core endpoints:
  /assign       — majority vote over retrieved similar tickets
  /resolve      — returns resolution from the closest matching ticket
  /assign-rag   — uses an LLM to reason over retrieved tickets and pick a team
  /resolve-rag  — uses an LLM to generate a tailored resolution
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from collections import Counter
import subprocess
import json
import logging
import traceback
import requests
import os

# Prevent sentence-transformers from trying to download models at runtime
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from backend.embedder import embed_text
from backend.endee_client import search, check_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartSupport AI",
    description="AI-powered ticket assignment & auto-resolution using Endee (RAG)",
    version="1.0.0"
)

# Serve the frontend UI from the /static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")


@app.on_event("startup")
async def preload_model():
    """
    Load the embedding model at startup so the first request isn't slow.
    Without this, the first call to embed_text() would trigger a model load.
    """
    from backend.embedder import get_model
    get_model()
    logger.info("Embedding model ready.")


# --- Request / Response Models ---

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)  # How many similar tickets to retrieve

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Ticket text cannot be empty or whitespace')
        return v.strip()


class AssignResponse(BaseModel):
    predicted_team: str
    confidence: float       # Fraction of retrieved tickets that agreed on this team (0.0–1.0)
    similar_tickets: int
    status: str = "success"


class ResolveResponse(BaseModel):
    suggested_resolution: str
    status: str = "success"


class RAGAssignResponse(BaseModel):
    team: str
    reason: str   # LLM's explanation for the team assignment
    status: str = "success"


class RAGResolveResponse(BaseModel):
    resolution: str
    status: str = "success"


class HealthResponse(BaseModel):
    api: str
    endee: bool
    ollama: bool
    model: bool


# --- Helper Functions ---

def call_ollama(prompt: str, timeout: int = 60) -> str:
    """
    Send a prompt to the local Ollama server and return the text response.
    Ollama must be running and have llama3 pulled for RAG endpoints to work.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not reachable"
    except Exception as e:
        return f"Error: {str(e)}"


def majority_vote(items):
    """Return the most common item and how many times it appeared."""
    if not items:
        return None, 0
    return Counter(items).most_common(1)[0]


def extract_metadata(item):
    """
    Pull the metadata dict out of a search result.
    Handles multiple possible field names for robustness.
    """
    if isinstance(item, dict):
        if "metadata" in item and isinstance(item["metadata"], dict):
            return item["metadata"]
        if "payload" in item and isinstance(item["payload"], dict):
            return item["payload"]
        if "meta" in item and isinstance(item["meta"], dict):
            return item["meta"]
        return item
    return {}


def build_context(matches, max_items=5):
    """
    Format the top search results into a numbered text block.
    This becomes the 'context' section injected into the LLM prompt.
    Long resolutions are truncated to keep the prompt a manageable size.
    """
    context = ""
    for i, m in enumerate(matches[:max_items], 1):
        meta = extract_metadata(m)
        resolution = meta.get('resolution', 'No resolution available')
        if len(resolution) > 300:
            resolution = resolution[:297] + "..."
        context += f"{i}. Team: {meta.get('team', 'Unknown')}\n   Resolution: {resolution}\n\n"
    return context


def extract_json_safely(raw: str, default: dict) -> dict:
    """
    Parse JSON from an LLM response, stripping markdown code fences if present.
    Falls back to `default` if parsing fails or expected keys are missing.
    """
    try:
        raw = raw.replace("```json", "").replace("```", "")
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            return default
        parsed = json.loads(raw[start:end])
        # Only accept the response if it contains all the keys we expect
        return parsed if all(k in parsed for k in default) else default
    except Exception:
        return default


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check all system components and return their status. Returns 503 if core services are down."""
    health = {"api": "healthy", "endee": False, "ollama": False, "model": False}

    try:
        health["endee"] = check_connection()
    except Exception:
        pass

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=2)
        health["ollama"] = result.returncode == 0
    except Exception:
        pass

    try:
        from backend.embedder import get_model
        get_model()
        health["model"] = True
    except Exception:
        pass

    # Return 503 if Endee or the embedding model is unavailable
    status = 200 if health["endee"] and health["model"] else 503
    return JSONResponse(content=health, status_code=status)


@app.post("/assign", response_model=AssignResponse)
def assign_ticket(req: TicketRequest):
    """
    Assign a ticket to a team using vector similarity + majority voting.

    Steps:
      1. Embed the ticket text into a 384-dim vector
      2. Retrieve the top-k most similar historical tickets from Endee
      3. Collect the team label from each result
      4. Return the most common team and a confidence score
    """
    try:
        # Step 1: Embed
        vector = embed_text(req.text)

        # Step 2: Search Endee
        result = search(vector, top_k=req.top_k)
        matches = result.get("results", [])

        if not matches:
            return AssignResponse(predicted_team="Unknown", confidence=0.0, similar_tickets=0, status="no_matches")

        # Step 3: Extract team labels from each result
        teams = [extract_metadata(m).get("team") for m in matches if extract_metadata(m).get("team")]

        if not teams:
            return AssignResponse(predicted_team="Unknown", confidence=0.0, similar_tickets=0, status="no_team_labels")

        # Step 4: Majority vote — confidence = fraction of results that agreed
        team, count = majority_vote(teams)
        confidence = count / len(teams)

        # Flag low-confidence predictions so agents know to double-check
        if confidence < 0.5:
            team, status = "Manual Review", "low_confidence"
        else:
            status = "success"

        return AssignResponse(
            predicted_team=team,
            confidence=round(confidence, 2),
            similar_tickets=len(teams),
            status=status
        )

    except HTTPException:
        raise
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal error during ticket assignment")


@app.post("/resolve", response_model=ResolveResponse)
def resolve_ticket(req: TicketRequest):
    """
    Suggest a resolution by returning the resolution from the closest matching historical ticket.
    No LLM involved — fast and deterministic.
    """
    try:
        vector = embed_text(req.text)
        result = search(vector, top_k=req.top_k)
        matches = result.get("results", [])

        if not matches:
            return ResolveResponse(suggested_resolution="No similar tickets found", status="no_matches")

        # Use the top result's resolution as the suggestion
        resolution = extract_metadata(matches[0]).get("resolution", "Resolution not available")
        return ResolveResponse(suggested_resolution=resolution, status="success")

    except HTTPException:
        raise
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal error during resolution")


@app.post("/assign-rag", response_model=RAGAssignResponse)
def assign_ticket_rag(req: TicketRequest):
    """
    RAG-based assignment: retrieve similar tickets → inject as context → ask LLM which team.
    More accurate than majority voting for ambiguous tickets but requires Ollama.
    """
    try:
        vector = embed_text(f"Support ticket: {req.text}")
        matches = search(vector, top_k=req.top_k).get("results", [])

        if not matches:
            return RAGAssignResponse(team="Unknown", reason="No similar historical tickets found", status="no_matches")

        # Build the numbered context block from retrieved tickets
        context = build_context(matches)

        prompt = f"""You are an AI system that assigns customer support tickets to teams.

Here are similar past tickets and how they were handled:
{context}
New ticket:
"{req.text}"

Based on the similar tickets above, decide which team should handle this new ticket.
Respond ONLY with valid JSON: {{"team": "...", "reason": "..."}}

Your JSON response:"""

        raw = call_ollama(prompt, timeout=60)

        if raw.startswith("Error:"):
            return RAGAssignResponse(team="Unknown", reason=raw, status="llm_error")

        parsed = extract_json_safely(raw, {"team": "Unknown", "reason": "Failed to parse LLM response"})
        return RAGAssignResponse(team=parsed["team"], reason=parsed["reason"], status="success")

    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal error during RAG assignment")


@app.post("/resolve-rag", response_model=RAGResolveResponse)
def resolve_ticket_rag(req: TicketRequest):
    """
    RAG-based resolution: retrieve similar tickets → inject as context → ask LLM to write a resolution.
    Generates a tailored response rather than copying a past resolution verbatim.
    """
    try:
        vector = embed_text(req.text)
        matches = search(vector, top_k=req.top_k).get("results", [])

        if not matches:
            return RAGResolveResponse(
                resolution="No similar tickets found. Please contact support.",
                status="no_matches"
            )

        prompt = f"""You are a professional customer support assistant.

Here are past resolutions for similar tickets:
{build_context(matches)}
New ticket:
"{req.text}"

Write a clear, helpful, 2-4 sentence resolution for this ticket.

Resolution:"""

        resolution = call_ollama(prompt, timeout=30)

        if resolution.startswith("Error:"):
            return RAGResolveResponse(
                resolution="Unable to generate resolution. Please contact support.",
                status="llm_error"
            )

        return RAGResolveResponse(resolution=resolution, status="success")

    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal error during RAG resolution")


@app.get("/")
def root():
    return {
        "name": "SmartSupport AI",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health", "docs": "/docs",
            "assign": "/assign", "resolve": "/resolve",
            "assign_rag": "/assign-rag", "resolve_rag": "/resolve-rag"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)