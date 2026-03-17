# 🎫 SmartTicket AI

> AI-powered support ticket routing and auto-resolution — built with FastAPI, Endee vector database, and Llama 3

SmartTicket AI automatically assigns incoming support tickets to the correct team and generates resolution suggestions by retrieving similar historical tickets using vector search, then reasoning over them with a local LLM. No data leaves your machine.

---

## ✨ Features

- **⚡ Smart Team Assignment** — vector similarity search over 16,340 historical tickets with majority-vote prediction
- **💡 RAG Resolution Generation** — Llama 3 generates tailored responses using retrieved ticket context
- **🎯 Confidence Scoring** — percentage score showing how strongly similar tickets agree on a team
- **🔄 Graceful Fallback** — if the LLM is slow or unavailable, automatically falls back to vector-only matching
- **🖥️ Jira-style UI** — clean, office-friendly interface served at `/ui`
- **🔒 Fully local** — Endee + Ollama run entirely on-device; no API keys, no cloud calls

---

## 🏗️ Architecture

```
Browser (static/index.html)
         │
         ▼
  FastAPI :8000  (backend/main.py)
         │
    ┌────┴─────────────────┐
    ▼                       ▼
Endee Vector DB :8080    Ollama :11434
HNSW · cosine · int8d    llama3 (local LLM)
    ▲
    │
SentenceTransformers
all-MiniLM-L6-v2 (384d)
```

### Data Flow

```
Ingest:  cleaned_tickets.csv → embed_batch() → insert_batch() → Endee index
Query:   user ticket → embed_text() → search() → top-K results
Assign:  top-K teams → majority_vote() → predicted team + confidence
RAG:     top-K results → build_context() → LLM prompt → team / resolution
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Vector DB | [Endee](https://github.com/billionai/endee) — HNSW, cosine, int8 quantization |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| LLM | [Ollama](https://ollama.com) — llama3 (runs locally) |
| Frontend | Vanilla HTML / CSS / JS — no framework |

---

## 📦 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed
- Endee binary (`ndd-avx2`) built and available
- `cleaned_tickets.csv` placed in the `data/` directory

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-support-ai.git
cd smart-support-ai
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the LLM model

```bash
ollama pull llama3
```

### 4. Start Endee

```bash
cd ~/endee
export NDD_DATA_DIR=$(pwd)/data
./build/ndd-avx2
```

### 5. Ingest ticket data *(run once)*

```bash
cd ~/smart-support-ai
source venv/bin/activate
python ingest_tickets.py
```

Embeds all 16k tickets from `data/cleaned_tickets.csv` and loads them into Endee.

### 6. Start the API server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Open the UI

```
http://127.0.0.1:8000/ui
```

---

## ✅ Verify Setup

Run the diagnostics script to confirm all components are working:

```bash
python test_setup.py
```

Expected output:
```
✅  CSV file          (11.9 MB)
✅  Endee server      Running at localhost:8080
✅  Embedding model   all-MiniLM-L6-v2 (dim=384)
✅  Index exists      total_elements=16,340
✅  Index search      Sample result — team: Billing and Payments
✅  Ollama server     Running at localhost:11434
✅  llama3 model      Available
✅  API server        Running at localhost:8000
✅  index.html        Found at static/index.html
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status of all system components |
| `POST` | `/assign` | Team assignment via vector similarity + majority vote |
| `POST` | `/resolve` | Resolution from the closest matching historical ticket |
| `POST` | `/assign-rag` | Team assignment using RAG + Llama 3 |
| `POST` | `/resolve-rag` | Resolution generation using RAG + Llama 3 |
| `GET` | `/ui` | Frontend interface |
| `GET` | `/docs` | Interactive Swagger API docs |

### Request body (all POST endpoints)

```json
{
  "text": "Ticket description (10–5000 characters)",
  "top_k": 5
}
```

### Example — Assign ticket (RAG)

```bash
curl -X POST http://localhost:8000/assign-rag \
  -H "Content-Type: application/json" \
  -d '{"text": "My laptop cannot connect to the VPN after the Windows update", "top_k": 5}'
```

```json
{
  "team": "Technical Support",
  "reason": "Similar VPN connectivity issues were consistently routed to Technical Support.",
  "status": "success"
}
```

### Example — Get resolution (RAG)

```bash
curl -X POST http://localhost:8000/resolve-rag \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription this month", "top_k": 5}'
```

```json
{
  "resolution": "We apologise for the duplicate charge. Our billing team has been notified and will process a refund within 3–5 business days. Please check your email for a confirmation.",
  "status": "success"
}
```

---

## 🧪 Usage Examples

Try these in the UI at `http://127.0.0.1:8000/ui` to see SmartTicket AI in action.

---

### Example 1 — Technical Support (VPN Issue)

| Field | Value |
|---|---|
| **Summary** | Laptop won't connect to VPN |
| **Description** | My laptop stopped connecting to the company VPN after the latest Windows update. I get error code 800. Other colleagues on the same network are fine. |
| **Issue Type** | 🐛 Bug / Error |
| **Affected System** | VPN |
| **Priority** | 🟠 High |

**Expected result:**
```
🔧 Team:   Technical Support
💬 Reason: Similar VPN and network connectivity issues were handled by Technical Support.
```

---

### Example 2 — Billing & Payments (Duplicate Charge)

| Field | Value |
|---|---|
| **Summary** | Charged twice for monthly subscription |
| **Description** | I was charged twice for my monthly subscription in January. My bank shows two transactions of $49.99 on the same day. |
| **Issue Type** | 💳 Billing |
| **Affected System** | Payment Portal |
| **Priority** | 🟠 High |

**Expected result:**
```
💳 Team:   Billing and Payments
💬 Reason: Duplicate charge issues have consistently been resolved by the Billing and Payments team.
```

---

## 📁 Project Structure

```
smart-support-ai/
├── backend/
│   ├── embedder.py          # SentenceTransformers wrapper with LRU model cache
│   ├── endee_client.py      # Direct HTTP client for Endee (bypasses pydantic v1 SDK)
│   └── main.py              # FastAPI app — endpoints, RAG logic, static file serving
├── static/
│   └── index.html           # Jira-style frontend UI
├── data/
│   └── cleaned_tickets.csv  # 16,340 historical support tickets (gitignored)
├── ingest_tickets.py        # One-time data ingestion pipeline
├── test_setup.py            # Full system diagnostics (9 checks)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

**`backend/endee_client.py`**
```python
INDEX_NAME = "tickets"    # Endee index name
DIMENSION  = 384          # Must match embedding model output
BASE_URL   = "http://127.0.0.1:8080/api/v1"
```

**`backend/embedder.py`**
```python
MODEL_NAME    = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```

**Environment variables** (prevent HuggingFace network calls on every startup):
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| API times out on startup | `export TRANSFORMERS_OFFLINE=1` before starting uvicorn |
| `assign-rag` / `resolve-rag` times out | Llama 3 is slow on first inference — retry once it's warm |
| Index is empty | Run `python ingest_tickets.py` |
| Endee not running | `cd ~/endee && export NDD_DATA_DIR=$(pwd)/data && ./build/ndd-avx2` |
| Ollama not reachable | `ollama serve` then `ollama pull llama3` |

---

## 📄 License

MIT