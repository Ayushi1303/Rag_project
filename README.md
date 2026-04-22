# RAG Customer Support Assistant
### Built with LangChain · LangGraph · ChromaDB · GPT-4o

A production-quality Retrieval-Augmented Generation (RAG) system that
answers customer questions from a PDF knowledge base, uses a graph-based
workflow for control logic, and escalates low-confidence queries to a
human agent via HITL (Human-in-the-Loop).

---

## Project Structure

```
rag_project/
├── config.py           # All tuneable parameters in one place
├── ingest.py           # PDF → chunks → embeddings → ChromaDB
├── retriever.py        # Load ChromaDB + semantic search
├── prompt_builder.py   # LLM prompt templates
├── state.py            # LangGraph GraphState TypedDict
├── nodes.py            # intake_node, rag_node, router_node, hitl_node
├── graph.py            # LangGraph workflow builder
├── hitl_handler.py     # CLI interrupt/resume handler
├── display.py          # Pretty CLI output formatting
├── main.py             # Entry point — interactive Q&A loop
│
├── data/               # ← Put your PDF files here
├── chroma_db/          # ← Auto-created by ingest.py (do not edit)
├── logs/
│   └── escalation_log.jsonl  # Auto-created on first escalation
│
├── tests/
│   ├── test_routing.py # Unit tests for routing logic (offline)
│   └── test_ingest.py  # Unit tests for ingestion pipeline (offline)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (Step by Step)

### Step 1 — Clone / download the project

```bash
cd rag_project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your OpenAI API key

```bash
cp .env.example .env
```
Open `.env` and replace `sk-your-openai-api-key-here` with your real key.

> Get your API key at: https://platform.openai.com/api-keys

### Step 5 — Add your PDF document(s)

Place one or more PDF files in the `data/` folder.

```
data/
  product_manual.pdf
  faq.pdf
  returns_policy.pdf
```

### Step 6 — Run ingestion (one-time setup)

```bash
python ingest.py
```

This reads every PDF in `data/`, splits them into chunks, computes
embeddings via OpenAI, and stores them in `chroma_db/`.

You only need to re-run this when you add or update PDF files.

### Step 7 — Start the assistant

```bash
python main.py
```

---

## Example Session

```
============================================================
  RAG Customer Support Assistant
  Powered by GPT-4o + ChromaDB + LangGraph
============================================================
  Type your question and press Enter.
  Type 'quit' or 'exit' to end the session.
============================================================

You: What is the return policy?
  ⏳ Searching knowledge base and generating response...

──────────────────────────────────────────────────────────
  🤖 ASSISTANT:
──────────────────────────────────────────────────────────

  Customers may return unused items within 30 days of
  purchase with the original receipt. Damaged items require
  photo proof submitted within 7 days of delivery.

──────────────────────────────────────────────────────────
  📄 Sources:
     • returns_policy.pdf  —  Page 4
  📊 Confidence: HIGH
──────────────────────────────────────────────────────────

You: I want to sue the company for a faulty product
  ⏳ Searching knowledge base and generating response...

  ⚠  ESCALATED TO HUMAN AGENT
────────────────────────────────────────────────────────
  Query:  I want to sue the company for a faulty product
  Reason: sensitive_topic
────────────────────────────────────────────────────────

════════════════════════════════════════════════════════
  HUMAN AGENT INPUT REQUIRED
════════════════════════════════════════════════════════
  Customer query : I want to sue the company for a faulty product
  Escalation reason: sensitive_topic
────────────────────────────────────────────────────────
  Please type your response for the customer:
────────────────────────────────────────────────────────
  Your response: We're sorry to hear about your experience.
                 Please contact our legal team at legal@company.com.

──────────────────────────────────────────────────────────
  🧑 HUMAN AGENT RESPONSE:
──────────────────────────────────────────────────────────

  We're sorry to hear about your experience.
  Please contact our legal team at legal@company.com.

──────────────────────────────────────────────────────────
  [Source: Human Agent]
──────────────────────────────────────────────────────────

You: quit

============================================================
  Session ended. Goodbye!
============================================================
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are offline — they do not make API calls and are free to run.

---

## Configuration

All settings are in `config.py`. Key parameters:

| Parameter           | Default                    | Description                            |
|---------------------|----------------------------|----------------------------------------|
| `CHAT_MODEL`        | `gpt-4o`                   | OpenAI model for generation            |
| `EMBEDDING_MODEL`   | `text-embedding-ada-002`   | OpenAI model for embeddings            |
| `CHUNK_SIZE`        | `500`                      | Characters per chunk                   |
| `CHUNK_OVERLAP`     | `100`                      | Overlap between adjacent chunks        |
| `TOP_K`             | `4`                        | Chunks retrieved per query             |
| `MIN_ANSWER_LENGTH` | `30`                       | Answers shorter than this → escalate   |
| `TEMPERATURE`       | `0.0`                      | LLM determinism (0 = fully deterministic) |

---

## How Routing Works

Every query goes through this decision tree in `router_node`:

```
1. Upstream error?               → ESCALATE
2. No chunks retrieved?          → ESCALATE
3. Answer < 30 characters?       → ESCALATE
4. Uncertainty phrase in answer? → ESCALATE
   ("I don't have enough information", "I'm not sure", etc.)
5. Sensitive keyword in query?   → ESCALATE
   (refund, lawsuit, legal, complaint, compensation, sue, ...)
6. None of the above             → ANSWER  ✓
```

---

## Escalation Log

Every escalation is recorded in `logs/escalation_log.jsonl`:

```json
{
  "timestamp": "2025-07-15T10:32:00Z",
  "query": "I want to sue the company",
  "llm_answer": "I don't have enough information...",
  "reason": "sensitive_topic",
  "human_answer": "Please contact legal@company.com",
  "resolved": true
}
```

---

## Adding More Documents

1. Place new PDF files in `data/`
2. Re-run: `python ingest.py`

The new documents are added to the existing ChromaDB — no need to
delete old embeddings.

---

## Tech Stack

| Component            | Technology                        |
|----------------------|-----------------------------------|
| Document loading     | LangChain PyPDFLoader             |
| Text splitting       | RecursiveCharacterTextSplitter    |
| Embeddings           | OpenAI text-embedding-ada-002     |
| Vector database      | ChromaDB (local, persistent)      |
| LLM                  | OpenAI GPT-4o                     |
| Workflow engine      | LangGraph (StateGraph)            |
| HITL                 | LangGraph interrupt() + Command() |
| Interface            | Python CLI (stdin/stdout)         |

---

## Streamlit Web UI

Start the FastAPI server first, then launch the UI:

```bash
# Terminal 1 — API server
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit UI
streamlit run streamlit_app.py
```

Open: **http://localhost:8501**

### UI Features
| Feature | Description |
|---|---|
| Chat bubbles | Color-coded user / AI / human agent messages |
| Confidence badge | HIGH (green) or LOW (red) per response |
| Source citations | Clickable chips showing PDF filename + page |
| 👍 / 👎 buttons | Sends feedback to LangSmith per answer |
| HITL panel | Human agent text area appears when a query escalates |
| PDF uploader | Drag-and-drop PDFs; click Ingest to update knowledge base |
| System status | Live sidebar showing ChromaDB vector count + LangSmith status |
| Response timing | Optional ms timer per response |

---

## FastAPI REST Server

```bash
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/status` | ChromaDB vector count, LangSmith connection |
| `POST` | `/chat` | Send a question, receive an answer |
| `POST` | `/hitl/respond` | Human agent submits answer for escalated query |
| `POST` | `/feedback` | Submit 👍/👎 for a run (goes to LangSmith) |
| `POST` | `/ingest` | Trigger background re-ingestion of all PDFs |

### Example cURL

```bash
# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy?", "thread_id": "user-001"}'

# Submit thumbs-up feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc-123", "score": 1.0, "comment": "Perfect answer"}'

# Check system status
curl http://localhost:8000/status
```

### HITL Flow via API

```
1. POST /chat               → returns 202 + thread_id if escalated
2. Human reads the query from escalation_log.jsonl or the Streamlit panel
3. POST /hitl/respond       → submits human answer, resumes the graph
4. Customer receives the human answer
```

---

## LangSmith Tracing

### Setup (2 minutes)

1. Create a free account at **https://smith.langchain.com**
2. Go to Settings → API Keys → Create API Key
3. Add to your `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__your-key-here
LANGSMITH_PROJECT=rag-customer-support
```
4. Restart the server — tracing is now active.

### What Gets Traced Automatically

Every query creates a trace tree in LangSmith:

```
rag-customer-support run
  ├── intake_node          (timing, input query, output clean_query)
  ├── rag_node
  │     ├── Retriever      (query embedding, top-k chunks returned)
  │     └── ChatOpenAI     (full prompt, completion, token counts, cost)
  ├── router_node          (route decision: answer / escalate)
  └── hitl_node            (only if escalated)
```

### LangSmith Dashboard Features
- **Trace tree** — see every node, its inputs/outputs, and timing
- **Token usage** — input tokens, output tokens, estimated cost per query
- **Latency** — P50/P95 latency breakdown per node
- **Feedback** — thumbs-up/down ratings submitted via `/feedback`
- **Tags** — filter by `source:api`, `source:streamlit`, `sensitive:true`
- **Dataset** — save query/answer pairs for evaluation

### Local Fallback Traces

Even without LangSmith, every query is logged locally:

```
logs/
  query_traces.jsonl   — one record per query (timing, route, sources)
  local_traces.jsonl   — one record per node execution
  escalation_log.jsonl — one record per HITL escalation
```

---

## Running Everything Together

```bash
# Terminal 1: Ingest your PDFs (one-time)
python ingest.py

# Terminal 2: FastAPI backend
uvicorn api.server:app --reload --port 8000

# Terminal 3: Streamlit frontend
streamlit run streamlit_app.py

# Terminal 4: CLI (optional, works in parallel)
python main.py
```

---

## Updated Project Structure

```
rag_project/
├── config.py              # All config including LangSmith + API settings
├── tracing.py             # LangSmith client, metadata, feedback, local logs
├── state.py               # GraphState TypedDict
├── ingest.py              # PDF → chunks → embeddings → ChromaDB
├── retriever.py           # Load ChromaDB + semantic search
├── prompt_builder.py      # LLM prompt templates
├── nodes.py               # LangGraph nodes (all decorated with @traced_node)
├── graph.py               # LangGraph workflow (LangSmith metadata in config)
├── hitl_handler.py        # CLI HITL resume handler
├── display.py             # CLI output formatting
├── main.py                # CLI entry point
├── streamlit_app.py       # Streamlit web UI
│
├── api/
│   ├── __init__.py
│   ├── server.py          # FastAPI app with all endpoints
│   ├── schemas.py         # Pydantic request/response models
│   └── dependencies.py    # Shared app state + dependency injection
│
├── data/                  # Put PDF files here
├── chroma_db/             # Auto-created by ingest.py
├── logs/
│   ├── escalation_log.jsonl
│   ├── query_traces.jsonl
│   └── local_traces.jsonl
│
├── tests/
│   ├── test_routing.py
│   └── test_ingest.py
│
├── requirements.txt
├── .env.example
└── README.md
```
