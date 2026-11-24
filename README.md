# Humanip

A production-ready tool that converts AI-generated, robotic, or unnatural text into fluent, natural human prose — while preserving all facts, citations, numbers, and technical terms.

## Features

- **7 Writing Modes**: Natural, Academic, Professional, Simple English, Research, Conversational, Personal
- **3 Intensity Levels**: Light, Balanced, Strong
- **3 Rewrite Scopes**: Sentence-by-sentence, Paragraph-by-paragraph, Full document
- **Preservation Engine**: Automatically protects citations, numbers, and technical terms from modification
- **Quality Evaluation**: 8-metric scoring system (semantic similarity, factual accuracy, grammar, readability, repetition, citation integrity, sentence variety, paragraph coherence)
- **Word-level diff**: Side-by-side before/after comparison with colour highlights
- **NLP Analysis**: Detects AI patterns before rewriting — clichés, repetition, unnatural openers
- **File Upload**: TXT, DOCX, PDF support with magic-byte validation
- **Download**: Export as plain text or Word document
- **History**: Browse and restore past humanizations
- **Undo/Redo**: 20-step undo stack
- **Multi-provider**: OpenAI, Anthropic Claude, Google Gemini, Ollama (local), or any custom OpenAI-compatible endpoint
- **British/American English**: Automatic spelling variant conversion

---

## Quick Start

### Requirements

- Python 3.12+
- 2 GB RAM minimum (4 GB recommended if using local sentence-transformers)
- An API key for at least one LLM provider

### Installation

```bash
# 1. Clone or download the project
cd humanizer

# 2. Run setup (creates venv, installs dependencies, downloads NLP models)
python setup.py

# 3. Add your API key to .env
#    Edit .env and set OPENAI_API_KEY (or ANTHROPIC_API_KEY, etc.)
nano .env   # or notepad .env on Windows

# 4. Start the server
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python run.py

# 5. Open in browser
#    http://127.0.0.1:8000
```

For a lightweight install (no PyTorch/transformers — semantic similarity uses Jaccard fallback):
```bash
python setup.py --light
```

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Active provider: `openai` \| `anthropic` \| `cohere` \| `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Model name |
| `COHERE_API_KEY` | — | Cohere API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model pulled in Ollama |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model (`sm` / `md` / `lg`) |
| `LLM_TEMPERATURE` | `0.7` | Creativity (0=deterministic, 1=creative) |
| `LLM_MAX_TOKENS` | `4096` | Maximum output tokens |
| `UPLOAD_MAX_SIZE_MB` | `10` | File upload size limit |
| `HISTORY_RETENTION_DAYS` | `30` | Auto-prune history after N days |

---

## Project Structure

```
humanizer/
├── backend/
│   ├── api/
│   │   ├── health.py          # GET /health
│   │   ├── humanize.py        # POST /api/v1/humanize  (+ batch + analyze)
│   │   ├── history.py         # GET/DELETE /api/v1/history
│   │   ├── providers.py       # GET/POST /api/v1/providers
│   │   └── upload.py          # POST /api/v1/upload  + download
│   ├── core/
│   │   ├── analyzer.py        # NLP analysis pipeline (spaCy + NLTK)
│   │   ├── engine.py          # Humanization orchestrator
│   │   ├── prompts.py         # All LLM prompt templates
│   │   └── schemas.py         # Pydantic request/response models
│   ├── db/
│   │   └── database.py        # SQLAlchemy async ORM + CRUD
│   ├── preservation/
│   │   └── guardian.py        # Placeholder injection system
│   ├── providers/
│   │   ├── base.py            # Abstract LLM provider + registry
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── other_providers.py # Cohere + Ollama
│   ├── quality/
│   │   └── evaluator.py       # 8-metric quality engine
│   ├── utils/
│   │   └── file_parser.py     # Secure TXT/DOCX/PDF extraction
│   └── main.py                # FastAPI app + lifespan + routing
├── config/
│   ├── settings.py            # Pydantic settings (all config in one place)
│   └── logging_config.py      # Loguru configuration
├── frontend/
│   ├── index.html             # Single-page app shell
│   └── static/
│       ├── css/main.css       # Complete UI stylesheet
│       └── js/
│           ├── api.js         # All fetch calls to backend
│           ├── ui.js          # DOM rendering functions
│           └── app.js         # Application controller
├── tests/
│   ├── test_analyzer.py       # NLP analysis unit tests
│   ├── test_preservation.py   # Guardian placeholder tests
│   ├── test_quality.py        # Quality evaluator tests
│   ├── test_api.py            # FastAPI integration tests
│   └── test_file_parser.py    # File parser security tests
├── data/                      # Auto-created: DB + uploads
├── .env.example               # Configuration template
├── requirements.txt
├── setup.py                   # One-command setup script
├── run.py                     # Server entry point
└── pytest.ini
```

---

## API Reference

Full interactive docs at `/docs` once the server is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | App health check |
| `/api/v1/humanize` | POST | Humanize a single text |
| `/api/v1/humanize/batch` | POST | Humanize up to 20 texts |
| `/api/v1/analyze` | POST | NLP analysis without rewriting |
| `/api/v1/providers` | GET | List all LLM providers |
| `/api/v1/providers/switch` | POST | Change active provider |
| `/api/v1/providers/{name}/ping` | GET | Test provider connectivity |
| `/api/v1/upload` | POST | Upload TXT/DOCX/PDF |
| `/api/v1/download/txt` | POST | Download text as .txt |
| `/api/v1/download/docx` | POST | Download text as .docx |
| `/api/v1/history` | GET | Paginated history list |
| `/api/v1/history/{id}` | GET | Full history record |
| `/api/v1/history/{id}` | DELETE | Soft-delete a record |
| `/api/v1/history` | DELETE | Clear all history |

### Example: Humanize request

```python
import httpx

response = httpx.post("http://localhost:8000/api/v1/humanize", json={
    "text": "In conclusion, it is important to note that AI has significantly impacted...",
    "mode": "natural",
    "intensity": "balanced",
    "granularity": "document",
    "english_variant": "american",
    "preserve_citations": True,
    "preserve_technical_terms": True,
})

data = response.json()
print(data["humanized_text"])
print(f"Quality score: {data['quality']['overall_score']}/100")
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=backend --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Single module
pytest tests/test_analyzer.py -v
```

---

## Using Ollama (Local Models — No API Key)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2       # or mistral, phi3, gemma2, etc.

# Set in .env:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2

python run.py
```

---

## Using a Custom OpenAI-Compatible Endpoint (LM Studio, OpenRouter)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=your-local-model-name
```

---

## Security Notes

- API keys are read from `.env` server-side — never sent to the browser
- Uploaded files are validated with magic-byte checks before parsing
- File size is capped before reading into memory
- No uploaded content is logged or stored long-term by default
- All document text is kept private — delete endpoints fully erase records
- The `.env` file should never be committed to version control

---

## Performance

| Config | Typical Request Time |
|--------|----------------------|
| GPT-4o (OpenAI) | 5–15 s |
| Claude Sonnet (Anthropic) | 6–12 s |
| Llama 3.2 (Ollama, GPU) | 10–25 s |
| GPT-4o-mini | 3–8 s |

Quality evaluation adds ~1–3 s when sentence-transformers is installed.

---

## Licence

MIT — see LICENSE for details.
