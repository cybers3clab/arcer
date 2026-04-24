# ARCER: an Agentic RAG for the Automated Definition of Cyber Ranges

ARCER (Agentic RAG for the Automated Definition of Cyber Ranges) is an intelligent agent built with **LangGraph** and **LangChain** that automates the generation and deployment of cyber range (CR) configuration files. Given a natural-language description of a cyber range scenario, ARCER retrieves relevant documentation, generates a valid configuration file for the target CR framework (e.g. [CyRIS](https://github.com/cyb3rlab/cyris)), verifies its syntax via a remote API, and optionally deploys the cyber range.

---

## Features

- 🧠 **RAG-based generation** — loads Cyber Range (CR) platform documentation (`.yml` and `.pdf`) into an in-memory vector store with HuggingFace embeddings and uses MMR retrieval to ground configuration generation
- ✅ **Automatic syntax verification** — iteratively calls a remote CR Platoform API to check and fix the generated file until the syntax is correct
- 🚀 **One-click deployment** — triggers cyber range deployment on the remote host via API
- 🗂️ **Conversational memory** — uses LangGraph's `MemorySaver` to maintain context across multi-turn interactions
- 🔌 **Multi-LLM support** — compatible with OpenAI, Anthropic, Mistral, and HuggingFace models via LangChain

---

## Architecture

```
User prompt
    │
    ▼
LangGraph ReAct Agent
    ├── Tool: retrieval
    ├── Tool: verify_syntax
    └── Tool: deploy_cyber_range
```

---

## Requirements

- Python 3.10+
- A running **CyRIS API** instance (set the URL in `arcer.py`)
- API keys for the selected LLM provider and LangSmith (optional but recommended)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/arcer.git
cd arcer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key        # optional, for tracing
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

### 4. Set the CyRIS API URL

In `arcer.py`, replace the placeholder with your CyRIS API base URL:

```python
api_base_url = "http://<your-cyris-host>:<port>/"
```

### 5. Add CyRIS documentation

Place your CyRIS reference files (`.yml` examples and `.pdf` docs) inside the `./cyris_docs/` folder. ARCER will automatically load and index them at startup.

```
cyris_docs/
├── example_scenario.yml
├── cyris_documentation.pdf
└── ...
```

---

## Usage

Run the agent:

```bash
python arcer.py
```

By default, ARCER sends the following messages to the agent:

1. `"Hi, please write me a configuration file for CyRIS-based CR. You choose all the scenario characteristics."`
2. `"Fix any errors and do the syntax verification step until the output is correct."`

You can customise the scenario by editing the `messages` list in the `main()` function. An example of a detailed prompt is already included as a comment in the source code.

---

## Project Structure

```
arcer/
├── arcer.py           # Main agent: setup, tools, RAG, and entry point
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (not committed)
└── cyris_docs/        # CyRIS documentation and example YAML files
```