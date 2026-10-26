# Monopoly Rules RAG API (LlamaIndex + FastAPI + Gemini)

A Retrieval-Augmented Generation (RAG) API built with **LlamaIndex**, **FastAPI**, and **Google Gemini** that parses the official Monopoly rulebook PDF and answers complex gameplay questions using **Metadata Extraction**, **Recursive Retrieval**, and **Cross-Encoder Reranking**.

---

## Architecture & How It Works

Instead of relying on basic naive chunking, this pipeline implements an advanced multi-stage retrieval workflow:

1. **Document Ingestion & Base Chunking (`rag_agent.py`)**
   * Reads `assets/monopoly-manual.pdf` using `PDFReader`.
   * Splits the document into base chunks (`1024` tokens) with deterministic IDs (`node-0`, `node-1`, etc.) via `SentenceSplitter`.
2. **LLM Metadata Extraction (`rag_agent.py`)**
   * Uses `GoogleGenAI` (`gemini-3.1-flash-lite`) with `SummaryExtractor` and `QuestionsAnsweredExtractor` to generate summaries and 5 hypothetical questions that each base chunk can answer.
   * Links these smaller metadata `IndexNode` objects back to their parent `BaseNode` chunks.
3. **Local Embeddings & Vector Persistence (`populate_vector_database.py`)**
   * Embeds all parent and child nodes locally using HuggingFace's **`BAAI/bge-small-en`** embedding model.
   * Persists the index and document store to disk under `storage/monopoly_rag_data`.
4. **Recursive Retrieval & Reranking (`load_vector_database.py`)**
   * Loads the persisted index on FastAPI startup.
   * Uses a `RecursiveRetriever` (`similarity_top_k=15`) to match user queries against the specific summaries/questions and automatically fetch the full parent text chunk.
   * Reranks the top 15 retrieved chunks down to the **top 3** most relevant nodes using **`BAAI/bge-reranker-large`** (`FlagEmbeddingReranker`).
5. **FastAPI Async Serving (`main.py`)**
   * Loads the RAG pipeline once during the FastAPI `lifespan` startup context and exposes an asynchronous `POST /query` endpoint.

---

## Project Structure

```text
├── .devcontainer/               # VS Code Dev Container configuration
├── assets/
│   └── monopoly-manual.pdf      # Source PDF rulebook
├── storage/
│   └── monopoly_rag_data/       # Persisted vector index & docstore (generated)
├── rag_agent.py                 # Document parsing, chunking, and metadata extraction logic
├── populate_vector_database.py  # Script to build and save the vector index to disk
├── load_vector_database.py      # Loads persisted index and builds the RecursiveRetriever + Reranker
├── UserRequest.py               # Pydantic request model for the API
├── main.py                      # FastAPI application and /query endpoint
├── .env                         # Environment variables (Ignored by Git)
└── README.md
```

---

## Prerequisites

* **Python 3.11+** (or Docker Desktop + VS Code with the **Dev Containers** extension)
* **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

*(Optional) If using VS Code Dev Containers, open the folder in VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P`), and select **Dev Containers: Reopen in Container**.*

### 2. Install Dependencies

Install the required packages for LlamaIndex, FastAPI, local HuggingFace embeddings, and the FlagEmbedding reranker:

```bash
pip install \
  "fastapi[standard]" \
  pydantic \
  python-dotenv \
  llama-index-core \
  llama-index-readers-file \
  llama-index-llms-google-genai \
  llama-index-embeddings-huggingface \
  llama-index-postprocessor-flag-embedding-reranker \
  FlagEmbedding \
  pypdf \
  torch \
  sentence-transformers
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **Note:** Never commit your `.env` file to version control. Ensure `.env` is listed in your `.gitignore`.

---

## Running the Project

### Step 1: Populate the Vector Database

Before starting the FastAPI server for the first time, generate the embeddings, run the metadata extractors, and persist the vector index to `storage/monopoly_rag_data`:

```bash
python populate_vector_database.py
```

*Note: On the first run, this will download the `BAAI/bge-small-en` model and call the Gemini API to extract node summaries and questions. Wait until you see `saved.` in the terminal.*

### Step 2: Launch the FastAPI Server

Start the FastAPI development server:

```bash
fastapi dev main.py
```

*Note: On initial startup, the server will download and initialize the `BAAI/bge-reranker-large` model.*

Once running, the API will be available at:
* **API Endpoint:** `http://127.0.0.1:8000/query`
* **Interactive Swagger UI Docs:** `http://127.0.0.1:8000/docs`

---

## Usage & Testing

### Option A: Using the Swagger UI
1. Open `http://127.0.0.1:8000/docs` in your browser.
2. Click on **`POST /query`** -> **Try it out**.
3. Enter your question in the JSON request body and click **Execute**.

### Option B: Using `curl`

```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do we keep our games short and sweet?"}'
```

**Example Response:**
```json
{
  "response": "To keep games short and sweet, you should avoid using house rules, always auction properties when a player lands on them and chooses not to buy them, never loan money to other players or make deals to waive rent, and never put cash in the center of the board for landing on Free Parking."
}
```

---

## Sample Test Questions

Use these questions to test the accuracy of the recursive retriever and reranker against the Monopoly manual:

| Question | Expected Ground Truth |
| :--- | :--- |
| *"How much money does each player start with and what is the bill breakdown?"* | **M1,500 total**: 5x M1, 1x M5, 2x M10, 1x M20, 1x M50, 4x M100, and 2x M500. |
| *"If my opponent owns both Utilities and I roll an 8 on the dice, how much rent do I owe?"* | **M80** (When an owner has 2 Utilities, rent is $10\times$ the dice roll). |
| *"Can I buy Houses even when it is not my turn?"* | **Yes**, as soon as you own a complete color set, you can buy Houses without waiting for your turn. |
| *"What happens if I land on an unowned Street and decide not to buy it?"* | The Banker must immediately auction it to all players, with bidding starting at **M10**. |
| *"Do I get a cash bonus when landing on Free Parking?"* | **No**, nothing happens on Free Parking. |