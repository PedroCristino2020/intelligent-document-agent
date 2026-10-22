# Intelligent Document Agent

A document question-answering application built with Python, Gemini and Retrieval-Augmented Generation (RAG). It allows users to ask questions about PDF documents and receive answers grounded in retrieved text fragments, with source references.

## Features

- **PDF ingestion:** extract text from PDF documents and split it into chunks.
- **Embeddings:** generate vector representations using the Gemini API.
- **Semantic retrieval:** retrieve relevant document fragments using cosine similarity.
- **LLM-powered answers:** generate responses using Gemini and the retrieved context.
- **Source references:** display source documents, page numbers and retrieval similarity scores.
- **SQLite storage:** persist document chunks and embeddings.
- **Interactive interface:** ask questions through a Streamlit web application.

## Tech stack

- Python
- Google Gemini API (`google-genai`)
- Retrieval-Augmented Generation (RAG)
- NumPy
- SQLite
- Streamlit
- PyPDF

## Project structure

```text
intelligent-document-agent/
├── app.py                 # Streamlit web interface
├── src/
│   ├── ask.py             # Question-answering pipeline
│   ├── embeddings.py      # Embedding generation
│   ├── indexer.py         # PDF indexing pipeline
│   ├── llm.py             # LLM integration
│   └── retriever.py       # Semantic search
├── documents/             # Input PDF documents (not tracked)
├── data/                  # Local SQLite index (not tracked)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PedroCristino2020/intelligent-document-agent.git
cd intelligent-document-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the example value with your own Gemini API key. Never commit your `.env` file or publish your API key.

### 5. Add PDF documents

Place your PDF files in the `documents/` directory.

### 6. Build the document index

Run the indexing pipeline:

```bash
python -m src.indexer
```

### 7. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Usage

1. Enter a question about the indexed documents.
2. Choose how many relevant fragments to retrieve.
3. Click **Consultar documentos**.
4. Read the generated answer and expand the source sections to inspect the retrieved text, document, page and similarity score.

The quality of the answer depends on the indexed documents and the relevance of the retrieved context.

## Architecture

1. PDF files are processed into text chunks.
2. Gemini generates embeddings for the chunks.
3. Chunks and embeddings are stored in SQLite.
4. A user submits a question through Streamlit.
5. The question is embedded and compared with stored embeddings using cosine similarity.
6. The most relevant fragments are passed to Gemini as context.
7. The application displays the answer and the retrieved source fragments.

## Security and limitations

- API credentials are configured locally through environment variables.
- PDF files, the local index and secrets are excluded from version control.
- Answers are generated from retrieved context and may still be incomplete or incorrect; source references should be checked.
- The application currently runs locally and requires a configured Gemini API key.

## Author

**Pedro Cristino Moreno**

- GitHub: [PedroCristino2020](https://github.com/PedroCristino2020)
