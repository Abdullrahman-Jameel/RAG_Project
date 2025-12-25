# RAG Microservice with FastAPI, LangChain, and Supabase

This project is a complete Retrieval-Augmented Generation (RAG) microservice built with a modern Python stack. It allows you to upload documents (.pdf or .txt) and ask questions about their content. The backend is a streaming-first FastAPI application, and it uses Supabase for a scalable vector store and LangChain for orchestrating the RAG pipeline.

## Features

-   **FastAPI Backend:** A high-performance, asynchronous web framework.
-   **Document Upload:** Upload `.pdf` and `.txt` files through a simple API endpoint.
-   **RAG Pipeline:** Uses LangChain to orchestrate document loading, chunking, embedding, and retrieval.
-   **OpenAI Integration:** Leverages OpenAI's powerful models for embeddings (`text-embedding-3-small`) and question-answering (`gpt-3.5-turbo`).
-   **Supabase Vector Store:** Uses Supabase's Postgres database with the `pgvector` extension for a scalable and managed vector store.
-   **Streaming Responses:** Answers to your questions are streamed back token-by-token for a real-time experience.
-   **Simple Frontend:** A basic HTML frontend is provided to interact with the backend.
-   **Easy to Run:** The project can be run locally with just a few commands.

## Tech Stack

-   **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
-   **RAG & AI:** [LangChain](https://python.langchain.com/), [OpenAI](https://openai.com/)
-   **Vector Database:** [Supabase](https://supabase.io/) (with `pgvector`)
-   **Dependencies:** `pydantic`, `python-dotenv`, `pypdf`, `tiktoken`, `numpy`, `gunicorn`

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

Before you begin, you will need:

-   **Python 3.8+**
-   A **Supabase account**. You will need to create a project and get the URL and service key.
-   An **OpenAI API key**.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abdullrahman-Jameel/RAG_Project.git
    cd RAG_Project
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    Create a file named `.env` in the root of the project and add your Supabase and OpenAI credentials:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    SUPABASE_URL="your_supabase_url"
    SUPABASE_SERVICE_KEY="your_supabase_service_key"
    ```

4.  **Set up the Supabase database:**
    In your Supabase project's SQL Editor, run the following SQL script to create the `documents` table and the `match_documents` function. This is a one-time setup.

    ```sql
    -- Enable the pgvector extension
    create extension if not exists vector;

    -- Create the documents table
    create table documents (
      id bigserial primary key,
      content text not null,
      metadata jsonb,
      embedding vector(1536)
    );

    -- Create the match_documents function for similarity search
    create or replace function match_documents (
      query_embedding vector(1536),
      match_threshold float,
      match_count int
    )
    returns table (
      id bigint,
      content text,
      metadata jsonb,
      similarity float
    )
    language sql stable
    as $$
      select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where 1 - (documents.embedding <=> query_embedding) > match_threshold
      order by similarity desc
      limit match_count;
    $$;
    ```

### Running the Application

Once you have completed the installation and setup, you can run the application with:
```bash
uvicorn backend:app --reload
```
The application will be available at `http://localhost:8000`.

## API Endpoints

The following are the main API endpoints:

-   `GET /`: Serves the main HTML page.
-   `POST /upload`: Upload a document.
    -   **File:** `file` (the document to upload)
-   `POST /query`: Ask a question.
    -   **Body:** `{"question": "Your question here"}`
-   `DELETE /clear`: Deletes all documents from the database.
-   `GET /health`: Health check endpoint.

You can interact with these endpoints using the provided HTML page or a tool like `curl` or Postman.

## Project Structure

```
.
├── backend.py            # FastAPI application
├── rag_supabase.py       # RAG pipeline with Supabase
├── rag_pipeline.py       # RAG pipeline with local FAISS (example)
├── requirements.txt      # Python dependencies
├── static/page1.html     # Simple HTML frontend
├── test_document.txt     # A sample document for testing
├── .env                  # Environment variables (you need to create this)
└── README.md             # This file
```
