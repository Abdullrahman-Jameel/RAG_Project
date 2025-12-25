# RAG Microservice with FastAPI, LangChain, and Supabase

Unlock the power of your documents with this **Retrieval-Augmented Generation (RAG) Microservice**. Built with a cutting-edge Python stack, this project allows you to seamlessly upload PDF and text files, then query their content using natural language. Experience intelligent, context-aware answers delivered through a streaming-first FastAPI backend, powered by a scalable Supabase vector store and orchestrated by LangChain.

## 💡 How It Works

At its core, this RAG microservice integrates several powerful components:

1.  **Document Ingestion:** When you upload a document (PDF or TXT), it's automatically processed.
    *   **Loading:** LangChain document loaders extract text from the file.
    *   **Chunking:** The text is split into smaller, manageable "chunks" to optimize retrieval.
    *   **Embedding:** Each chunk is converted into a numerical vector (embedding) using OpenAI's `text-embedding-3-small` model.
    *   **Storage:** These embeddings, along with their original text content and metadata, are stored in a Supabase Postgres database with the `pgvector` extension.

2.  **Question Answering (RAG):** When you ask a question:
    *   **Query Embedding:** Your question is also converted into a numerical vector.
    *   **Retrieval:** The system searches the Supabase vector store to find the most semantically similar document chunks to your question.
    *   **Augmentation:** These retrieved chunks provide context to the Large Language Model (LLM).
    *   **Generation:** OpenAI's `gpt-3.5-turbo` LLM then uses this context to generate a concise and relevant answer, streamed back to you in real-time.

## Features

-   **FastAPI Backend:** A high-performance, asynchronous web framework.
-   **Document Upload:** Upload `.pdf` and `.txt` files through a simple API endpoint.
-   **RAG Pipeline:** Uses LangChain to orchestrate document loading, chunking, embedding, and retrieval.
-   **OpenAI Integration:** Leverages OpenAI's powerful models for embeddings (`text-embedding-3-small`) and question-answering (`gpt-3.5-turbo`).
-   **Supabase Vector Store:** Uses Supabase's Postgres database with the `pgvector` extension for a scalable and managed vector store.
-   **Streaming Responses:** Answers to your questions are streamed back token-by-token for a real-time experience.
-   **Simple Frontend:** A basic HTML frontend is provided to interact with the backend.

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

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Abdullrahman-Jameel/RAG_Project.git
    cd RAG_Project
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    # Activate the virtual environment:
    # On macOS/Linux: source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```
    Then install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a file named `.env` in the root of the project and add your Supabase and OpenAI credentials:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    SUPABASE_URL="your_supabase_url"
    SUPABASE_SERVICE_KEY="your_supabase_service_key"
    ```

4.  **Set up the Supabase database:**
    Navigate to your Supabase project's SQL Editor and execute the following script. This initializes the `documents` table and the `match_documents` function, essential for vector storage and retrieval. This is a one-time setup.

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

## 🚀 Running the Application

Once you have completed the installation and setup, you can run the application with:
```bash
uvicorn backend:app --reload
```
The application will be available at `http://localhost:8000`.

## 🌐 API Endpoints

Here's an overview of the primary API endpoints available in this microservice:

-   `GET /`: Serves the main HTML page.
-   `POST /upload`: Upload a document.
    -   **File:** `file` (the document to upload)
-   `POST /query`: Ask a question.
    -   **Body:** `{"question": "Your question here"}`
-   `DELETE /clear`: Deletes all documents from the database.
-   `GET /health`: Health check endpoint.

You can interact with these endpoints via the provided `static/page1.html` frontend, or directly using tools like `curl`, Postman, or Insomnia.

## 📁 Project Structure

A quick overview of the project's layout:

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

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📝 License

This project is licensed under the MIT License.
