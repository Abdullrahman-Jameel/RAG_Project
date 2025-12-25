from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rag_supabase import SupabaseRAG
from pydantic import BaseModel
import os
import shutil

app = FastAPI(title="RAG Microservice")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG
rag = SupabaseRAG()

# Create uploads directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return FileResponse("static/page1.html")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and store in database
        num_chunks = rag.load_and_process_document(file_path)
        
        return {
            "success": True,
            "message": f"Processed {num_chunks} chunks from {file.filename}",
            "filename": file.filename
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents with streaming response"""
    async def generate():
        async for chunk in rag.query_stream(request.question):
            # Server-Sent Events format
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.delete("/clear")
async def clear_database():
    """Clear all documents"""
    try:
        rag.clear_database()
        return {"success": True, "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
