from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from typing import List, Dict
import json

load_dotenv()

class SupabaseRAG:
    def __init__(self):
        print("🚀 Initializing SupabaseRAG...")
        
        # --- Configuration Validation ---
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable not set.")
        if not self.supabase_key:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable not set.")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        print(f"📡 Connecting to Supabase: {self.supabase_url}")
        
        # --- Client Initialization ---
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize LLM with streaming
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=True
        )
        
        print("✅ SupabaseRAG initialized")
        
    def load_and_process_document(self, file_path: str):
        """Load document, chunk it, and store in Supabase"""
        print(f"📄 Processing: {file_path}")
        
        # Load document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✂️  Created {len(chunks)} chunks")
        
        # Store in Supabase
        self._store_chunks(chunks, file_path)
        print(f"💾 Stored in Supabase database")
        
        return len(chunks)
    
    def _store_chunks(self, chunks, source_file):
        """Store document chunks with embeddings in Supabase using batch insertion"""
        print(f"🔄 Preparing {len(chunks)} chunks for batch insertion...")
        
        insert_data = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embeddings.embed_query(chunk.page_content)
            
            # Prepare metadata
            metadata = {
                "source": source_file,
                "chunk_index": i,
                "page": chunk.metadata.get("page", 0)
            }
            
            insert_data.append({
                "content": chunk.page_content,
                "metadata": json.dumps(metadata),
                "embedding": embedding
            })
        
        # Perform batch insert
        try:
            result = self.supabase.table('documents').insert(insert_data).execute()
            print(f"✅ Successfully batch inserted {len(insert_data)} chunks into database.")
        except Exception as e:
            print(f"❌ Error during batch insertion: {e}")
            raise
    
    def retrieve_relevant_chunks(self, question: str, k: int = 4) -> List[Dict]:
        """Retrieve relevant chunks using properly formatted vector"""
        print(f"\n🔍 Searching for: '{question}'")
        
        try:
            # Generate embedding
            query_embedding = self.embeddings.embed_query(question)
            print(f"✅ Generated query embedding (dimension: {len(query_embedding)})")
            
            # Check total documents
            count_result = self.supabase.table('documents').select('id', count='exact').execute()
            print(f"📊 Total documents in database: {count_result.count if hasattr(count_result, 'count') else len(count_result.data)}")
            
            print(f"🔎 Calling match_documents RPC...")
            
            # Call RPC with properly formatted vector string
            result = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.0,
                    'match_count': k
                }
            ).execute()
            
            if result.data:
                print(f"✅ Found {len(result.data)} relevant chunks")
                for i, doc in enumerate(result.data):
                    print(f"   Chunk {i+1}: similarity={doc.get('similarity', 'N/A'):.4f}")
                return result.data
            else:
                print(f"⚠️  No results found. Result: {result}")
                return []
            
        except Exception as e:
            print(f"❌ Error in retrieve_relevant_chunks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def query_stream(self, question: str):
        """Stream answer token by token"""
        print(f"\n❓ Question: {question}")
        
        # Retrieve relevant chunks
        relevant_docs = self.retrieve_relevant_chunks(question, k=3)
        
        if not relevant_docs:
            print("⚠️  No relevant documents found")
            yield "I don't have enough information to answer that question. Please make sure documents are uploaded and the database is set up correctly."
            return
        
        # Format context
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        print(f"📝 Context length: {len(context)} characters")
        
        # Create prompt
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer, say so. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Stream response
        chain = prompt | self.llm | StrOutputParser()
        
        print("🤖 Starting to stream response...")
        async for chunk in chain.astream({"context": context, "question": question}):
            yield chunk
        
        # Yield sources at the end
        yield "\n\n📚 **Sources:**\n"
        for i, doc in enumerate(relevant_docs, 1):
            metadata = json.loads(doc['metadata'])
            similarity = doc.get('similarity', 0)
            yield f"\n{i}. {metadata['source']} (page {metadata.get('page', 'N/A')}, similarity: {similarity:.2f})"
    
    def clear_database(self):
        """Clear all documents from database"""
        try:
            result = self.supabase.table('documents').delete().neq('id', 0).execute()
            print("🗑️  Database cleared")
            return True
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False
