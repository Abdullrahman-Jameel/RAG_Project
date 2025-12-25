from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        
    def load_documents(self, file_path: str):
        print(f"📄 Loading documents from: {file_path}")
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file type. Use .pdf or .txt")
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s)")
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        print(f"✂️  Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks):
        print("🧮 Creating embeddings and building vector store...")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        print("✅ Vector store created successfully")
        return self.vector_store
    
    def setup_retriever(self, k=4):
        print(f"🔍 Setting up retriever (k={k})")
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': k})
        print("✅ Retriever ready")
        return self.retriever
    
    def create_rag_chain(self):
        print("🔗 Building RAG chain...")
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # LCEL chain - modern LangChain syntax
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ RAG chain created")
        return self.rag_chain
    
    def query(self, question: str):
        print(f"\n❓ Question: {question}")
        
        answer = self.rag_chain.invoke(question)
        source_docs = self.retriever.invoke(question)
        
        print(f"💡 Answer: {answer}\n")
        print(f"📚 Used {len(source_docs)} source chunks")
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.page_content[:200] + "..." for doc in source_docs]
        }
    
    def save_vector_store(self, path="faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"💾 Vector store saved to {path}")
    
    def load_vector_store(self, path="faiss_index"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"📂 Vector store loaded from {path}")
        return self.vector_store


if __name__ == "__main__":
    rag = RAGPipeline()
    
    documents = rag.load_documents("test_document.txt")
    chunks = rag.split_documents(documents, chunk_size=500, chunk_overlap=50)
    rag.create_vector_store(chunks)
    rag.setup_retriever(k=3)
    rag.create_rag_chain()
    
    print("\n" + "="*60)
    response = rag.query("What is RAG?")
    print(f"\nAnswer: {response['answer']}")
