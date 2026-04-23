import os
from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.retriever import create_vector_store, get_retriever
from src.llm import get_llm
from src.rag_pipeline import create_rag_graph
from dotenv import load_dotenv

load_dotenv()

def main():
    # Setup
    pdf_path = "data/knowledge_base.pdf"
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(pdf_path):
        print(f"Error: Knowledge base not found at {pdf_path}")
        print("Please place your 'knowledge_base.pdf' in the 'data' folder.")
        return

    print("--- Initializing RAG System with Groq ---")
    embedding_model = get_embedding_model()
    llm = get_llm()
    
    # Ingest document if chroma_db doesn't exist
    if not os.path.exists("./chroma_db"):
        print("Processing knowledge base...")
        docs = load_pdf(pdf_path)
        chunks = chunk_documents(docs)
        create_vector_store(chunks, embedding_model)
        print("Embeddings stored in ChromaDB.")
    
    retriever = get_retriever(embedding_model)
    graph = create_rag_graph(retriever, llm)
    
    print("RAG System Ready. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        initial_state = {
            "query": user_input,
            "context": "",
            "response": "",
            "intent": "general",
            "confidence": 1.0,
            "history": [],
            "hitl_required": False
        }
        
        try:
            final_state = graph.invoke(initial_state)
            print(f"Bot: {final_state['response']}")
            if final_state.get("hitl_required"):
                print("[System Note: This query was flagged for human review]")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
