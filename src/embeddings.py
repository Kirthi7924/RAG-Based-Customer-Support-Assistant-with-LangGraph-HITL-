from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model():
    # Using a popular local model for embeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
