from langchain_community.vectorstores import Chroma
import os

def create_vector_store(chunks, embedding_model, persist_directory="./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectorstore

def get_retriever(embedding_model, persist_directory="./chroma_db"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})
