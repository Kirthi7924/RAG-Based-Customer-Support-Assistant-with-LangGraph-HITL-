from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(model_name="llama-3.3-70b-versatile", temperature=0):
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
