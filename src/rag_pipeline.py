from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class AgentState(TypedDict):
    query: str
    context: str
    response: str
    intent: str
    confidence: float
    history: List[BaseMessage]
    hitl_required: bool

def intent_router(state: AgentState):
    # Simplified intent routing logic
    query = state["query"].lower()
    if any(word in query for word in ["help", "human", "support", "complex"]):
        return "human_escalation"
    return "process_query"

def process_query_node(state: AgentState, retriever, llm):
    query = state["query"]
    
    # CASE 2: Explicit human request (Direct Hit)
    query_lower = query.lower()
    if any(word in query_lower for word in ["human", "agent", "support", "person", "escalate"]):
        return {
            "hitl_required": True,
            "intent": "explicit_escalation",
            "response": "Connecting you to a human agent now..."
        }

    # Retrieve context
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    # CASE 1: No context found at all
    if not context or len(context.strip()) < 10:
        return {
            "context": context,
            "response": "I’m not confident I can answer this accurately. Let me connect you to a human agent.",
            "hitl_required": True,
            "intent": "low_confidence"
        }
        
    # Generate answer if context is present
    prompt = f"""
    You are a Customer Support Assistant.
    
    [CONTEXT]
    {context}
    
    [USER QUERY]
    {query}
    
    [INSTRUCTIONS]
    1. Check if the [USER QUERY] is a **General Question** or a **Specific Personal Complaint**.
    
    2. If it is a **General Question** (e.g., "What are qualities of service?"):
       - Answer the question directly using information from the [CONTEXT].
       
    3. If it is a **Specific Personal Complaint** (e.g., "My order is delayed", "I was cheated"):
       - ONLY answer if the [CONTEXT] contains a specific company procedure or policy for that problem.
       - If the context only contains general advice or generic empathy, you MUST reply with: "TRIGGER_ESCALATION".
       
    4. If the [CONTEXT] does not contain the answer at all:
       - You MUST reply with: "TRIGGER_ESCALATION".
    """
    response = llm.invoke(prompt)
    
    # CASE 1: LLM determines answer is unclear/missing in context
    if "TRIGGER_ESCALATION" in response.content.upper():
        return {
            "context": context,
            "response": "I’m not confident I can answer this accurately. Let me connect you to a human agent.",
            "hitl_required": True,
            "intent": "low_confidence"
        }
        
    return {
        "context": context,
        "response": response.content,
        "hitl_required": False,
        "intent": "rag_answer"
    }

def human_node(state: AgentState):
    # This node ensures we use the correct message for escalation
    # If the response was already set in process_query_node, we keep it
    response = state.get("response")
    if not response or "ESCALATED" in response.upper():
        if state.get("intent") == "explicit_escalation":
            response = "Connecting you to a human agent now..."
        else:
            response = "I’m not confident I can answer this accurately. Let me connect you to a human agent."
            
    return {
        "response": response,
        "hitl_required": True
    }

def create_rag_graph(retriever, llm):
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("process", lambda state: process_query_node(state, retriever, llm))
    workflow.add_node("human", human_node)
    
    # Define Edges
    workflow.set_entry_point("process")
    
    # Conditional edge based on intent or low confidence
    workflow.add_conditional_edges(
        "process",
        lambda state: "human" if state.get("hitl_required") else END,
        {
            "human": "human",
            END: END
        }
    )
    
    workflow.add_edge("human", END)
    
    return workflow.compile()
