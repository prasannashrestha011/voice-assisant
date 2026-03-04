from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

# Use Ollama LLM
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.3,
    think=False,        # disables Qwen3 thinking mode
    device="cuda"       # use GPU if available
)

# --- Define the state ---
class ProjectState(TypedDict):
    project_type: str       # e.g., "agentic AI study assistant"
    features: str           # planner output
    impl_steps: str         # writer output
    testing_tips: str       # editor output

# --- Planner Node ---
def planner(state: ProjectState) -> dict:
    prompt = (
        f"List the key features of a {state['project_type']} project.\n"
        f"Provide them in clear bullet points, concise and actionable."
    )
    result = llm.invoke(prompt)
    content = result.content or result.additional_kwargs.get("content", "")
    print(f"📋 Features done ({len(content)} chars)")
    return {"features": content}

# --- Writer Node ---
def writer(state: ProjectState) -> dict:
    print(f"📝 Writer received features:\n{state['features'][:100]}...")
    prompt = (
        f"Write a step-by-step implementation plan for a {state['project_type']} "
        f"based on these features:\n{state['features']}\n\nProvide clear numbered steps."
    )
    result = llm.invoke(prompt)
    content = result.content or result.additional_kwargs.get("content", "")
    print(f"✍️ Implementation steps done ({len(content)} chars)")
    return {"impl_steps": content}

# --- Editor Node ---
def editor(state: ProjectState) -> dict:
    print(f"✏️ Editor received implementation steps:\n{state['impl_steps'][:100]}...")
    prompt = (
        f"Review these implementation steps and provide practical testing tips, "
        f"highlight common pitfalls, and improvements:\n{state['impl_steps']}"
    )
    result = llm.invoke(prompt)
    content = result.content or result.additional_kwargs.get("content", "")
    print(f"✅ Testing tips generated ({len(content)} chars)")
    return {"testing_tips": content}

# --- Build the graph ---
builder = StateGraph(ProjectState)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("editor", editor)

builder.add_edge(START, "planner")
builder.add_edge("planner", "writer")
builder.add_edge("writer", "editor")
builder.add_edge("editor", END)

graph = builder.compile()

# --- Invoke the graph ---
print("⏳ Planning project...")
result = graph.invoke({"project_type": "agentic AI study assistant"})
print("✅ Done!\n")

# --- Output all stages for full trace ---
print("=== Features ===")
print(result["features"], "\n")
print("=== Implementation Steps ===")
print(result["impl_steps"], "\n")
print("=== Testing Tips ===")
print(result["testing_tips"])