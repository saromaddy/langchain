import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

# ==================================================
# PDF LOADER
# ==================================================

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Saravana_Kumar_T_Resume.pdf")
documents = loader.load()

# ==================================================
# CHUNKING
# ==================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# ==================================================
# EMBEDDINGS
# ==================================================

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# ==================================================
# VECTOR STORE
# ==================================================

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==================================================
# GEMINI
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ==================================================
# TOOL
# ==================================================

from langchain.tools import tool

@tool
def document_search(question: str) -> str:
    """
    Search resume knowledge base.
    """

    docs = retriever.invoke(question)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ==================================================
# LANGGRAPH STATE
# ==================================================

class GraphState(TypedDict):
    question: str
    context: str
    answer: str

# ==================================================
# RETRIEVER AGENT NODE
# ==================================================

def retriever_agent(state: GraphState):

    context = document_search.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "context": context
    }

# ==================================================
# ANSWER AGENT NODE
# ==================================================

def answer_agent(state: GraphState):

    prompt = f"""
You are an AI Resume Assistant.

Answer ONLY using the provided context.

Context:
{state['context']}

Question:
{state['question']}

If answer not found say:
'I could not find this information in the document.'
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# ==================================================
# REVIEWER AGENT NODE
# ==================================================

def reviewer_agent(state: GraphState):

    review_prompt = f"""
You are a quality reviewer.

Context:
{state['context']}

Answer:
{state['answer']}

Instructions:

1. If the answer is correct, return EXACTLY the answer.
2. Do not say 'correct'.
3. Do not explain your review.
4. Do not add comments.
5. Only return the final answer text.
6. If incorrect, return a corrected version.

Final Answer:
"""

    response = llm.invoke(review_prompt)

    return {
        "answer": response.content
    }

# ==================================================
# LANGGRAPH
# ==================================================

from langgraph.graph import (
    StateGraph,
    END
)

builder = StateGraph(GraphState)

builder.add_node(
    "retriever_agent",
    retriever_agent
)

builder.add_node(
    "answer_agent",
    answer_agent
)

builder.add_node(
    "reviewer_agent",
    reviewer_agent
)

builder.set_entry_point(
    "retriever_agent"
)

builder.add_edge(
    "retriever_agent",
    "answer_agent"
)

builder.add_edge(
    "answer_agent",
    "reviewer_agent"
)

builder.add_edge(
    "reviewer_agent",
    END
)

graph = builder.compile()

# ==================================================
# FASTAPI
# ==================================================

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="LangGraph RAG API"
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():

    return {
        "status": "running"
    }

@app.post("/ask")
def ask(data: QuestionRequest):

    result = graph.invoke(
        {
            "question": data.question,
            "context": "",
            "answer": ""
        }
    )

    return {
        "question": data.question,
        "answer": result["answer"]
    }