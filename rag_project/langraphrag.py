import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv("../.env",override=True)

# =====================================================
# PDF LOADING
# =====================================================

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Saravana_Kumar_T_Resume.pdf")
documents = loader.load()

# =====================================================
# CHUNKING
# =====================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print('chunks',chunks)
print(f"Chunks Created: {len(chunks)}")

# =====================================================
# EMBEDDINGS
# =====================================================

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# =====================================================
# FAISS
# =====================================================

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print('retriever',retriever)

# =====================================================
# GEMINI
# =====================================================

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# =====================================================
# STATE
# =====================================================

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    reviewed_answer: str

# =====================================================
# RETRIEVER NODE
# =====================================================

def retriever_node(state: GraphState):

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }

# =====================================================
# ANSWER NODE
# =====================================================

def answer_node(state: GraphState):

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the supplied context.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# =====================================================
# REVIEWER NODE
# =====================================================

def reviewer_node(state: GraphState):

    review_prompt = f"""
You are a senior reviewer.

Check whether the answer is grounded
in the supplied context.

If grounded:
Return the answer unchanged.

If not grounded:
Return a corrected answer.

Context:
{state['context']}

Answer:
{state['answer']}

Improve the answer if necessary.
"""

    response = llm.invoke(review_prompt)

    return {
        "reviewed_answer": response.content
    }

# =====================================================
# LANGGRAPH
# =====================================================

from langgraph.graph import StateGraph
from langgraph.graph import END

builder = StateGraph(GraphState)

builder.add_node(
    "retriever",
    retriever_node
)

builder.add_node(
    "answer",
    answer_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.set_entry_point(
    "retriever"
)

builder.add_edge(
    "retriever",
    "answer"
)

builder.add_edge(
    "answer",
    "reviewer"
)

builder.add_edge(
    "reviewer",
    END
)

graph = builder.compile()

# =====================================================
# CHAT LOOP
# =====================================================

print("\nLangGraph RAG Ready")
print("Type exit to quit\n")

while True:

    question = input("Ask Question: ")

    if question.lower() == "exit":
        break

    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "reviewed_answer": ""
        }
    )

    print("\nFINAL ANSWER:\n")
    print(result["reviewed_answer"])

    print("\n" + "=" * 80)