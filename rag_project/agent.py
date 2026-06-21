import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ==================================================
# DOCUMENT LOADING
# ==================================================

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Saravana_Kumar_T_Resume.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# ==================================================
# CHUNKING
# ==================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# ==================================================
# EMBEDDINGS
# ==================================================

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# ==================================================
# VECTOR STORE
# ==================================================

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

print("Vector Store Ready")

# ==================================================
# RETRIEVER
# ==================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==================================================
# GEMINI MODEL
# ==================================================

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# ==================================================
# TOOL
# ==================================================

from langchain.tools import tool

@tool
def document_search(question: str) -> str:
    """
    Search the PDF knowledge base
    and return relevant context.
    """

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context

# ==================================================
# AGENT
# ==================================================

from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[document_search],
    system_prompt="""
You are a RAG assistant.

Always use the document_search tool
before answering any question.

Answer ONLY from retrieved context.

If the answer is not available in the document,
say:

'I could not find this information in the document.'
"""
)

# ==================================================
# CHAT LOOP
# ==================================================

print("\nRAG Agent Ready")
print("Type exit to quit\n")

while True:

    question = input("Ask Question: ")

    if question.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\nAnswer:\n")

    print(
        result["messages"][-1].content
    )

    print("\n" + "=" * 80)