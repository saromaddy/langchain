import os
from dotenv import load_dotenv

load_dotenv("../.env")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
print(os.environ["GOOGLE_API_KEY"])

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --------------------------------------
# STEP 1: Load PDF
# --------------------------------------

pdf_path = "Saravana_Kumar_T_Resume.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# --------------------------------------
# STEP 2: Chunk Documents
# --------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# --------------------------------------
# STEP 3: Create Embeddings
# --------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# --------------------------------------
# STEP 4: Create Vector Database
# --------------------------------------

vector_store = FAISS.from_documents(
    docs,
    embeddings
)

print("Vector DB Created")

# --------------------------------------
# STEP 5: Create Retriever
# --------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------------
# STEP 6: Load Gemini
# --------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------------------------
# STEP 7: Prompt Template
# --------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

Answer:
"""
)

# --------------------------------------
# STEP 8: Ask Questions
# --------------------------------------

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    print("\nAnswer:")
    print(response.content)

    print("\n" + "=" * 80)