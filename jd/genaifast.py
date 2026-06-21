from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv("../.env")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question:

    {question}
    """
)

chain = prompt | llm


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_ai(data: Query):

    response = chain.invoke(
        {
            "question": data.question
        }
    )

    return {
        "answer": response.content
    }