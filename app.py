from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import rag_app, vectorstore

app = FastAPI(title="Agentic AI RAG Chatbot")

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    result = rag_app.invoke({"question": query.question})

    docs_and_scores = vectorstore.similarity_search_with_score(query.question, k=4)

    contexts = [doc.page_content for doc, score in docs_and_scores]
    scores = [score for doc, score in docs_and_scores]

    confidence = round(1 / (1 + sum(scores)/len(scores)), 2)

    return {
        "answer": result["answer"],
        "contexts": contexts,
        "confidence": confidence
    }
