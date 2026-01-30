from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str

def retrieve(state: RAGState):
    docs = retriever.get_relevant_documents(state["question"])
    context = [doc.page_content for doc in docs]
    return {"context": context}

def generate(state: RAGState):
    context_text = "\n\n".join(
        [f"[{i+1}] {chunk}" for i, chunk in enumerate(state["context"])]
    )

    prompt = PromptTemplate.from_template("""
You are an AI assistant answering strictly from the provided PDF context.

Rules:
- Use ONLY the context below
- If answer is not present, say: "The document does not contain this information."
- Cite sources using [1], [2], etc.

Context:
{context}

Question: {question}
Answer:
""")

    chain = prompt | llm
    response = chain.invoke({
        "context": context_text,
        "question": state["question"]
    })

    return {"answer": response.content}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()
