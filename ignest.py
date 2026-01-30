from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    print("🧠 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("💾 Saving to FAISS vector database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("✅ Ingestion complete! Vector index saved.")

if __name__ == "__main__":
    ingest_pdf()
