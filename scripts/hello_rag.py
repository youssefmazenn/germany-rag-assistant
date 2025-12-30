import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: str = "data"):
    """Load PDFs/TXT files from data/ folder into LangChain Document objects."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Create a 'data/' folder and add a PDF or TXT file.")

    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)

        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif fname.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    if not docs:
        raise ValueError("No documents found. Add at least one PDF or TXT to data/")

    return docs


def build_vectorstore(docs, persist_dir: str = "chroma_db"):
    """
    Chunk documents + embed chunks + store them in Chroma.
    Persisting means we can reuse the DB without recomputing embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    return vectordb


def retrieve(vectordb, query: str, k: int = 4):
    """Return the top-k most similar chunks for a query."""
    return vectordb.similarity_search(query, k=k)


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} pages. Building vector store...")
    vectordb = build_vectorstore(docs)

    question = (
        "When does the 18-month job-seeker residence permit start after graduation?"
    )
    print(f"\nQuestion: {question}\n")

    results = retrieve(vectordb, question, k=4)

    print("Top retrieved chunks (preview):\n")
    for i, r in enumerate(results, 1):
        snippet = r.page_content[:350].replace("\n", " ")
        source = r.metadata.get("source", "unknown")
        page = r.metadata.get("page", "n/a")
        print(f"[{i}] source={source}, page={page}\n{snippet}\n")
