import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"
MANIFEST_PATH = "data_manifest.json"
PDF_DIR = os.path.join("data_raw", "pdfs")


@dataclass
class DocMeta:
    doc_id: str
    title: str
    authority: str
    category: str
    filename: str
    notes: str = ""


def load_manifest(path: str) -> List[DocMeta]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Create it in the repo root.")

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs = []
    for it in items:
        docs.append(
            DocMeta(
                doc_id=it["doc_id"],
                title=it.get("title", ""),
                authority=it.get("authority", ""),
                category=it.get("category", ""),
                filename=it["filename"],
                notes=it.get("notes", ""),
            )
        )
    return docs


def reset_vector_db(persist_dir: str) -> None:
    """Delete old DB to ensure ingestion is reproducible."""
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)


def ingest_documents(manifest: List[DocMeta]) -> List:
    """
    Load PDFs and return a list of LangChain Document objects.
    We also attach rich metadata for citations and auditing later.
    """
    all_pages = []

    for doc in manifest:
        pdf_path = os.path.join(PDF_DIR, doc.filename)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Attach metadata for every page
        for p in pages:
            p.metadata.update(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "authority": doc.authority,
                    "category": doc.category,
                    "source_file": doc.filename,
                }
            )

        all_pages.extend(pages)

    return all_pages


def build_vectorstore(pages) -> Chroma:
    """
    Chunk pages, embed chunks, store in Chroma.
    Chunking settings matter a lot for legal/policy text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


def main():
    print("Loading manifest...")
    manifest = load_manifest(MANIFEST_PATH)
    print(f"Manifest entries: {len(manifest)}")

    print("Resetting local vector DB...")
    reset_vector_db(PERSIST_DIR)

    print("Loading PDFs + attaching metadata...")
    pages = ingest_documents(manifest)
    print(f"Loaded pages: {len(pages)}")

    print("Building vector store (chunk -> embed -> store)...")
    vectordb = build_vectorstore(pages)

    print("\n✅ Ingestion complete.")
    print(f"Chroma DB persisted at: {PERSIST_DIR}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Total chunks indexed: {vectordb._collection.count()}")


if __name__ == "__main__":
    main()
