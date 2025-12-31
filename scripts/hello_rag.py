from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    )

    questions = [
        "When does the 18-month job-seeker residence permit start after graduation?",
        "Can I work full-time while on the 18-month job-seeker residence permit?",
        "What are the main requirements for the EU Blue Card in Germany?",
        "How many days can I work as a student in Germany?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print(f"Question: {q}\n")

        results = vectordb.similarity_search(q, k=4)

        for i, r in enumerate(results, 1):
            snippet = r.page_content[:350].replace("\n", " ")
            meta = r.metadata
            print(
                f"[{i}] authority={meta.get('authority')} | doc_id={meta.get('doc_id')} | "
                f"source_file={meta.get('source_file')} | page={meta.get('page')}\n"
                f"{snippet}\n"
            )


if __name__ == "__main__":
    main()
