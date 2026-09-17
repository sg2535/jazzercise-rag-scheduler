from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from load_documents import load_all_documents


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "jazzercise_schedule"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# CREATE EMBEDDING MODEL
# ---------------------------------------------------------

def create_embeddings():
    """
    Create a local Hugging Face embedding model.

    No OpenAI API key is required.
    """

    print()
    print("Loading Hugging Face embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ---------------------------------------------------------
# BUILD VECTOR STORE
# ---------------------------------------------------------

def build_vectorstore():
    print()
    print("=" * 70)
    print("BUILDING JAZZERCISE VECTOR STORE")
    print("=" * 70)

    # Load our validated LangChain documents
    documents = load_all_documents()

    print()
    print(
        f"Documents ready for embedding: "
        f"{len(documents)}"
    )

    # Create local embedding model
    embeddings = create_embeddings()

    print()
    print("Creating embeddings and storing them in ChromaDB...")

    # Build persistent Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY)
    )

    print()
    print("=" * 70)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Documents embedded: "
        f"{len(documents)}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Chroma directory: "
        f"{CHROMA_DIRECTORY}"
    )

    return vectorstore


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vectorstore()