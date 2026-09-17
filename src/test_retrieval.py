from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "jazzercise_schedule"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ---------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------

def load_vectorstore():

    embeddings = create_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIRECTORY)
    )


# ---------------------------------------------------------
# RETRIEVAL TEST
# ---------------------------------------------------------

def run_test(
    vectorstore,
    test_name,
    query,
    expected_text,
    metadata_filter=None
):

    print()
    print("=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

    print("\nQUESTION:")
    print(query)

    if metadata_filter:

        print("\nRETRIEVAL STRATEGY:")
        print("Metadata-filtered vector retrieval")

        print("\nFILTER:")
        print(metadata_filter)

        results = vectorstore.similarity_search_with_score(
            query,
            k=4,
            filter=metadata_filter
        )

    else:

        print("\nRETRIEVAL STRATEGY:")
        print("Semantic vector retrieval")

        results = vectorstore.similarity_search_with_score(
            query,
            k=4
        )

    expected_found = False
    expected_rank = None

    print("\nTOP RETRIEVED DOCUMENTS")
    print("-" * 80)

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print()
        print(f"RANK {rank}")
        print(f"Distance score: {score:.4f}")

        print("\nCONTENT:")
        print(document.page_content)

        print("\nSOURCE:")
        print(document.metadata.get("source"))

        print("\nMETADATA:")
        print(document.metadata)

        print("-" * 80)

        if (
            expected_text.lower()
            in document.page_content.lower()
        ):
            expected_found = True

            if expected_rank is None:
                expected_rank = rank

    print("\nRETRIEVAL RESULT:")

    if expected_found:

        print(
            f"PASS - Expected evidence "
            f"'{expected_text}' found at Rank "
            f"{expected_rank}."
        )

    else:

        print(
            f"FAIL - Expected evidence "
            f"'{expected_text}' was not retrieved."
        )

    return expected_found


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("JAZZERCISE RAG RETRIEVAL TEST — IMPROVED")
    print("=" * 80)

    print("\nLoading ChromaDB...")

    vectorstore = load_vectorstore()

    tests = [

        # -------------------------------------------------
        # Semantic rule retrieval
        # -------------------------------------------------

        {
            "name": "Second Tuesday room exception",

            "query": (
                "Which room does the Waikiki Community "
                "Center class use on the second Tuesday "
                "of the month?"
            ),

            "expected": "Room 205E",

            "filter": None
        },

        # -------------------------------------------------
        # Semantic historical-pattern retrieval
        # -------------------------------------------------

        {
            "name": "St. Peters Monday class time",

            "query": (
                "What time is the St. Peters "
                "Jazzercise class on Monday?"
            ),

            "expected": "5:00 PM",

            "filter": None
        },

        # -------------------------------------------------
        # Exact-date retrieval
        # -------------------------------------------------

        {
            "name": "Waikiki January 19 cancellation",

            "query": (
                "Is there a Waikiki Community Center "
                "class on January 19, 2026?"
            ),

            "expected": "NO CLASS",

            "filter": {
                "$and": [
                    {
                        "date": "2026-01-19"
                    },
                    {
                        "location":
                        "Waikiki Community Center"
                    }
                ]
            }
        }
    ]

    passed = 0

    for test in tests:

        result = run_test(
            vectorstore=vectorstore,
            test_name=test["name"],
            query=test["query"],
            expected_text=test["expected"],
            metadata_filter=test["filter"]
        )

        if result:
            passed += 1

    print()
    print("=" * 80)
    print("IMPROVED RETRIEVAL TEST SUMMARY")
    print("=" * 80)

    print(
        f"Tests passed: "
        f"{passed}/{len(tests)}"
    )

    if passed == len(tests):

        print(
            "PASS - Improved retrieval tests successful."
        )

    else:

        print(
            "REVIEW - Retrieval quality still "
            "needs improvement."
        )


if __name__ == "__main__":
    main()