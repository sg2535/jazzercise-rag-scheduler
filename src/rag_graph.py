import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "jazzercise_schedule"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2:3b"


# =========================================================
# GRAPH STATE
# =========================================================

class RAGState(TypedDict):
    question: str
    route: str
    extracted_date: Optional[str]
    extracted_location: Optional[str]
    documents: List[Document]
    answer: str


# =========================================================
# EMBEDDINGS
# =========================================================

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


# =========================================================
# VECTOR STORE
# =========================================================

def load_vectorstore():

    embeddings = create_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIRECTORY)
    )


# =========================================================
# LOCAL LLM
# =========================================================

def create_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )


# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_date(question):

    month_pattern = (
        r"\b("
        r"January|February|March|April|May|June|"
        r"July|August|September|October|November|December"
        r")\s+(\d{1,2}),?\s+(\d{4})\b"
    )

    match = re.search(
        month_pattern,
        question,
        re.IGNORECASE
    )

    if not match:
        return None

    date_text = (
        f"{match.group(1)} "
        f"{match.group(2)} "
        f"{match.group(3)}"
    )

    parsed_date = datetime.strptime(
        date_text,
        "%B %d %Y"
    )

    return parsed_date.strftime(
        "%Y-%m-%d"
    )


# =========================================================
# LOCATION EXTRACTION
# =========================================================

def extract_location(question):

    lower = question.lower()

    if (
        "waikiki" in lower
        or "wkk" in lower
    ):
        return "Waikiki Community Center"

    if (
        "st. peters" in lower
        or "st peters" in lower
        or "st. peter" in lower
        or "st peter" in lower
    ):
        return "St. Peters"

    if "palolo" in lower:
        return "Palolo Hongwanji"

    return None


# =========================================================
# NODE 1 — ANALYZE QUESTION
# =========================================================

def analyze_question(state: RAGState):

    question = state["question"]

    extracted_date = extract_date(
        question
    )

    extracted_location = extract_location(
        question
    )

    if extracted_date:
        route = "metadata"
    else:
        route = "semantic"

    print()
    print("LANGGRAPH NODE: ANALYZE QUESTION")
    print("-" * 70)

    print(f"Route: {route}")
    print(f"Date: {extracted_date}")
    print(f"Location: {extracted_location}")

    return {
        "route": route,
        "extracted_date": extracted_date,
        "extracted_location": extracted_location
    }


# =========================================================
# ROUTING FUNCTION
# =========================================================

def route_question(state: RAGState):

    return state["route"]


# =========================================================
# NODE 2A — SEMANTIC RETRIEVAL
# =========================================================

def semantic_retrieve(state: RAGState):

    print()
    print("LANGGRAPH NODE: SEMANTIC RETRIEVAL")
    print("-" * 70)

    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        state["question"],
        k=4
    )

    print(
        f"Documents retrieved: "
        f"{len(documents)}"
    )

    return {
        "documents": documents
    }


# =========================================================
# NODE 2B — METADATA-AWARE RETRIEVAL
# =========================================================

def metadata_retrieve(state: RAGState):

    print()
    print("LANGGRAPH NODE: METADATA RETRIEVAL")
    print("-" * 70)

    vectorstore = load_vectorstore()

    date_value = state["extracted_date"]
    location = state["extracted_location"]

    if date_value and location:

        metadata_filter = {
            "$and": [
                {
                    "date": date_value
                },
                {
                    "location": location
                }
            ]
        }

    elif date_value:

        metadata_filter = {
            "date": date_value
        }

    else:

        metadata_filter = None

    print(
        f"Metadata filter: "
        f"{metadata_filter}"
    )

    if metadata_filter:

        documents = vectorstore.similarity_search(
            state["question"],
            k=4,
            filter=metadata_filter
        )

    else:

        documents = vectorstore.similarity_search(
            state["question"],
            k=4
        )

    print(
        f"Documents retrieved: "
        f"{len(documents)}"
    )

    return {
        "documents": documents
    }


# =========================================================
# NODE 3 — GENERATE GROUNDED ANSWER
# =========================================================

def generate(state: RAGState):

    print()
    print("LANGGRAPH NODE: GENERATE")
    print("-" * 70)

    question = state["question"]
    documents = state["documents"]

    if not documents:

        return {
            "answer": (
                "I don't have enough information "
                "in the schedule corpus."
            )
        }

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
[Source {index}]

{document.page_content}

Source file:
{document.metadata.get("source", "Unknown")}

Metadata:
{document.metadata}
"""
        )

    context = "\n".join(
        context_parts
    )

    llm = create_llm()

    prompt = f"""
You are a Jazzercise schedule assistant.

Answer the user's question using ONLY the
retrieved context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the context does not contain enough
   information, say exactly:

   I don't have enough information in the schedule corpus.

4. Keep the answer concise.
5. Cite supporting evidence using [Source 1],
   [Source 2], etc.
6. Cite only sources that support the answer.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

def build_graph():

    workflow = StateGraph(
        RAGState
    )

    workflow.add_node(
        "analyze",
        analyze_question
    )

    workflow.add_node(
        "semantic_retrieve",
        semantic_retrieve
    )

    workflow.add_node(
        "metadata_retrieve",
        metadata_retrieve
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.add_edge(
        START,
        "analyze"
    )

    workflow.add_conditional_edges(
        "analyze",
        route_question,
        {
            "semantic": "semantic_retrieve",
            "metadata": "metadata_retrieve"
        }
    )

    workflow.add_edge(
        "semantic_retrieve",
        "generate"
    )

    workflow.add_edge(
        "metadata_retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()


# =========================================================
# RUN ONE QUESTION
# =========================================================

def run_question(
    graph,
    question
):

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    result = graph.invoke(
        {
            "question": question,
            "route": "",
            "extracted_date": None,
            "extracted_location": None,
            "documents": [],
            "answer": ""
        }
    )

    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for index, document in enumerate(
        result["documents"],
        start=1
    ):

        print()
        print(f"[Source {index}]")
        print(document.page_content)

        print(
            "Metadata:",
            document.metadata
        )

    print()
    print("=" * 80)
    print("GROUNDED ANSWER")
    print("=" * 80)

    print(result["answer"])

    return result


# =========================================================
# MAIN — REGRESSION TEST
# =========================================================

def main():

    print()
    print("=" * 80)
    print("JAZZERCISE LANGGRAPH RAG — ROUTED RETRIEVAL")
    print("=" * 80)

    graph = build_graph()

    questions = [
        (
            "Which room does the Waikiki Community "
            "Center class use on the second Tuesday "
            "of the month?"
        ),

        (
            "What time is the St. Peters "
            "Jazzercise class on Monday?"
        ),

        (
            "Is there a Waikiki Community Center "
            "class on January 19, 2026?"
        )
    ]

    for question in questions:

        run_question(
            graph,
            question
        )

    print()
    print("=" * 80)
    print("LANGGRAPH REGRESSION TEST COMPLETE")
    print("=" * 80)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()