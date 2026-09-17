import json
from pathlib import Path

from langchain_core.documents import Document


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jazzercise_corpus.json"
)

RULES_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "waikiki_schedule_rules.md"
)

LANGCHAIN_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "langchain_documents.json"
)


# ---------------------------------------------------------
# LOAD HISTORICAL CALENDAR DOCUMENTS
# ---------------------------------------------------------

def load_calendar_documents():
    """
    Load normalized historical Jazzercise schedule records
    from jazzercise_corpus.json and convert them into
    LangChain Document objects.
    """

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    documents = []

    for record in records:
        document = Document(
            page_content=record["page_content"],
            metadata=record["metadata"]
        )

        documents.append(document)

    return documents


# ---------------------------------------------------------
# LOAD AND CHUNK WAIKIKI SCHEDULE RULES
# ---------------------------------------------------------

def load_rule_documents():
    """
    Load Waikiki room and scheduling rules.

    Each Markdown section becomes its own LangChain Document
    so that rules such as the second-Tuesday exception can
    be retrieved independently.
    """

    with open(
        RULES_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        rules_text = file.read()

    sections = rules_text.split("## ")

    documents = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # Skip the main document title.
        if section.startswith("# Waikiki Community Center"):
            continue

        lines = section.splitlines()

        heading = lines[0].strip()

        content = "\n".join(
            lines[1:]
        ).strip()

        if not content:
            continue

        page_content = (
            "Waikiki Community Center scheduling rule.\n"
            f"Rule: {heading}\n"
            f"{content}"
        )

        document = Document(
            page_content=page_content,
            metadata={
                "source": RULES_FILE.name,
                "document_type": "schedule_rule",
                "location": "Waikiki Community Center",
                "rule_name": heading
            }
        )

        documents.append(document)

    return documents


# ---------------------------------------------------------
# SAVE FINAL LANGCHAIN DOCUMENTS TO JSON
# ---------------------------------------------------------

def save_documents_to_json(documents):
    """
    Export the final LangChain Documents to JSON.

    This makes the final chunked corpus easy to inspect
    before embeddings and vector-store creation.
    """

    output_records = []

    for doc in documents:
        output_records.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        )

    LANGCHAIN_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        LANGCHAIN_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"LangChain documents saved to: "
        f"{LANGCHAIN_OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# LOAD COMPLETE RAG CORPUS
# ---------------------------------------------------------

def load_all_documents():
    """
    Combine historical calendar chunks and scheduling-rule
    chunks into the complete LangChain corpus.
    """

    calendar_documents = load_calendar_documents()
    rule_documents = load_rule_documents()

    all_documents = (
        calendar_documents
        + rule_documents
    )

    # Save a human-readable copy of the final chunks.
    save_documents_to_json(
        all_documents
    )

    print()
    print("=" * 70)
    print("LANGCHAIN DOCUMENT LOAD COMPLETE")
    print("=" * 70)

    print(
        f"Calendar documents: "
        f"{len(calendar_documents)}"
    )

    print(
        f"Rule documents: "
        f"{len(rule_documents)}"
    )

    print(
        f"Total documents: "
        f"{len(all_documents)}"
    )

    return all_documents


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    documents = load_all_documents()

    print()
    print("EXAMPLE CALENDAR DOCUMENT")
    print("-" * 70)

    if documents:
        print(documents[0])

    print()
    print("RULE DOCUMENTS")
    print("-" * 70)

    for doc in documents:

        if (
            doc.metadata.get("document_type")
            == "schedule_rule"
        ):
            print(doc.page_content)
            print(doc.metadata)
            print()