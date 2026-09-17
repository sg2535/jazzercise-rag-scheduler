import json
from pathlib import Path

from rag_graph import build_graph


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "rag_eval_dataset.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "rag_eval_results.json"
)


# =========================================================
# LOAD EVALUATION DATASET
# =========================================================

def load_eval_dataset():

    with open(
        EVAL_DATASET,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results):

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


# =========================================================
# RUN EVALUATION
# =========================================================

def main():

    print()
    print("=" * 80)
    print("JAZZERCISE RAG EVALUATION")
    print("=" * 80)

    dataset = load_eval_dataset()

    print()
    print(
        f"Evaluation cases loaded: "
        f"{len(dataset)}"
    )

    print()
    print("Building LangGraph...")

    graph = build_graph()

    results = []

    passed = 0

    # -----------------------------------------------------
    # RUN EACH TEST CASE
    # -----------------------------------------------------

    for test in dataset:

        test_id = test["id"]
        category = test["category"]
        question = test["question"]

        expected = test[
            "expected_answer_contains"
        ]

        print()
        print("=" * 80)
        print(
            f"{test_id} | {category}"
        )
        print("=" * 80)

        print()
        print("QUESTION:")
        print(question)

        # -----------------------------------------------
        # Run question through LangGraph
        # -----------------------------------------------

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

        answer = result["answer"]

        documents = result["documents"]

        # -----------------------------------------------
        # Simple deterministic evaluation
        # -----------------------------------------------

        is_pass = (
            expected.lower()
            in answer.lower()
        )

        status = (
            "PASS"
            if is_pass
            else "FAIL"
        )

        if is_pass:
            passed += 1

        print()
        print("EXPECTED:")
        print(expected)

        print()
        print("GENERATED ANSWER:")
        print(answer)

        print()
        print(
            f"RESULT: {status}"
        )

        # -----------------------------------------------
        # Capture retrieved source information
        # -----------------------------------------------

        retrieved_sources = []

        for rank, document in enumerate(
            documents,
            start=1
        ):

            retrieved_sources.append(
                {
                    "rank": rank,
                    "page_content":
                        document.page_content,
                    "metadata":
                        document.metadata
                }
            )

        # -----------------------------------------------
        # Store result
        # -----------------------------------------------

        results.append(
            {
                "id": test_id,
                "category": category,
                "question": question,
                "expected_answer_contains":
                    expected,
                "generated_answer":
                    answer,
                "result":
                    status,
                "route":
                    result.get("route"),
                "extracted_date":
                    result.get(
                        "extracted_date"
                    ),
                "extracted_location":
                    result.get(
                        "extracted_location"
                    ),
                "retrieved_sources":
                    retrieved_sources
            }
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    total = len(dataset)

    failed = total - passed

    accuracy = (
        passed / total * 100
        if total
        else 0
    )

    print()
    print("=" * 80)
    print("RAG EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Passed: {passed}/{total}"
    )

    print(
        f"Failed: {failed}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.1f}%"
    )

    # -----------------------------------------------------
    # CATEGORY SUMMARY
    # -----------------------------------------------------

    print()
    print("RESULTS BY CATEGORY")
    print("-" * 80)

    categories = {}

    for result in results:

        category = result["category"]

        if category not in categories:

            categories[category] = {
                "passed": 0,
                "total": 0
            }

        categories[category]["total"] += 1

        if result["result"] == "PASS":
            categories[category][
                "passed"
            ] += 1

    for category, values in categories.items():

        category_accuracy = (
            values["passed"]
            / values["total"]
            * 100
        )

        print(
            f"{category}: "
            f"{values['passed']}/"
            f"{values['total']} "
            f"({category_accuracy:.1f}%)"
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    save_results(
        results
    )

    print()
    print(
        f"Detailed results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
