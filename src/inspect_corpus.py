from load_documents import load_all_documents


def inspect_documents():
    documents = load_all_documents()

    print("\n" + "=" * 70)
    print("CORPUS INSPECTION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Waikiki room / exception rules
    # --------------------------------------------------
    print("\nWAIKIKI ROOM / EXCEPTION RULES")
    print("-" * 70)

    for doc in documents:
        if doc.metadata.get("document_type") == "schedule_rule":
            print(doc.page_content)
            print("Metadata:")
            print(doc.metadata)
            print()

    # --------------------------------------------------
    # 2. Monday sample
    # --------------------------------------------------
    print("\nMONDAY SAMPLE — JANUARY 12")
    print("-" * 70)

    for doc in documents:
        if doc.metadata.get("date") == "2026-01-12":
            print(doc.page_content)
            print(doc.metadata)
            print()

    # --------------------------------------------------
    # 3. Second Tuesday calendar sample
    # --------------------------------------------------
    print("\nSECOND TUESDAY SAMPLE — JANUARY 13")
    print("-" * 70)

    for doc in documents:
        if doc.metadata.get("date") == "2026-01-13":
            print(doc.page_content)
            print(doc.metadata)
            print()

    # --------------------------------------------------
    # 4. Validate second Tuesday room rule
    # --------------------------------------------------
    print("\nSECOND TUESDAY ROOM RULE")
    print("-" * 70)

    second_tuesday_rule_found = False

    for doc in documents:
        if (
            doc.metadata.get("document_type") == "schedule_rule"
            and doc.metadata.get("rule_name") == "Tuesday"
        ):
            print(doc.page_content)
            print("Metadata:")
            print(doc.metadata)
            print()

            content = doc.page_content.lower()

            if (
                "second tuesday" in content
                and "room 205e" in content
            ):
                second_tuesday_rule_found = True

    print(
        "Second Tuesday exception found:",
        second_tuesday_rule_found
    )

    # --------------------------------------------------
    # 5. Cancellation records
    # --------------------------------------------------
    print("\nCANCELLATION RECORDS")
    print("-" * 70)

    cancellation_records = [
        doc
        for doc in documents
        if doc.metadata.get("status") == "NO CLASS"
    ]

    for doc in cancellation_records:
        print(doc.page_content)
        print(doc.metadata)
        print()

    print(
        f"Cancellation records found: "
        f"{len(cancellation_records)}"
    )

    # --------------------------------------------------
    # 6. Unspecified location records
    # --------------------------------------------------
    print("\nUNSPECIFIED LOCATION RECORDS")
    print("-" * 70)

    unspecified_records = [
        doc
        for doc in documents
        if doc.metadata.get("location") == "Unspecified"
    ]

    for doc in unspecified_records:
        print(doc.page_content)
        print(doc.metadata)
        print()

    print(
        f"Unspecified-location records found: "
        f"{len(unspecified_records)}"
    )

    # --------------------------------------------------
    # 7. Duplicate check
    # --------------------------------------------------
    print("\nDUPLICATE CHECK")
    print("-" * 70)

    seen = set()
    duplicates = []

    for doc in documents:

        # Skip business-rule documents
        if doc.metadata.get("document_type") == "schedule_rule":
            continue

        location = doc.metadata.get("location")

        # Structured schedule record
        if location != "Unspecified":
            key = (
                doc.metadata.get("date"),
                location,
                doc.metadata.get("time"),
                doc.metadata.get("status"),
            )

        # Unspecified/general notes need their original
        # source text and Excel cell to distinguish them.
        else:
            key = (
                doc.metadata.get("date"),
                location,
                doc.metadata.get("raw_entry"),
                doc.metadata.get("cell"),
            )

        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)

    print(f"Duplicate schedule records: {len(duplicates)}")

    if duplicates:
        print("\nDuplicate details:")

        for duplicate in duplicates[:10]:
            print(duplicate)
    else:
        print("PASS - No duplicate schedule records found.")
    # --------------------------------------------------
    # 8. Metadata completeness
    # --------------------------------------------------
    print("\nMETADATA COMPLETENESS CHECK")
    print("-" * 70)

    required_calendar_fields = [
        "source",
        "sheet",
        "date",
        "day_of_week",
        "month",
        "year",
        "location",
        "status",
        "cell",
    ]

    required_rule_fields = [
        "source",
        "document_type",
        "location",
        "rule_name",
    ]

    missing_metadata = []

    for index, doc in enumerate(
        documents,
        start=1
    ):

        if (
            doc.metadata.get("document_type")
            == "schedule_rule"
        ):
            required_fields = required_rule_fields
        else:
            required_fields = required_calendar_fields

        missing_fields = []

        for field in required_fields:

            if (
                field not in doc.metadata
                or doc.metadata.get(field) is None
                or doc.metadata.get(field) == ""
            ):
                missing_fields.append(field)

        if missing_fields:

            missing_metadata.append(
                {
                    "document_number": index,
                    "missing_fields": missing_fields,
                    "content": doc.page_content,
                }
            )

    print(
        "Documents with missing required metadata:",
        len(missing_metadata)
    )

    if missing_metadata:

        print("\nFirst 10 metadata issues:")

        for issue in missing_metadata[:10]:
            print(issue)

    else:
        print(
            "PASS - Required metadata is complete."
        )

    # --------------------------------------------------
    # 9. Corpus summary
    # --------------------------------------------------
    calendar_count = 0
    rules_count = 0

    for doc in documents:

        if (
            doc.metadata.get("document_type")
            == "schedule_rule"
        ):
            rules_count += 1
        else:
            calendar_count += 1

    print("\nCORPUS SUMMARY")
    print("-" * 70)

    print(
        f"Calendar documents: "
        f"{calendar_count}"
    )

    print(
        f"Schedule rule documents: "
        f"{rules_count}"
    )

    print(
        f"Unspecified-location documents: "
        f"{len(unspecified_records)}"
    )

    print(
        f"Total documents: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    # 10. Final validation
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("PRE-CHROMA VALIDATION")
    print("=" * 70)

    checks = {
        "Second Tuesday rule": second_tuesday_rule_found,
        "No duplicates": len(duplicates) == 0,
        "Metadata complete": len(missing_metadata) == 0,
    }

    for check_name, passed in checks.items():

        status = "PASS" if passed else "REVIEW"

        print(
            f"{status} - {check_name}"
        )

    if all(checks.values()):

        print()
        print(
            "PASS - Corpus is ready "
            "for Chroma vector-store testing."
        )

    else:

        print()
        print(
            "REVIEW NEEDED - Fix the items above "
            "before loading into Chroma."
        )


if __name__ == "__main__":
    inspect_documents()