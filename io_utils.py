import csv


def write_csv_headers(output_csv):
    headers = [
        "caption",
        "image_path",
        "label",
        "category",
        "step1_intuition",
        "semantic_report",
        "phys_report",
        "step2_conflict_report",
        "step3_strategy_raw",
        "step4_evidence",
        "step4_image_evidence",
        "reasoning",
        "prediction",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_result_row(output_csv, result):
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                result["caption"],
                result["image_path"],
                result["label"],
                result.get("category", "N/A"),
                result.get("step1_intuition", "N/A"),
                result.get("semantic_report", "N/A"),
                result.get("phys_report", "N/A"),
                result.get("step2_conflict_report", "N/A"),
                result.get("step3_strategy_raw", "N/A"),
                result.get("step4_evidence", "N/A"),
                result.get("step4_image_evidence", "N/A"),
                result.get("reasoning", "N/A"),
                result.get("prediction", "N/A"),
            ]
        )


def append_error_row(output_csv, row):
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["caption"],
                row["image_path"],
                row.get("label", None),
                row.get("category", "N/A"),
                row.get("step1_intuition", "N/A"),
                row.get("semantic_report", "N/A"),
                row.get("phys_report", "N/A"),
                row.get("step2_conflict_report", "N/A"),
                row.get("step3_strategy_raw", "N/A"),
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
            ]
        )
