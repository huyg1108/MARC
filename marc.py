from image_utils import image_check, load_image, load_image_w_resize
from model_utils import run_model_with_content
from prompts import (
    build_evidence_consolidation_prompt,
    build_evidence_filtering_prompt,
    build_final_pred,
    build_generic_analysis_prompt,
    build_image_context_consolidation_prompt,
    build_image_evidence_filtering_prompt,
    build_news_resolution_prompt,
    build_strategy_prompt,
    build_zero_shot_prompt,
)


def split_text_into_chunks(text, max_chars=1200, overlap=200):
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def process_fact_verification_item(
    row, image_dir, crawled_items, crawled_image_items, model, processor
):
    claim = row["caption"]
    image_path = row["image_path"]

    image = load_image_w_resize(image_path, image_dir)
    original_image = load_image(image_path, image_dir)

    intuition_prompt = build_zero_shot_prompt(claim)
    image_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": intuition_prompt},
    ]

    prior_prediction = run_model_with_content(
        model, processor, image_content, max_new_tokens=512
    )

    semantic_report, phys_report, conflict_report = image_check(
        claim, original_image, image, model, processor
    )

    strategy_prompt = build_strategy_prompt(claim)
    strategy_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": strategy_prompt},
    ]
    strategy_response = run_model_with_content(
        model, processor, strategy_content, max_new_tokens=128
    )
    category = (
        "Generic Scene"
        if ("Generic Scene" in strategy_response or "Internal Analysis" in strategy_response)
        else "News Event"
    )

    evidence_log = ""

    if category == "News Event":
        summarized_evidences = []
        summarized_image_evidences = []

        current_items = crawled_items[:10] if isinstance(crawled_items, list) else []
        current_image_items = (
            crawled_image_items[:5] if isinstance(crawled_image_items, list) else []
        )

        for item in current_items:
            raw_text = item.get("text", "").strip()
            if not raw_text:
                continue

            raw_text_chunks = split_text_into_chunks(
                raw_text, max_chars=1200, overlap=200
            )
            for chunk in raw_text_chunks[:10]:
                raw_text_chunk = f"{chunk}\n"

                filter_prompt = build_evidence_filtering_prompt(claim, raw_text_chunk)
                filter_content = [{"type": "text", "text": filter_prompt}]

                summary = run_model_with_content(
                    model, processor, filter_content, max_new_tokens=256
                )
                if "irrelevant" not in summary.lower():
                    summarized_evidences.append(summary)

        for item in current_image_items:
            raw_text = item.get("text", "").strip()
            if not raw_text:
                continue

            raw_text_chunks = split_text_into_chunks(
                raw_text, max_chars=1200, overlap=200
            )

            for chunk in raw_text_chunks[:10]:
                raw_text_chunk = f"{chunk}\n"

                filter_prompt = build_image_evidence_filtering_prompt(
                    claim, raw_text_chunk
                )
                filter_content = [{"type": "text", "text": filter_prompt}]

                summary = run_model_with_content(
                    model, processor, filter_content, max_new_tokens=256
                )
                if "irrelevant" not in summary.lower():
                    summarized_image_evidences.append(summary)

        if summarized_evidences:
            raw_evidence_text = "\n".join(
                [f"[Evidence {i+1}] {s}" for i, s in enumerate(summarized_evidences)]
            )
            consolidation_prompt = build_evidence_consolidation_prompt(
                claim, raw_evidence_text
            )
            consolidation_content = [{"type": "text", "text": consolidation_prompt}]
            clean_evidence = run_model_with_content(
                model, processor, consolidation_content, max_new_tokens=256
            ).strip()
            evidence_text = clean_evidence
        else:
            evidence_text = "No relevant evidence found."

        evidence_log = f"Consolidated Evidence:\n{evidence_text}"

        if summarized_image_evidences:
            raw_image_evidence_text = "\n".join(
                [
                    f"[Evidence {i+1}] {s}"
                    for i, s in enumerate(summarized_image_evidences)
                ]
            )
            image_consolidation_prompt = build_image_context_consolidation_prompt(
                raw_image_evidence_text
            )
            image_consolidation_content = [
                {"type": "text", "text": "Give claim image: "},
                {"type": "image", "image": image},
                {"type": "text", "text": image_consolidation_prompt},
            ]
            clean_image_evidence = run_model_with_content(
                model, processor, image_consolidation_content, max_new_tokens=256
            ).strip()
            image_evidence_text = clean_image_evidence
        else:
            image_evidence_text = "No relevant evidence found."

        evidence_image_log = f"Consolidated Evidence:\n{image_evidence_text}"

        resolution_prompt = build_news_resolution_prompt(
            claim, conflict_report, evidence_text, image_evidence_text
        )
        resolution_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": resolution_prompt},
        ]
        final_reasoning = run_model_with_content(
            model, processor, resolution_content, max_new_tokens=256
        )
    else:
        evidence_log = "Skipped Search (Generic)."
        evidence_image_log = "Skipped Search (Generic)."

        resolution_prompt = build_generic_analysis_prompt(
            claim, prior_prediction, conflict_report
        )
        resolution_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": resolution_prompt},
        ]
        final_reasoning = run_model_with_content(
            model, processor, resolution_content, max_new_tokens=256
        )

    final_pred_prompt = build_final_pred(final_reasoning)
    final_pred_content = [{"type": "text", "text": final_pred_prompt}]

    final_label = run_model_with_content(
        model, processor, final_pred_content, max_new_tokens=10
    )

    return {
        "caption": claim,
        "image_path": image_path,
        "label": row.get("label", None),
        "step1_intuition": prior_prediction,
        "semantic_report": semantic_report,
        "phys_report": phys_report,
        "step2_conflict_report": conflict_report,
        "category": category,
        "step3_strategy_raw": strategy_response,
        "step4_evidence": evidence_log,
        "step4_image_evidence": evidence_image_log,
        "reasoning": final_reasoning,
        "prediction": final_label,
    }
