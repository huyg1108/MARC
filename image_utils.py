import os

import cv2
import numpy as np
from PIL import Image

from model_utils import run_model_with_content
from prompts import (
    build_conflict_resolution_prompt,
    build_physical_prompt,
    build_semantic_prompt,
)


def normalize_child_path(child_path):
    return child_path.lstrip("/\\")


def load_image_w_resize(child_path, image_dir, max_side=480):
    img_path = os.path.join(image_dir, normalize_child_path(child_path))
    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")
        if max_side and (image.width > max_side or image.height > max_side):
            image.thumbnail((max_side, max_side), Image.LANCZOS)
    else:
        print(f"Warning: Image not found: {img_path}")
        image = Image.new("RGB", (50, 50), "white")
    return image


def load_image(child_path, image_dir):
    img_path = os.path.join(image_dir, normalize_child_path(child_path))
    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")
    else:
        print(f"Warning: Image not found: {img_path}")
        image = Image.new("RGB", (50, 50), "white")
    return image


def analyze_local_texture_consistency(image_rgb_pil, sensitivity=0.9):
    image_cv = cv2.cvtColor(np.array(image_rgb_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape

    temp_file = "temp_ela_check.jpg"
    cv2.imwrite(temp_file, image_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
    compressed_image = cv2.imread(temp_file)
    if os.path.exists(temp_file):
        os.remove(temp_file)

    diff = cv2.absdiff(image_cv, compressed_image)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    box_size = max(min(h, w) // 10, 20)
    block_scores = []
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            roi = diff_gray[y : y + box_size, x : x + box_size]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
            score = np.mean(roi)
            block_scores.append(score)

    if not block_scores:
        return "No data", image_rgb_pil

    target_percentile = sensitivity * 100
    threshold = np.percentile(block_scores, target_percentile)

    output_image = image_cv.copy()
    suspicious_count = 0
    suspicious_zones = []
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            roi = diff_gray[y : y + box_size, x : x + box_size]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
            score = np.mean(roi)
            if score >= threshold:
                cv2.rectangle(
                    output_image,
                    (x, y),
                    (x + box_size, y + box_size),
                    (0, 0, 255),
                    2,
                )
                overlay = output_image[y : y + box_size, x : x + box_size]
                red_block = np.full(overlay.shape, (0, 0, 255), dtype=np.uint8)
                cv2.addWeighted(overlay, 0.8, red_block, 0.2, 0, overlay)
                output_image[y : y + box_size, x : x + box_size] = overlay
                suspicious_count += 1
                if suspicious_count <= 3:
                    suspicious_zones.append(f"Box(x={x}, y={y})")

    debug_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    debug_image_pil = Image.fromarray(debug_image_rgb)

    if suspicious_count > 0:
        report = (
            f"FORENSIC SCAN: Flagged {suspicious_count} regions "
            f"(Top {100 - target_percentile:.0f}% highest artifact energy).\n"
            f"Potential AI artifacts/Edits at: {', '.join(suspicious_zones)}..."
        )
    else:
        report = "FORENSIC SCAN: Image too clean, no distinct artifacts found."

    return report, debug_image_pil


def combine_image(image_1, image_2, max_side=960):
    combined = Image.new("RGB", (image_1.width, image_1.height * 2))
    combined.paste(image_1, (0, 0))
    combined.paste(image_2, (0, image_1.height))

    if max_side and (combined.width > max_side or combined.height > max_side):
        combined.thumbnail((max_side, max_side), Image.LANCZOS)

    return combined


def image_check(claim, original_image, image, model, processor):
    texture_report, texture_debug_img = analyze_local_texture_consistency(original_image)
    combined_image = combine_image(original_image, texture_debug_img)

    semantic_prompt_text = build_semantic_prompt(forensic_report=texture_report)
    semantic_content = [
        {"type": "image", "image": combined_image},
        {"type": "text", "text": semantic_prompt_text},
    ]
    semantic_report = run_model_with_content(
        model, processor, semantic_content, max_new_tokens=512
    )

    phys_prompt_text = build_physical_prompt(claim)
    phys_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": phys_prompt_text},
    ]
    phys_report = run_model_with_content(model, processor, phys_content, max_new_tokens=512)

    conflict_resol_prompt = build_conflict_resolution_prompt(
        claim, semantic_report, phys_report
    )
    conflict_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": conflict_resol_prompt},
    ]
    conflict_report = run_model_with_content(
        model, processor, conflict_content, max_new_tokens=256
    )

    return semantic_report, phys_report, conflict_report
