import argparse
import gc
import json

import pandas as pd
import torch
from tqdm import tqdm

from io_utils import append_error_row, append_result_row, write_csv_headers
from marc import process_fact_verification_item
from model_utils import init_model, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="MARC fact verification runner")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-csv", default="MMFakeBench_test.csv")
    parser.add_argument("--test-img", default="MMFakeBench_test")
    parser.add_argument("--output-csv", default="output.csv")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--crawled-json", default=None)
    parser.add_argument("--crawled-image-json", default=None)
    return parser.parse_args()


def load_json_if_exists(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_crawled_items(crawled_data, idx):
    if isinstance(crawled_data, list) and idx < len(crawled_data):
        return crawled_data[idx].get("crawled", [])
    if isinstance(crawled_data, dict):
        item = crawled_data.get(str(idx), crawled_data.get(idx))
        if isinstance(item, dict):
            return item.get("crawled", [])
    return []


def main():
    args = parse_args()

    seed_everything(args.seed)
    try:
        from settings import SETTINGS

        SETTINGS.update({"seed": args.seed})
    except Exception:
        pass

    model, processor = init_model(
        args.model_id,
        dtype_name=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
    )

    test_df = pd.read_csv(args.test_csv)
    write_csv_headers(args.output_csv)

    # Load crawled data from internet if provided
    crawled_data = load_json_if_exists(args.crawled_json) or []
    crawled_image_indexed_data = load_json_if_exists(args.crawled_image_json) or {}

    print(f"Processing {len(test_df)} claims for fact verification with MARC...")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Verifying claims"):
        try:
            crawled_items = get_crawled_items(crawled_data, idx)
            crawled_image_items = (
                crawled_image_indexed_data.get(str(idx), {}).get("crawled", [])
                if isinstance(crawled_image_indexed_data, dict)
                else []
            )

            result = process_fact_verification_item(
                row, args.test_img, crawled_items, crawled_image_items, model, processor
            )
            append_result_row(args.output_csv, result)

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing claim {idx}: {str(e)}")
            append_error_row(args.output_csv, row)

    print(f"\nCompleted! Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
