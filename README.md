# MARC: Agent-Based Adaptive Routing for Multimodal Fact Verification

Official repository for the paper **"MARC: Agent-Based Adaptive Routing for Multimodal Fact Verification"** (KES 2026).

## Overview

Modern misinformation often seamlessly combines deceptive text with manipulated images, creating complex challenges for digital fact-checking. Current verification systems typically force all claims through a single inflexible pipeline, which either executes costly web retrievals for purely synthetic content or relies solely on internal model memory, causing hallucinations.

We propose **MARC (Multi-Agent Routing for Cross-modal verification)**, an agent-centric framework that transforms fact-checking into a collaborative workflow managed by specialized roles:
- **Orchestrator**: Analyzes the informational intent of a claim and dynamically routes the task.
- **News Researcher**: Gathers external multimodal evidence from web search and reverse image search for verifiable "News Events".
- **Generic Reasoner**: Relies on intuition prediction and deductive rethinking for internal visual inspection of "Generic Scenes".
- **Forensic Analyst**: Independently performs signal-level visual integrity assessment using Dynamic Error-Level Aggregation (DELA) to support the final verdict.

This adaptive delegation prevents unnecessary search noise, optimizing the context utilization of Small Vision-Language Models (SVLMs). MARC enables compact models (like Qwen3-VL 4B) to achieve highly competitive F1-scores, outperforming significantly larger foundation models.

## Repository Structure

- `main.py`: Entry point for evaluating the MARC framework on a dataset.
- `marc.py`: Core implementation of the MARC pipeline, including intent routing, evidence consolidation, and final resolution.
- `image_utils.py`: Implements the Forensic Analyst module, handling image loading, processing, and the Dynamic Error-Level Aggregation (DELA) scanner for texture anomaly detection.
- `prompts.py`: Contains the system prompts and instructions tailored for each specialized agent role within the framework.
- `model_utils.py`: Utilities for initializing and interacting with the Vision-Language Model.
- `io_utils.py`: Helpers for reading input data and appending results to CSV files.

## Installation

Ensure you have Python 3.11 installed. You can install the required dependencies using `pip`. Note that Qwen3-VL requires the latest `transformers` (`>=4.57.0`) and its specific vision utilities:

```bash
pip install torch "transformers>=4.57.0" qwen-vl-utils pandas tqdm opencv-python pillow numpy
```

## Usage

You can run the MARC framework on a dataset formatted as a CSV (e.g., MMFakeBench). The CSV should contain at minimum a `caption` (the text claim) and `image_path` column.

```bash
python main.py \
    --model-id "Qwen/Qwen3-VL-4B-Instruct" \
    --test-csv MMFakeBench_test.csv \
    --test-img MMFakeBench_test_images \
    --output-csv output.csv \
    --dtype float16
```

### Command-Line Arguments

- `--model-id`: Hugging Face model identifier for the SVLM (default: `Qwen/Qwen3-VL-4B-Instruct`).
- `--test-csv`: Path to the input CSV containing claims.
- `--test-img`: Directory containing the images referenced in the CSV.
- `--output-csv`: Path to save the detailed evaluation and verification results.
- `--dtype`: Precision for model inference (`float16`, `bfloat16`, `float32`).
- `--device-map`: Device mapping strategy for Transformers (default: `auto`).
- `--crawled-json`: (Optional) Pre-crawled text search documents for News Events to save retrieval time.
- `--crawled-image-json`: (Optional) Pre-crawled reverse image search documents.

## Citation

If you find this repository or our paper useful in your research, please consider citing:

```bibtex
@inproceedings{trieu2026marc,
  title={MARC: Agent-Based Adaptive Routing for Multimodal Fact Verification},
  author={Trieu, Huy and Nguyen-Tran, Duy-Minh and Nguyen, Huy Tien and Le, Tung},
  booktitle={30th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES 2026)},
  year={2026},
  note={To appear}
}
```