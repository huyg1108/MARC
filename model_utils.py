import gc
import os
import random

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_torch_dtype(dtype_name):
    if not dtype_name:
        return None
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_name)


def init_model(model_id, dtype_name="float16", device_map="auto", attn_implementation="sdpa"):
    dtype = parse_torch_dtype(dtype_name)
    loaded_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, attn_implementation=attn_implementation
    )
    loaded_processor = AutoProcessor.from_pretrained(model_id)
    return loaded_model, loaded_processor


def run_model_with_content(model, processor, content, max_new_tokens=128):
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    torch.cuda.empty_cache()
    gc.collect()

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
