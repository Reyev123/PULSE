"""Lazy PULSE model loader and single-image inference for the GUI.

The model (~14 GB) is loaded once on first use and cached for the process, so
each GUI inference click reuses the same weights instead of reloading.
"""

import os
import threading

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

DEFAULT_MODEL = os.environ.get("PULSE_MODEL_PATH", "PULSE-ECG/PULSE-7B")
CONV_MODE = "llava_v1"

_MODEL = None
_LOCK = threading.Lock()


def get_model(model_path=DEFAULT_MODEL):
    """Load and cache (tokenizer, model, image_processor, name)."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            disable_torch_init()
            name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path, None, name)
            _MODEL = (tokenizer, model, image_processor, name)
    return _MODEL


def run_inference(image, prompt, model_path=DEFAULT_MODEL, max_new_tokens=1024):
    """Run PULSE on a PIL image and return the generated report text."""
    tokenizer, model, image_processor, _ = get_model(model_path)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
