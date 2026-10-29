#!/usr/bin/env python3
"""Run one OpenVLA action prediction on a supplied image."""
import argparse
import json
import math
from pathlib import Path


MODELS = {
    "base": ("openvla/openvla-7b", "bridge_orig"),
    "spatial": ("openvla/openvla-7b-finetuned-libero-spatial", "libero_spatial"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True, help="Robot instruction in English")
    parser.add_argument("--unnorm-key", help="Override the checkpoint's default action statistics key")
    parser.add_argument("--center-crop", action="store_true", help="Approximate a central 90%% area crop")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and PyTorch CUDA installation.")

    model_id, default_key = MODELS[args.model]
    key = args.unnorm_key or default_key
    image = Image.open(args.image).convert("RGB")
    if args.center_crop:
        width, height = image.size
        crop_w, crop_h = width * math.sqrt(0.9), height * math.sqrt(0.9)
        image = image.crop(((width - crop_w) / 2, (height - crop_h) / 2,
                            (width + crop_w) / 2, (height + crop_h) / 2))
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    if key not in model.norm_stats:
        raise SystemExit(f"Unknown unnorm_key {key!r}; available: {list(model.norm_stats)}")
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
    print(json.dumps({"model": model_id, "unnorm_key": key, "instruction": args.instruction,
                      "action": action.tolist()}, indent=2))


if __name__ == "__main__":
    main()
