#!/usr/bin/env python3
"""Capture an initial LIBERO-Spatial observation and its task instruction."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("samples/libero"))
    args = parser.parse_args()

    import torch
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f"task-id must be between 0 and {suite.n_tasks - 1}")
    task = suite.get_task(args.task_id)
    state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # LIBERO's distributed initial states may contain NumPy objects. Only load
    # this trusted file from the installed LIBERO checkout, never arbitrary input.
    states = torch.load(state_path, map_location="cpu", weights_only=False)
    if not 0 <= args.state_id < len(states):
        parser.error(f"state-id must be between 0 and {len(states) - 1}")

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(states[args.state_id])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        # Match the camera orientation used by the official OpenVLA evaluation.
        image = Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "scene.png"
    instruction_path = args.output_dir / "instruction.txt"
    metadata_path = args.output_dir / "metadata.json"
    image.save(image_path)
    instruction_path.write_text(task.language + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"suite": "libero_spatial", "task_id": args.task_id,
        "state_id": args.state_id, "task_name": task.name, "instruction": task.language,
        "image": image_path.name}, indent=2) + "\n", encoding="utf-8")
    print(f"Image: {image_path}\nInstruction: {task.language}\nInstruction file: {instruction_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
