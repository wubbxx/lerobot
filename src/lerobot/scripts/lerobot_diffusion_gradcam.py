#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.gradcam_visualization import (
    export_inference_gradcam,
    export_train_gradcam,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import init_logging


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Diffusion Grad-CAM visualizations to local files.")
    parser.add_argument(
        "--enable-gradcam",
        action="store_true",
        default=False,
        help="Enable Grad-CAM export. Disabled by default to avoid affecting normal workflows.",
    )
    parser.add_argument(
        "--train-config-path",
        type=Path,
        required=True,
        help="Path to train_config.json used to construct dataset and policy metadata.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Path to pretrained_model directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gradcam"),
        help="Directory to save Grad-CAM images.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index for visualization.",
    )
    parser.add_argument(
        "--action-step",
        type=int,
        default=0,
        help="Action step index for inference Grad-CAM target.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=0,
        help="Action dimension index for inference Grad-CAM target.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay blending strength in [0, 1].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override policy device. If omitted, uses the config device.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    init_logging()

    if not args.enable_gradcam:
        logging.info("Grad-CAM export is disabled. Pass --enable-gradcam to run visualization.")
        return

    train_config_path = args.train_config_path
    if not train_config_path.exists():
        fallback = args.policy_path / "train_config.json"
        if fallback.exists():
            logging.warning(
                "train-config-path not found: %s. Falling back to %s",
                train_config_path,
                fallback,
            )
            train_config_path = fallback
        else:
            raise FileNotFoundError(
                f"train-config-path not found: {train_config_path}. "
                f"Also checked fallback: {fallback}"
            )

    cfg = TrainPipelineConfig.from_pretrained(train_config_path)

    if cfg.policy is None:
        raise ValueError("No policy configuration found in train config.")

    cfg.policy.pretrained_path = args.policy_path
    if args.device is not None:
        cfg.policy.device = args.device

    if cfg.policy.type != "diffusion":
        raise ValueError(f"This script supports diffusion policy only. Got policy.type={cfg.policy.type}.")

    logging.info("Loading dataset")
    dataset = make_dataset(cfg)

    logging.info("Loading policy from checkpoint")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs = {
        "dataset_stats": dataset.meta.stats,
        "preprocessor_overrides": {
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} is out of bounds for dataset size {len(dataset)}.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    raw_batch = None
    for idx, item in enumerate(loader):
        if idx == args.sample_index:
            raw_batch = item
            break

    if raw_batch is None:
        raise RuntimeError("Failed to load the requested dataset sample.")

    output_dir = args.output_dir / f"sample_{args.sample_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.enable_grad():
        train_score = export_train_gradcam(
            policy=policy,
            preprocessor=preprocessor,
            raw_batch=raw_batch,
            output_dir=output_dir,
            alpha=args.alpha,
        )
        inference_score = export_inference_gradcam(
            policy=policy,
            preprocessor=preprocessor,
            raw_batch=raw_batch,
            output_dir=output_dir,
            action_step=args.action_step,
            action_dim=args.action_dim,
            alpha=args.alpha,
        )

    summary_path = output_dir / "summary.txt"
    summary_lines = [
        f"policy_path={args.policy_path}",
        f"train_config_path={args.train_config_path}",
        f"sample_index={args.sample_index}",
        f"train_loss_target={train_score}",
        f"inference_action_target={inference_score}",
        f"action_step={args.action_step}",
        f"action_dim={args.action_dim}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    logging.info("Grad-CAM export finished. Results saved under: %s", output_dir)


if __name__ == "__main__":
    main()
