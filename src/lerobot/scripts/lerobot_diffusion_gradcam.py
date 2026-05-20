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
from torch.utils.data import DataLoader, default_collate
from tqdm import trange

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.gradcam_visualization import (
    compute_inference_gradcam_tensors,
    export_inference_gradcam,
    export_train_gradcam,
    render_gradcam_frame,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.io_utils import write_video
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
        default=0.35,
        help="Overlay blending strength in [0, 1].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override policy device. If omitted, uses the config device.",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        default=False,
        help="Also export a video showing Grad-CAM dynamics across consecutive dataset samples.",
    )
    parser.add_argument(
        "--video-num-samples",
        type=int,
        default=120,
        help="Number of consecutive dataset samples to render into the Grad-CAM video.",
    )
    parser.add_argument(
        "--video-full-episode",
        action="store_true",
        default=False,
        help="Render the full episode containing sample-index. Overrides video-num-samples.",
    )
    parser.add_argument(
        "--video-episode-index",
        type=int,
        default=None,
        help="Render this exact episode index. Implies video-full-episode and overrides sample-index for video.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="FPS of the exported Grad-CAM video.",
    )
    parser.add_argument(
        "--video-view",
        type=str,
        default="overlay",
        choices=["overlay", "heat", "cam", "raw"],
        help="Which visualization to encode in the video.",
    )
    parser.add_argument(
        "--video-obs-step",
        type=int,
        default=-1,
        help="Observation step to render in the video. -1 means the latest/current observation.",
    )
    parser.add_argument(
        "--video-camera-index",
        type=int,
        default=0,
        help="Camera index to render in the video.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Optional output video filename. Defaults to gradcam_<view>_sample_<start>_<end>.mp4.",
    )
    return parser


def _export_inference_gradcam_video(
    policy,
    preprocessor,
    dataset,
    output_dir: Path,
    start_index: int,
    num_samples: int,
    fps: int,
    view: str,
    obs_step: int,
    camera_index: int,
    action_step: int,
    action_dim: int,
    alpha: float,
    video_name: str | None,
) -> Path:
    if num_samples <= 0:
        raise ValueError("video-num-samples must be positive.")
    end_index = min(start_index + num_samples, len(dataset))
    if start_index >= end_index:
        raise ValueError("Video sample range is empty.")

    dataset_fps = getattr(dataset, "fps", fps)
    sample_stride = max(1, round(dataset_fps / fps))
    sample_indices = list(range(start_index, end_index, sample_stride))

    output_dir.mkdir(parents=True, exist_ok=True)
    if video_name is None:
        video_name = (
            f"gradcam_{view}_sample_{start_index:06d}_{end_index - 1:06d}"
            f"_{fps}fps_stride{sample_stride}.mp4"
        )
    video_path = output_dir / video_name

    logging.info(
        "Rendering Grad-CAM video from dataset indices [%s, %s): dataset_fps=%s, target_fps=%s, "
        "sample_stride=%s, rendered_frames=%s",
        start_index,
        end_index,
        dataset_fps,
        fps,
        sample_stride,
        len(sample_indices),
    )

    frames = []
    target_values = []
    for idx in trange(len(sample_indices), desc="Rendering Grad-CAM video frames"):
        sample_idx = sample_indices[idx]
        raw_batch = default_collate([dataset[sample_idx]])
        with torch.enable_grad():
            raw_images, cams, target_value = compute_inference_gradcam_tensors(
                policy=policy,
                preprocessor=preprocessor,
                raw_batch=raw_batch,
                action_step=action_step,
                action_dim=action_dim,
            )
        frame = render_gradcam_frame(
            raw_images=raw_images,
            cams=cams,
            obs_step=obs_step,
            camera_index=camera_index,
            view=view,
            alpha=alpha,
        )
        frames.append(frame.numpy())
        target_values.append((sample_idx, target_value))

    write_video(video_path, frames, fps=fps)
    (output_dir / f"{video_path.stem}_targets.txt").write_text(
        "\n".join(f"{sample_idx},{value}" for sample_idx, value in target_values) + "\n"
    )
    return video_path


def _get_episode_range_for_sample(dataset, sample_index: int) -> tuple[int, int, int]:
    """Return (episode_index, start_index, end_index_exclusive) for the sample's episode."""
    item = dataset[sample_index]
    episode_index = int(item["episode_index"].item())
    episode = dataset.meta.episodes[episode_index]
    start_index = int(episode["dataset_from_index"])
    end_index = int(episode["dataset_to_index"])
    return episode_index, start_index, end_index


def _get_episode_range(dataset, episode_index: int) -> tuple[int, int, int]:
    """Return (episode_index, start_index, end_index_exclusive) for an explicit episode index."""
    if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
        raise IndexError(
            f"episode-index {episode_index} is out of bounds for {len(dataset.meta.episodes)} episodes."
        )
    episode = dataset.meta.episodes[episode_index]
    start_index = int(episode["dataset_from_index"])
    end_index = int(episode["dataset_to_index"])
    return episode_index, start_index, end_index


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

    if args.export_video:
        video_start_index = args.sample_index
        video_num_samples = args.video_num_samples
        if args.video_episode_index is not None:
            episode_index, episode_start, episode_end = _get_episode_range(dataset, args.video_episode_index)
            video_start_index = episode_start
            video_num_samples = episode_end - episode_start
            logging.info(
                "Rendering explicit episode %s: dataset indices [%s, %s), total raw frames=%s, target fps=%s.",
                episode_index,
                episode_start,
                episode_end,
                video_num_samples,
                args.video_fps,
            )
        elif args.video_full_episode:
            episode_index, episode_start, episode_end = _get_episode_range_for_sample(dataset, args.sample_index)
            video_start_index = episode_start
            video_num_samples = episode_end - episode_start
            logging.info(
                "sample-index %s belongs to episode %s. Rendering the FULL episode from its first frame: "
                "dataset indices [%s, %s), total raw frames=%s, target fps=%s. "
                "If you want to start exactly at sample-index, omit --video-full-episode.",
                args.sample_index,
                episode_index,
                episode_start,
                episode_end,
                video_num_samples,
                args.video_fps,
            )

        video_path = _export_inference_gradcam_video(
            policy=policy,
            preprocessor=preprocessor,
            dataset=dataset,
            output_dir=args.output_dir,
            start_index=video_start_index,
            num_samples=video_num_samples,
            fps=args.video_fps,
            view=args.video_view,
            obs_step=args.video_obs_step,
            camera_index=args.video_camera_index,
            action_step=args.action_step,
            action_dim=args.action_dim,
            alpha=args.alpha,
            video_name=args.video_name,
        )
        logging.info("Grad-CAM video saved to: %s", video_path)

    logging.info("Grad-CAM export finished. Results saved under: %s", output_dir)


if __name__ == "__main__":
    main()
