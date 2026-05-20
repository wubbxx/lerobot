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

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def _stack_camera_images(batch: dict[str, Any], image_keys: list[str], n_obs_steps: int) -> Tensor:
    camera_tensors: list[Tensor] = []
    for key in image_keys:
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor for '{key}', got {type(value)}.")
        if n_obs_steps == 1 and value.ndim == 4:
            value = value.unsqueeze(1)
        camera_tensors.append(value)

    # (B, S, N, C, H, W)
    return torch.stack(camera_tensors, dim=-4)


def _tensor_to_uint8_hwc(image_chw: Tensor) -> Tensor:
    image = image_chw.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    return image.permute(1, 2, 0).contiguous()


def _make_overlay(image_chw: Tensor, cam_hw: Tensor, alpha: float) -> Tensor:
    # Simple red-yellow style map without extra plotting dependencies.
    image = image_chw.detach().float().cpu().clamp(0.0, 1.0)
    cam = cam_hw.detach().float().cpu().clamp(0.0, 1.0)

    red = cam
    green = cam * 0.6
    blue = torch.zeros_like(cam)
    heat_rgb = torch.stack((red, green, blue), dim=0)

    overlay = (1.0 - alpha) * image + alpha * heat_rgb
    overlay = overlay.clamp(0.0, 1.0)
    return _tensor_to_uint8_hwc(overlay)


def _save_visualization_grid(
    output_dir: Path,
    prefix: str,
    raw_images: Tensor,
    cams: Tensor,
    alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # raw_images: (B,S,N,C,H,W), cams: (B,S,N,H,W)
    batch_size, n_obs_steps, num_cameras = raw_images.shape[:3]

    for b in range(batch_size):
        for s in range(n_obs_steps):
            for n in range(num_cameras):
                stem = f"{prefix}_b{b}_s{s}_cam{n}"
                raw_uint8 = _tensor_to_uint8_hwc(raw_images[b, s, n])
                cam_uint8 = (cams[b, s, n].detach().float().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                overlay_uint8 = _make_overlay(raw_images[b, s, n], cams[b, s, n], alpha)

                Image.fromarray(raw_uint8.numpy()).save(output_dir / f"{stem}_raw.png")
                Image.fromarray(cam_uint8.numpy(), mode="L").save(output_dir / f"{stem}_cam.png")
                Image.fromarray(overlay_uint8.numpy()).save(output_dir / f"{stem}_overlay.png")


def _register_backbone_hooks(policy: PreTrainedPolicy) -> tuple[list[Any], dict[int, Tensor], dict[int, Tensor]]:
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        raise TypeError("Expected a diffusion policy (missing 'diffusion' module).")

    rgb_encoder = getattr(diffusion, "rgb_encoder", None)
    if rgb_encoder is None:
        raise TypeError("Policy does not contain an RGB encoder.")

    feats: dict[int, Tensor] = {}
    grads: dict[int, Tensor] = {}
    handles: list[Any] = []

    def _make_fwd_hook(index: int):
        def _hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            feats[index] = output

        return _hook

    def _make_bwd_hook(index: int):
        def _hook(_module: nn.Module, _grad_inputs: tuple[Tensor, ...], grad_outputs: tuple[Tensor, ...]) -> None:
            grads[index] = grad_outputs[0]

        return _hook

    if isinstance(rgb_encoder, nn.ModuleList):
        for idx, encoder in enumerate(rgb_encoder):
            handles.append(encoder.backbone.register_forward_hook(_make_fwd_hook(idx)))
            handles.append(encoder.backbone.register_full_backward_hook(_make_bwd_hook(idx)))
    else:
        handles.append(rgb_encoder.backbone.register_forward_hook(_make_fwd_hook(0)))
        handles.append(rgb_encoder.backbone.register_full_backward_hook(_make_bwd_hook(0)))

    return handles, feats, grads


def _compute_cams(
    feats: dict[int, Tensor],
    grads: dict[int, Tensor],
    batch_size: int,
    n_obs_steps: int,
    num_cameras: int,
    target_hw: tuple[int, int],
) -> Tensor:
    cams_by_camera: list[Tensor] = []

    if len(feats) != len(grads):
        raise RuntimeError("Failed to collect complete Grad-CAM hooks (feature/gradient mismatch).")

    if len(feats) == 1:
        fmap = feats[0]
        gmap = grads[0]
        weights = gmap.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam.view(batch_size, n_obs_steps, num_cameras, target_hw[0], target_hw[1])
        cams = cam
    else:
        for cam_idx in range(num_cameras):
            fmap = feats[cam_idx]
            gmap = grads[cam_idx]
            weights = gmap.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)
            cam = cam.view(batch_size, n_obs_steps, target_hw[0], target_hw[1])
            cams_by_camera.append(cam)
        cams = torch.stack(cams_by_camera, dim=2)

    flat = cams.flatten(start_dim=3)
    cam_min = flat.min(dim=-1, keepdim=True).values.view(batch_size, n_obs_steps, num_cameras, 1, 1)
    cam_max = flat.max(dim=-1, keepdim=True).values.view(batch_size, n_obs_steps, num_cameras, 1, 1)
    cams = (cams - cam_min) / (cam_max - cam_min + 1e-6)
    return cams


def export_train_gradcam(
    policy: PreTrainedPolicy,
    preprocessor: Any,
    raw_batch: dict[str, Any],
    output_dir: Path,
    alpha: float = 0.45,
) -> float:
    """Export Grad-CAM maps for the training loss path and return the scalar loss."""
    if not getattr(policy.config, "image_features", None):
        raise ValueError("This policy has no image features. Grad-CAM export requires visual inputs.")

    image_keys = list(policy.config.image_features.keys())
    raw_images = _stack_camera_images(raw_batch, image_keys, policy.config.n_obs_steps)
    target_hw = (raw_images.shape[-2], raw_images.shape[-1])

    batch = preprocessor(raw_batch)
    for key in image_keys:
        batch[key] = batch[key].detach().requires_grad_(True)

    handles, feats, grads = _register_backbone_hooks(policy)
    try:
        policy.zero_grad(set_to_none=True)
        policy.train()
        loss, _ = policy.forward(batch)
        loss.backward()

        cams = _compute_cams(
            feats=feats,
            grads=grads,
            batch_size=raw_images.shape[0],
            n_obs_steps=raw_images.shape[1],
            num_cameras=raw_images.shape[2],
            target_hw=target_hw,
        )
    finally:
        for handle in handles:
            handle.remove()

    _save_visualization_grid(output_dir, "train", raw_images, cams, alpha=alpha)
    return float(loss.detach().item())


def export_inference_gradcam(
    policy: PreTrainedPolicy,
    preprocessor: Any,
    raw_batch: dict[str, Any],
    output_dir: Path,
    action_step: int = 0,
    action_dim: int = 0,
    alpha: float = 0.45,
) -> float:
    """Export Grad-CAM maps for the inference action path and return the selected action scalar."""
    if not getattr(policy.config, "image_features", None):
        raise ValueError("This policy has no image features. Grad-CAM export requires visual inputs.")

    image_keys = list(policy.config.image_features.keys())
    raw_images = _stack_camera_images(raw_batch, image_keys, policy.config.n_obs_steps)
    target_hw = (raw_images.shape[-2], raw_images.shape[-1])

    batch = preprocessor(raw_batch)
    for key in image_keys:
        batch[key] = batch[key].detach().requires_grad_(True)

    obs_images = _stack_camera_images(batch, image_keys, policy.config.n_obs_steps)
    diffusion_batch = {
        OBS_STATE: batch[OBS_STATE],
        OBS_IMAGES: obs_images,
    }
    if policy.config.env_state_feature:
        diffusion_batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE]

    handles, feats, grads = _register_backbone_hooks(policy)
    try:
        policy.zero_grad(set_to_none=True)
        policy.eval()
        actions = policy.diffusion.generate_actions(diffusion_batch)
        step_index = max(0, min(action_step, actions.shape[1] - 1))
        dim_index = max(0, min(action_dim, actions.shape[2] - 1))
        target = actions[:, step_index, dim_index].sum()
        target.backward()

        cams = _compute_cams(
            feats=feats,
            grads=grads,
            batch_size=raw_images.shape[0],
            n_obs_steps=raw_images.shape[1],
            num_cameras=raw_images.shape[2],
            target_hw=target_hw,
        )
    finally:
        for handle in handles:
            handle.remove()

    _save_visualization_grid(output_dir, "inference", raw_images, cams, alpha=alpha)
    return float(target.detach().item())
