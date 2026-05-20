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


def _colorize_cam(cam_hw: Tensor, gamma: float = 0.55) -> Tensor:
    """Map a normalized CAM to a cold-to-warm RGB heatmap.

    Low values become dark blue/cyan, high values become yellow/red. A gamma below 1 makes weak but
    non-zero responses more visible, which is useful for regression policies whose gradients can be small.
    """
    cam = cam_hw.detach().float().cpu().clamp(0.0, 1.0)
    cam = cam.pow(gamma)

    # Piecewise-linear colormap with anchors:
    # 0.00: dark blue, 0.35: cyan, 0.65: yellow, 1.00: red
    anchors = torch.tensor(
        [
            [0.02, 0.04, 0.35],
            [0.00, 0.75, 1.00],
            [1.00, 0.95, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor([0.0, 0.35, 0.65, 1.0], dtype=torch.float32)

    heat = torch.empty(3, *cam.shape, dtype=torch.float32)
    for idx in range(len(positions) - 1):
        left = positions[idx]
        right = positions[idx + 1]
        mask = (cam >= left) & (cam <= right)
        ratio = ((cam - left) / (right - left)).clamp(0.0, 1.0)
        color = anchors[idx].view(3, 1, 1) * (1.0 - ratio) + anchors[idx + 1].view(3, 1, 1) * ratio
        heat[:, mask] = color[:, mask]
    return heat.clamp(0.0, 1.0)


def _make_overlay(image_chw: Tensor, cam_hw: Tensor, alpha: float) -> Tensor:
    # Cold-to-warm semi-transparent map without extra plotting dependencies.
    image = image_chw.detach().float().cpu().clamp(0.0, 1.0)
    heat_rgb = _colorize_cam(cam_hw)

    overlay = (1.0 - alpha) * image + alpha * heat_rgb
    overlay = overlay.clamp(0.0, 1.0)
    return _tensor_to_uint8_hwc(overlay)


def render_gradcam_frame(
    raw_images: Tensor,
    cams: Tensor,
    obs_step: int = -1,
    camera_index: int = 0,
    view: str = "overlay",
    alpha: float = 0.45,
) -> Tensor:
    """Render one Grad-CAM frame as an uint8 HWC tensor.

    Args:
        raw_images: Raw image tensor shaped (B, S, N, C, H, W), with values in [0, 1].
        cams: Normalized CAM tensor shaped (B, S, N, H, W), with values in [0, 1].
        obs_step: Observation step to render. Negative values follow Python indexing, so -1 means current
            latest observation.
        camera_index: Camera index to render.
        view: One of "overlay", "heat", "cam", or "raw".
        alpha: Overlay blending strength when view="overlay".
    """
    n_obs_steps = raw_images.shape[1]
    num_cameras = raw_images.shape[2]
    step_index = obs_step if obs_step >= 0 else n_obs_steps + obs_step
    step_index = max(0, min(step_index, n_obs_steps - 1))
    camera_index = max(0, min(camera_index, num_cameras - 1))

    image = raw_images[0, step_index, camera_index]
    cam = cams[0, step_index, camera_index]

    if view == "overlay":
        return _make_overlay(image, cam, alpha)
    if view == "heat":
        return _tensor_to_uint8_hwc(_colorize_cam(cam))
    if view == "cam":
        gray = (cam.detach().float().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return gray.unsqueeze(-1).expand(-1, -1, 3).contiguous()
    if view == "raw":
        return _tensor_to_uint8_hwc(image)
    raise ValueError(f"Unsupported view '{view}'. Expected one of: overlay, heat, cam, raw.")


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
                heat_uint8 = _tensor_to_uint8_hwc(_colorize_cam(cams[b, s, n]))
                overlay_uint8 = _make_overlay(raw_images[b, s, n], cams[b, s, n], alpha)

                Image.fromarray(raw_uint8.numpy()).save(output_dir / f"{stem}_raw.png")
                Image.fromarray(cam_uint8.numpy(), mode="L").save(output_dir / f"{stem}_cam.png")
                Image.fromarray(heat_uint8.numpy()).save(output_dir / f"{stem}_heat.png")
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
        # For regression/action outputs, standard Grad-CAM's ReLU can hide all negative contributions.
        # Use absolute contribution so that both positive and negative high-sensitivity regions are visible.
        cam = (weights * fmap).sum(dim=1, keepdim=True).abs()
        if float(cam.detach().max().cpu()) <= 1e-12:
            cam = (gmap.abs() * fmap.abs()).mean(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam.view(batch_size, n_obs_steps, num_cameras, target_hw[0], target_hw[1])
        cams = cam
    else:
        for cam_idx in range(num_cameras):
            fmap = feats[cam_idx]
            gmap = grads[cam_idx]
            weights = gmap.mean(dim=(2, 3), keepdim=True)
            cam = (weights * fmap).sum(dim=1, keepdim=True).abs()
            if float(cam.detach().max().cpu()) <= 1e-12:
                cam = (gmap.abs() * fmap.abs()).mean(dim=1, keepdim=True)
            cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)
            cam = cam.view(batch_size, n_obs_steps, target_hw[0], target_hw[1])
            cams_by_camera.append(cam)
        cams = torch.stack(cams_by_camera, dim=2)

    flat = cams.flatten(start_dim=3)
    cam_min = flat.min(dim=-1, keepdim=True).values.view(batch_size, n_obs_steps, num_cameras, 1, 1)
    # Percentile clipping improves visibility when only a few pixels dominate the map.
    cam_hi = torch.quantile(flat.float(), 0.995, dim=-1, keepdim=True).view(
        batch_size, n_obs_steps, num_cameras, 1, 1
    )
    cam_max = flat.max(dim=-1, keepdim=True).values.view(batch_size, n_obs_steps, num_cameras, 1, 1)
    cam_hi = torch.maximum(cam_hi, cam_max * 1e-6)
    cams = ((cams - cam_min) / (cam_hi - cam_min + 1e-12)).clamp(0.0, 1.0)
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


def compute_inference_gradcam_tensors(
    policy: PreTrainedPolicy,
    preprocessor: Any,
    raw_batch: dict[str, Any],
    action_step: int = 0,
    action_dim: int = 0,
) -> tuple[Tensor, Tensor, float]:
    """Compute inference Grad-CAM tensors without saving images.

    Returns:
        raw_images: (B, S, N, C, H, W), values in [0, 1], CPU/GPU as provided by dataset batch.
        cams: (B, S, N, H, W), normalized to [0, 1].
        target_value: Scalar action target used for backpropagation.
    """
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

    return raw_images, cams, float(target.detach().item())
