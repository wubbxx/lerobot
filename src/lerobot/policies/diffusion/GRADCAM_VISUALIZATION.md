# Diffusion Policy Grad-CAM / 梯度贡献可视化说明

本文档说明 Diffusion Policy 图像可解释性功能：用梯度贡献图观察模型在训练和推理时更依赖图像中的哪些区域，并支持把连续样本导出为动态视频。

## 1. 功能背景

当前 LeRobot `diffusion` policy 主要结构是：

```text
image -> ResNet backbone -> SpatialSoftmax -> image feature
state -> state feature
image/state condition -> 1D Conditional U-Net -> action trajectory
```

它没有 Transformer Multi-Head Attention，因此没有可直接读取的 QK attention map。本功能不是读取严格意义上的 attention，而是用梯度归因近似回答：

> 当前 loss 或动作输出对图像中哪些区域更敏感？

## 2. 可视化含义

输出的是梯度贡献强度图：

- 冷色/深蓝：梯度贡献小，模型对该区域不敏感。
- 暖色/黄红：梯度贡献大，模型对该区域更敏感。

它不等价于 Transformer attention 概率，但可以用于观察策略输出对图像区域的依赖程度。

## 3. 为什么使用绝对梯度贡献

标准 Grad-CAM 常用于分类任务，通常计算：

$$
L(i,j)=\mathrm{ReLU}\left(\sum_k \alpha_k A^k_{i,j}\right)
$$

但 Diffusion Policy 输出连续动作，是回归问题。负贡献区域也可能非常重要，如果直接用 ReLU 会被清零，容易导致热力图全黑。

因此当前实现使用绝对贡献：

$$
L(i,j)=\left|\sum_k \alpha_k A^k_{i,j}\right|
$$

若该值仍接近 0，则回退到更直接的梯度强度估计：

$$
L(i,j)=\frac{1}{K}\sum_k \left|\frac{\partial y}{\partial A^k_{i,j}}\right|\cdot |A^k_{i,j}|
$$

其中：

- $A^k_{i,j}$ 是 backbone 特征图第 $k$ 个通道在位置 $(i,j)$ 的激活。
- $y$ 是目标标量，训练时为 loss，推理时为某个动作分量。

通道权重为：

$$
\alpha_k=\frac{1}{H_fW_f}\sum_i\sum_j\frac{\partial y}{\partial A^k_{i,j}}
$$

最终 CAM 会被上采样到原图大小，并归一化到 $[0,1]$。

## 4. hook 是什么

Grad-CAM 需要两个中间量：

1. ResNet backbone 输出的 feature map。
2. 目标标量对 feature map 的梯度。

正常前向/反向只会得到 loss 或 action，不会自动把这些中间量暴露出来，所以需要在 backbone 上挂 hook。

```text
image -> backbone -> feature map -> SpatialSoftmax -> U-Net -> loss/action
           ↑
        hook 挂在这里
```

- forward hook：前向经过 backbone 时保存 feature map。
- backward hook：反向经过 backbone 时保存 gradient。

hook 只记录中间张量，不改变模型结构、参数或输出。

## 5. 支持的输出

### 5.1 单样本图片

每个样本会输出：

```text
*_raw.png      原图
*_cam.png      灰度热力图
*_heat.png     冷暖色热力图
*_overlay.png  冷暖色热力图半透明叠加原图
```

训练路径文件名前缀为 `train_`，推理路径文件名前缀为 `inference_`。

### 5.2 动态视频

支持把连续 dataset sample 的推理 Grad-CAM 导出为 mp4。

可选视频视图：

```text
overlay  热力图半透明叠加原图，推荐
heat     纯冷暖色热力图
cam      灰度热力图
raw      原图视频，用于对照
```

## 6. 运行示例

### 6.1 单样本图片

```bash
cd /home/wbx/embodied-ai/lerobot

/home/wbx/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_diffusion_gradcam \
  --enable-gradcam \
  --train-config-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/train_config.json \
  --policy-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/last/pretrained_model \
  --output-dir /home/wbx/embodied-ai/lerobot/outputs/gradcam \
  --sample-index 0 \
  --action-step 0 \
  --action-dim 0
```

### 6.2 完整 episode 视频，低帧率、更透明

```bash
cd /home/wbx/embodied-ai/lerobot

/home/wbx/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_diffusion_gradcam \
  --enable-gradcam \
  --train-config-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/train_config.json \
  --policy-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/last/pretrained_model \
  --output-dir /home/wbx/embodied-ai/lerobot/outputs/gradcam \
  --sample-index 0 \
  --action-step 0 \
  --action-dim 0 \
  --alpha 0.3 \
  --export-video \
  --video-full-episode \
  --video-fps 10 \
  --video-view overlay
```

说明：

- `--video-full-episode`：自动找到 `--sample-index` 所在 episode 并导出完整 episode。
- `--video-fps 10`：目标视频采样 FPS。如果数据集是 30 FPS，则每 3 帧采样 1 帧做 Grad-CAM，推理帧数减少到约 1/3，同时视频时长接近原始 episode。
- `--alpha 0.3`：热力图更透明。

注意：如果使用 `--video-full-episode`，`--sample-index` 的作用是“选择它所在的 episode”，不是“从这个 sample 开始”。例如 `sample-index=44` 和 `sample-index=0` 如果在同一个 episode，导出的视频会是同一个完整 episode。如果希望视频从 sample 44 开始，不要加 `--video-full-episode`，改用 `--video-num-samples` 指定从 44 开始导出多少个原始样本范围。

例如从 sample 44 开始导出后续 300 个原始样本范围，并按 10 FPS 抽帧：

```bash
/home/wbx/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_diffusion_gradcam \
  --enable-gradcam \
  --train-config-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/train_config.json \
  --policy-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/last/pretrained_model \
  --output-dir /home/wbx/embodied-ai/lerobot/outputs/gradcam \
  --sample-index 44 \
  --action-step 0 \
  --action-dim 0 \
  --alpha 0.3 \
  --export-video \
  --video-num-samples 300 \
  --video-fps 10 \
  --video-view overlay
```

## 7. 主要实现文件

```text
src/lerobot/policies/diffusion/gradcam_visualization.py
src/lerobot/scripts/lerobot_diffusion_gradcam.py
pyproject.toml
```

## 8. 注意事项

1. `select_action()` 带有 `@torch.no_grad()`，不能用于 Grad-CAM 反传，因此推理路径直接调用 `policy.diffusion.generate_actions()`。
2. 视频每一帧都需要 diffusion 推理和 backward，速度会比普通推理慢很多。
3. `--video-fps` 当前表示“目标采样 FPS + 输出 FPS”。例如数据集 30 FPS、目标 10 FPS 时，脚本每 3 帧计算一次 Grad-CAM，并以 10 FPS 写视频；这会减少推理帧数，同时保持接近原始时长。
4. 如果短命令 `lerobot-diffusion-gradcam` 不存在，需要重新执行 `pip install -e .`，或直接使用 `python -m lerobot.scripts.lerobot_diffusion_gradcam`。
