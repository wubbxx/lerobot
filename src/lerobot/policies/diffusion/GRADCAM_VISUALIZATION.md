# Diffusion Policy Grad-CAM 可视化说明

本文档说明新增的 Diffusion Policy 图像梯度可视化功能，包括：

1. 为什么要做这个功能。
2. 当前 Diffusion Policy 里为什么没有可直接读取的 Transformer Attention。
3. Grad-CAM 是什么。
4. PyTorch hook 是什么，以及为什么这里需要 hook。
5. 训练样本 Grad-CAM 和推理动作 Grad-CAM 分别怎么算。
6. 代码实现对应关系。
7. 最终如何运行、结果如何解读、常见问题如何排查。

---

## 1. 背景：为什么需要 Grad-CAM

当前使用的 LeRobot `diffusion` policy 是一个视觉-动作模型。它输入多帧观测图像和机器人状态，输出未来一段动作轨迹。

以当前数据流为例，模型输入大致是：

```text
observation.state        (B, S, state_dim)
observation.images.front (B, S, C, H, W)
```

其中：

- `B` 是 batch size。
- `S` 是观测帧数，例如 `n_obs_steps=2` 时，输入当前帧和上一帧。
- `C,H,W` 是图像通道、高、宽。

模型最终会输出动作：

```text
action chunk (B, n_action_steps, action_dim)
```

问题是：当模型输出一个动作时，我们想知道它主要参考了图像中的哪些区域。例如：

- 是否看到了黄色目标物？
- 是否关注了机械臂末端？
- 当前动作到底主要依赖上一帧还是当前帧？

如果模型里有 Transformer attention，可以尝试读取 attention map。但当前 Diffusion Policy 并没有这种可直接读取的 attention 矩阵，所以需要另一种解释方法。

Grad-CAM 就是用来回答「模型输出对图像哪个区域更敏感」的一种常用方法。

---

## 2. 当前 Diffusion Policy 里有没有 Attention

当前 Diffusion Policy 的主要结构是：

```text
图像输入
  -> ResNet backbone
  -> SpatialSoftmax
  -> image feature

机器人状态
  -> state feature

image feature + state feature
  -> global conditioning
  -> 1D Conditional U-Net
  -> action trajectory
```

这里有几个关键点：

1. 视觉 backbone 是 ResNet。
2. 图像特征经过 `SpatialSoftmax` 压缩为关键点特征。
3. 动作生成网络是 1D 卷积 U-Net。
4. 条件信息通过 FiLM 注入 U-Net。
5. 没有 Transformer Self-Attention，也没有 Multi-Head Attention。

所以，本功能不是在读取 Transformer 的 attention score，而是在做基于梯度的视觉解释。

### 2.1 SpatialSoftmax 和 Grad-CAM 的区别

`SpatialSoftmax` 会在 ResNet 输出的特征图上做空间 softmax，它确实会产生一种空间分布：

$$
p_{k,i,j}=\frac{\exp(F_{k,i,j})}{\sum_{u,v}\exp(F_{k,u,v})}
$$

其中：

- $F_{k,i,j}$ 是第 $k$ 个特征通道在空间位置 $(i,j)$ 的值。
- $p_{k,i,j}$ 是 softmax 后的空间权重。

但 `SpatialSoftmax` 主要用于把特征图压缩成关键点坐标，不等价于 Transformer attention。

Grad-CAM 的目标不同：它关心的是「某个输出目标」对「某个特征图区域」有多敏感。

---

## 3. Grad-CAM 是什么

Grad-CAM，全称 Gradient-weighted Class Activation Mapping。

它原本常用于分类模型，例如解释「为什么模型认为这张图是猫」。在这里，我们把它改造成用于机器人策略：解释「为什么模型输出这个动作」。

核心思想是：

> 如果某个图像区域对目标输出很重要，那么目标输出对该区域对应的深层特征应该有较大的梯度响应。

在当前功能里，目标输出可以有两种：

1. 训练路径：目标是训练 loss。
2. 推理路径：目标是某个动作分量，例如第 0 个动作步、第 0 个关节动作。

---

## 4. Grad-CAM 数学公式

假设我们选中 ResNet backbone 最后一层输出特征图：

$$
A \in \mathbb{R}^{K \times H_f \times W_f}
$$

其中：

- $K$ 是通道数。
- $H_f,W_f$ 是特征图的空间尺寸。
- $A^k_{i,j}$ 表示第 $k$ 个通道在位置 $(i,j)$ 的激活值。

设我们关心的目标标量为 $y$。

在训练 Grad-CAM 中：

$$
y = \mathcal{L}
$$

也就是训练 loss。

在推理 Grad-CAM 中：

$$
y = a_{b,t,d}
$$

也就是 batch 中第 $b$ 个样本、第 $t$ 个动作时间步、第 $d$ 个动作维度的输出。

### 4.1 计算通道权重

先对目标 $y$ 对特征图 $A^k$ 求梯度：

$$
\frac{\partial y}{\partial A^k_{i,j}}
$$

然后对空间维度做全局平均池化，得到第 $k$ 个通道的权重：

$$
\alpha_k = \frac{1}{H_f W_f}\sum_i\sum_j \frac{\partial y}{\partial A^k_{i,j}}
$$

直观理解：

- 如果某个通道的梯度整体很大，说明目标输出很依赖这个通道。
- $\alpha_k$ 就是这个通道对目标输出的重要性。

### 4.2 生成 CAM 热力图

用通道权重加权特征图，并经过 ReLU：

$$
L_{\text{Grad-CAM}}(i,j)=\text{ReLU}\left(\sum_k \alpha_k A^k_{i,j}\right)
$$

这里使用 ReLU 是为了只保留对目标有正贡献的区域。

之后把低分辨率 CAM 上采样到原图大小：

$$
L_{\text{up}} \in \mathbb{R}^{H \times W}
$$

最后归一化到 $[0,1]$：

$$
\hat{L}(i,j)=\frac{L(i,j)-\min(L)}{\max(L)-\min(L)+\epsilon}
$$

输出的 $\hat{L}$ 就是最终保存的灰度热力图。

---

## 5. PyTorch hook 是什么

很多人第一次看 Grad-CAM 会不理解「挂 hook」是什么意思。

### 5.1 正常前向和反向只能拿到什么

正常调用模型时，一般只会写：

```python
loss, _ = policy.forward(batch)
loss.backward()
```

这样可以完成训练，但我们拿不到中间层的特征图，也拿不到中间层特征图对应的梯度。

Grad-CAM 恰好需要这两个东西：

1. ResNet backbone 输出的 feature map。
2. 目标标量对这个 feature map 的 gradient。

所以需要一种方式在模型运行时「偷看」中间结果。

### 5.2 hook 的直观解释

hook 可以理解为挂在某个网络层上的监听器。

例如把 hook 挂在 ResNet backbone 上：

```text
image -> backbone -> feature map -> SpatialSoftmax -> U-Net -> action/loss
           ↑
        hook 挂在这里
```

当模型前向经过 backbone 时，forward hook 会自动触发，把 backbone 输出的 feature map 保存下来。

当模型反向传播经过 backbone 时，backward hook 会自动触发，把目标对 backbone 输出的梯度保存下来。

### 5.3 本功能里使用了两类 hook

1. forward hook

用于保存 feature map：

```python
feats[index] = output
```

2. backward hook

用于保存 gradient：

```python
grads[index] = grad_outputs[0]
```

这里的 `index` 是为了支持多相机或多个 encoder。

### 5.4 hook 是否会改变模型输出

不会。

hook 只是监听和记录中间张量，不修改模型结构，也不改变参数。它只在可视化脚本里使用，正常训练和推理不会触发。

---

## 6. 本功能的两条可视化路径

新增功能分为两个独立函数：

1. `export_train_gradcam`
2. `export_inference_gradcam`

它们都在：

```text
src/lerobot/policies/diffusion/gradcam_visualization.py
```

---

## 7. 训练样本 Grad-CAM

训练 Grad-CAM 的目标是解释：

> 当前训练样本的 loss 主要由图像中的哪些区域影响。

### 7.1 输入

来自 dataset 的一个 batch，例如：

```text
observation.state        (B, S, state_dim)
observation.images.front (B, S, C, H, W)
action                   (B, horizon, action_dim)
action_is_pad            (B, horizon)
```

### 7.2 处理流程

1. 从原始 batch 里取出图像，保存一份未归一化图像用于最后叠图。
2. 通过 preprocessor 把 batch 移到设备并做归一化。
3. 对图像张量设置：

```python
requires_grad_(True)
```

这表示需要保留和计算图像相关的梯度。

4. 在 ResNet backbone 上注册 hook。
5. 调用：

```python
loss, _ = policy.forward(batch)
```

6. 调用：

```python
loss.backward()
```

7. 根据 hook 保存的 feature map 和 gradient 计算 CAM。
8. 保存图片到本地。

### 7.3 数学目标

训练路径中，Grad-CAM 的目标标量是：

$$
y = \mathcal{L}_{\text{diffusion}}
$$

所以最终热力图表示：

> 哪些图像区域对当前训练 loss 更敏感。

---

## 8. 推理动作 Grad-CAM

推理 Grad-CAM 的目标是解释：

> 当前模型输出的某个动作分量主要依赖图像中的哪些区域。

### 8.1 输入

同样使用 dataset 中取出的一个样本，但推理路径不需要使用 action 标签作为目标。

它关心的是模型生成出来的动作：

```text
actions = policy.diffusion.generate_actions(diffusion_batch)
actions shape: (B, n_action_steps, action_dim)
```

### 8.2 action_step 和 action_dim 是什么

推理输出是一个动作 chunk，不是单个动作。

例如：

```text
actions (B, 8, 4)
```

含义是：

- 一次预测 8 个未来动作。
- 每个动作有 4 个维度。

如果设置：

```bash
--action-step 0
--action-dim 0
```

表示解释第 0 个未来动作的第 0 个关节维度。

数学上就是：

$$
y = a_{0,0,0}
$$

如果设置：

```bash
--action-step 3
--action-dim 2
```

则表示解释：

$$
y = a_{0,3,2}
$$

### 8.3 处理流程

1. 取出图像和状态。
2. 做 preprocessor。
3. 把多相机图像堆叠成内部格式：

```text
observation.images (B, S, N, C, H, W)
```

4. 构造 diffusion_batch。
5. 在 backbone 上注册 hook。
6. 调用：

```python
actions = policy.diffusion.generate_actions(diffusion_batch)
```

7. 选定目标动作标量：

```python
target = actions[:, action_step, action_dim].sum()
```

8. 调用：

```python
target.backward()
```

9. 用 feature map 和 gradient 计算 CAM。
10. 保存图片。

---

## 9. 为什么不直接用 `select_action`

当前 `DiffusionPolicy.select_action()` 上有 `@torch.no_grad()`。

这意味着它会关闭梯度计算。如果直接用 `select_action()`，就没法反向传播，也就算不了 Grad-CAM。

所以推理 Grad-CAM 里没有走 `select_action()`，而是直接调用：

```python
policy.diffusion.generate_actions(diffusion_batch)
```

这样可以保留计算图，从而对动作输出做反向传播。

---

## 10. 文件与函数对应关系

### 10.1 核心可视化模块

```text
src/lerobot/policies/diffusion/gradcam_visualization.py
```

核心函数：

```text
export_train_gradcam(...)
export_inference_gradcam(...)
```

辅助函数：

```text
_stack_camera_images(...)
_register_backbone_hooks(...)
_compute_cams(...)
_save_visualization_grid(...)
```

### 10.2 独立运行脚本

```text
src/lerobot/scripts/lerobot_diffusion_gradcam.py
```

职责：

1. 读取训练配置。
2. 加载 dataset。
3. 加载 checkpoint policy。
4. 构建 preprocessor。
5. 选取一个 dataset sample。
6. 调用训练 Grad-CAM。
7. 调用推理 Grad-CAM。
8. 写出图片和 summary。

### 10.3 命令入口

```text
pyproject.toml
```

新增入口：

```text
lerobot-diffusion-gradcam
```

如果没有重新 `pip install -e .`，这个命令可能不可用。此时可以直接用 `python -m` 方式运行。

---

## 11. 输出文件说明

默认输出目录类似：

```text
outputs/gradcam/sample_000000/
```

其中 `sample_000000` 对应 `--sample-index 0`。

每个时间步、每个相机都会输出三张图。

### 11.1 训练路径输出

```text
train_b0_s0_cam0_raw.png
train_b0_s0_cam0_cam.png
train_b0_s0_cam0_overlay.png
train_b0_s1_cam0_raw.png
train_b0_s1_cam0_cam.png
train_b0_s1_cam0_overlay.png
```

含义：

- `train`：训练 loss 的 Grad-CAM。
- `b0`：batch 中第 0 个样本。
- `s0/s1`：第几个观测帧。
- `cam0`：第几个相机。
- `raw`：原图。
- `cam`：灰度热力图。
- `overlay`：热力图叠加到原图。

### 11.2 推理路径输出

```text
inference_b0_s0_cam0_raw.png
inference_b0_s0_cam0_cam.png
inference_b0_s0_cam0_overlay.png
inference_b0_s1_cam0_raw.png
inference_b0_s1_cam0_cam.png
inference_b0_s1_cam0_overlay.png
```

含义同上，但 `inference` 表示目标是某个动作输出标量。

### 11.3 summary.txt

脚本还会输出：

```text
summary.txt
```

其中记录：

- policy path
- train config path
- sample index
- train loss target
- inference action target
- action step
- action dim

---

## 12. 最终用法

### 12.1 推荐方式：直接用环境 Python 运行

这种方式不依赖命令入口是否注册，最稳定。

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

### 12.2 使用命令入口

如果想使用：

```bash
lerobot-diffusion-gradcam
```

需要先重新安装当前工程：

```bash
cd /home/wbx/embodied-ai/lerobot
/home/wbx/miniconda3/envs/lerobot/bin/python -m pip install -e .
hash -r
```

然后检查：

```bash
lerobot-diffusion-gradcam --help
```

再运行：

```bash
lerobot-diffusion-gradcam \
  --enable-gradcam \
  --train-config-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/train_config.json \
  --policy-path /home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/last/pretrained_model \
  --output-dir /home/wbx/embodied-ai/lerobot/outputs/gradcam \
  --sample-index 0 \
  --action-step 0 \
  --action-dim 0
```

---

## 13. 参数说明

### `--enable-gradcam`

是否启用 Grad-CAM。

默认关闭。如果不传这个参数，脚本会直接退出，不会做任何可视化计算。

### `--train-config-path`

训练配置文件路径。

必须是真实存在的 `train_config.json`，不能使用 `/path/to/train_config.json` 这种占位符。

### `--policy-path`

checkpoint 中的 `pretrained_model` 目录。

该目录一般包含：

```text
config.json
model.safetensors
```

### `--output-dir`

Grad-CAM 图片输出根目录。

### `--sample-index`

选择 dataset 中第几个样本。

例如：

```bash
--sample-index 0
```

表示取第 0 个训练样本。

### `--action-step`

推理 Grad-CAM 中要解释的动作时间步。

例如模型输出：

```text
actions (B, 8, 4)
```

则 `--action-step 0` 到 `--action-step 7` 是合法范围。

### `--action-dim`

推理 Grad-CAM 中要解释的动作维度。

例如你的数据集中 action 是 4 维关节位置命令，则 `--action-dim 0` 到 `--action-dim 3` 是合法范围。

### `--alpha`

热力图叠加到原图时的透明度。

默认：

```bash
--alpha 0.45
```

### `--device`

可选，覆盖配置中的设备：

```bash
--device cpu
--device cuda
```

---

## 14. 如何解读结果

### 14.1 看哪张图

优先看：

```text
*_overlay.png
```

overlay 是热力图叠加原图，最直观。

### 14.2 亮的地方代表什么

亮色区域代表：

> 当前目标标量对该区域对应的视觉特征更敏感。

对于训练 Grad-CAM：

> 亮区表示当前 loss 对这些区域更敏感。

对于推理 Grad-CAM：

> 亮区表示当前选中的动作分量更依赖这些区域。

### 14.3 亮区是不是严格等于 Attention

不是。

Grad-CAM 是梯度归因图，不是 Transformer attention map。

更准确地说，它表示敏感度或贡献倾向，而不是概率意义上的注意力权重。

### 14.4 如何比较多帧

如果 `n_obs_steps=2`，会有：

```text
s0: 较早帧
s1: 当前帧
```

可以比较：

- 哪一帧热力图更集中。
- 哪一帧目标区域更亮。
- 哪一帧 overlay 更符合任务语义。

如果后续需要更量化，可以增加每帧 CAM 均值/最大值统计。

---

## 15. 常见问题

### 15.1 `lerobot-diffusion-gradcam: command not found`

原因：新增的命令入口还没有安装到当前环境。

解决方法 1：直接用 `python -m`。

解决方法 2：重新安装当前工程：

```bash
cd /home/wbx/embodied-ai/lerobot
/home/wbx/miniconda3/envs/lerobot/bin/python -m pip install -e .
hash -r
```

### 15.2 `ModuleNotFoundError: torch`

原因：使用了错误的 Python 环境。

解决：使用 lerobot conda 环境里的 Python：

```bash
/home/wbx/miniconda3/envs/lerobot/bin/python
```

### 15.3 `train_config.json` 找不到

原因：`--train-config-path` 仍然是占位符，或路径写错。

解决：替换为真实路径，例如：

```text
/home/wbx/embodied-ai/lerobot/outputs/train/reach_yellow/train_config.json
```

### 15.4 显存不足

Grad-CAM 需要保留计算图并做 backward，比普通推理占显存。

解决建议：

1. 使用 `--device cpu`，速度慢但省显存。
2. 只可视化少量样本。
3. 如果图像分辨率很高，可考虑后续增加 resize 可视化路径。

---

## 16. 为什么不会影响正常训练和推理

本功能没有修改原始训练主循环，也没有修改原始评估脚本。

新增内容是独立的：

```text
gradcam_visualization.py
lerobot_diffusion_gradcam.py
```

只有显式运行可视化脚本并传入：

```bash
--enable-gradcam
```

才会执行 hook、backward 和图片保存。

因此默认不会影响任何正常训练、评估或部署逻辑。

---

## 17. 后续可扩展方向

1. 导出每帧 CAM 强度 CSV，用于量化比较不同帧的重要性。
2. 同时导出 `SpatialSoftmax` 的空间分布，与 Grad-CAM 做对照。
3. 支持批量样本可视化。
4. 支持按 action dim 自动遍历，观察不同关节维度关注区域是否不同。
5. 支持保存 GIF 或 HTML 报告，方便快速浏览多帧结果。
