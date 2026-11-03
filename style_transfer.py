# SPDX-License-Identifier: MIT
"""VGG19 + L-BFGS 神经风格迁移引擎。

算法来自 StyleTransfer.ipynb（d2l 风格迁移示例），并做了工程化封装：
- 不再依赖 d2l / matplotlib，只保留 torch + torchvision + Pillow；
- 把风格权重、内容权重、TV 权重、笔触尺度等暴露为可调参数；
- 支持进度回调与中间预览，方便在 Web 界面中展示优化过程。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF

__version__ = "0.1.0"

# ImageNet 归一化参数（与 notebook 保持一致）
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

# VGG19 中对应的 relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
STYLE_LAYERS: tuple[int, ...] = (1, 6, 11, 20, 29)
# 内容层 relu4_2
CONTENT_LAYERS: tuple[int, ...] = (22,)

# notebook 中的原始风格层权重：浅层负责色彩，深层负责中大尺度笔触
BASE_STYLE_LAYER_WEIGHTS: tuple[float, ...] = (0.05, 0.10, 0.50, 2.50, 1.00)

ProgressCallback = Callable[[int, int, Image.Image | None], None]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def open_rgb_image(source) -> Image.Image:
    """打开图片，处理 EXIF 方向与透明通道，统一转换为 RGB。"""
    image = Image.open(source)
    image = ImageOps.exif_transpose(image)

    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return image.convert("RGB")


def fit_shape(image: Image.Image, max_size: int, min_side: int = 16) -> tuple[int, int]:
    """按原图长宽比缩放，使长边不超过 max_size；返回 (height, width)。

    对极端全景图，允许为保证短边不小于 min_side 而略微突破 max_size。
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效。")

    scale = min(1.0, max_size / float(max(width, height)))
    if min(width, height) * scale < min_side:
        scale = min_side / float(min(width, height))

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_height, new_width


@dataclass(frozen=True)
class StyleParams:
    """一次风格化所使用的全部可调参数。

    注意：num_steps 是 L-BFGS 的外层轮数，每轮内部默认执行 max_iter 次迭代。
    """

    style_weight: float = 3.0e6
    content_weight: float = 0.5
    color_strength: float = 1.0
    brush_strength: float = 1.0
    brush_scale: float = 0.55
    tv_weight: float = 0.2
    num_steps: int = 40
    output_size: int = 512
    seed: int = 42
    max_iter: int = 10
    learning_rate: float = 1.0
    preview_every: int = 5

    @property
    def style_input_size(self) -> int:
        """笔触尺度 -> 风格图输入长边。

        0.0 对应 512px（细笔触），1.0 对应 224px（粗笔触）。
        默认 0.55 约为 354px，与 notebook 中的 350px 基本一致。
        """
        return int(round(512 - 288 * _clamp(self.brush_scale, 0.0, 1.0)))

    def layer_weights(self) -> list[float]:
        """根据色彩/笔触滑块重新分配五个风格层的权重。"""
        color = max(0.0, float(self.color_strength))
        brush = max(0.0, float(self.brush_strength))

        weights = [
            BASE_STYLE_LAYER_WEIGHTS[0] * color,
            BASE_STYLE_LAYER_WEIGHTS[1] * color,
            # relu3_1 处于色彩与纹理之间，按比例混合两种控制
            BASE_STYLE_LAYER_WEIGHTS[2] * (0.35 * color + 0.65 * brush),
            BASE_STYLE_LAYER_WEIGHTS[3] * brush,
            BASE_STYLE_LAYER_WEIGHTS[4] * brush,
        ]
        # 避免所有层权重都为 0 时出现 0 梯度
        if sum(weights) <= 1e-12:
            weights = [1e-6] * len(weights)
        return weights

    def signature(self) -> tuple:
        return (
            round(self.style_weight, 3),
            round(self.content_weight, 3),
            round(self.color_strength, 3),
            round(self.brush_strength, 3),
            round(self.brush_scale, 3),
            round(self.tv_weight, 3),
            int(self.num_steps),
            int(self.output_size),
            int(self.seed),
        )


@dataclass
class StylizeResult:
    image: Image.Image
    history: list[dict[str, float]] = field(default_factory=list)
    elapsed: float = 0.0
    content_shape: tuple[int, int] = (0, 0)
    style_shape: tuple[int, int] = (0, 0)
    layer_weights: tuple[float, ...] = ()
    device_name: str = ""


class _SynthesizedImage(nn.Module):
    """待优化的合成图像，与 notebook 中的 SynthesizedImage 等价。"""

    def __init__(self, image_shape: Sequence[int]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*image_shape))

    def forward(self) -> torch.Tensor:
        return self.weight


class StyleTransferEngine:
    """加载一次 VGG19，随后可反复执行风格迁移。"""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用的 CUDA，已无法使用 cuda 设备。")

        # 只保留到 relu5_1，去掉无用的后半部分 VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = vgg.features

        layers: list[nn.Module] = []
        for index in range(max(CONTENT_LAYERS + STYLE_LAYERS) + 1):
            layer = features[index]
            if isinstance(layer, nn.MaxPool2d):
                # 平均池化比最大池化产生的风格迁移结果更柔和
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            elif isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            layers.append(layer)

        self.net = nn.Sequential(*layers).to(self.device).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

        del features, vgg

        self.mean = torch.tensor(RGB_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(RGB_STD, device=self.device).view(1, 3, 1, 1)
        self.min_value = (0.0 - self.mean) / self.std
        self.max_value = (1.0 - self.mean) / self.std

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            index = self.device.index or 0
            return torch.cuda.get_device_name(index)
        return "CPU"

    def _preprocess(self, image: Image.Image, shape: tuple[int, int]) -> torch.Tensor:
        height, width = shape
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        tensor = TF.pil_to_tensor(resized).float().div_(255.0)
        tensor = TF.normalize(tensor, mean=RGB_MEAN, std=RGB_STD)
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        image = tensor.detach()[0].to("cpu")
        image = image * self.std.cpu().squeeze(0) + self.mean.cpu().squeeze(0)
        image = image.clamp_(0.0, 1.0)
        return TF.to_pil_image(image)

    def _extract(self, tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        contents: list[torch.Tensor] = []
        styles: list[torch.Tensor] = []
        current = tensor
        for index, layer in enumerate(self.net):
            current = layer(current)
            if index in STYLE_LAYERS:
                styles.append(current)
            if index in CONTENT_LAYERS:
                contents.append(current)
        return contents, styles

    @staticmethod
    def _gram(tensor: torch.Tensor) -> torch.Tensor:
        channels = tensor.shape[1]
        positions = tensor.numel() // channels
        features = tensor.reshape(channels, positions)
        return features @ features.t() / (channels * positions)

    @staticmethod
    def _tv_loss(tensor: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
            + torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
        )

    def stylize(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        params: StyleParams | None = None,
        progress: ProgressCallback | None = None,
    ) -> StylizeResult:
        """执行风格迁移并返回结果。

        progress(done_steps, total_steps, preview_image) 会在每个外层轮次结束时调用，
        preview_image 仅在需要刷新预览的轮次非空。
        """
        params = params or StyleParams()
        if params.num_steps < 1:
            raise ValueError("num_steps 必须 >= 1。")
        if params.output_size < 128:
            raise ValueError("output_size 过小。")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()

        content_shape = fit_shape(content_image, params.output_size)
        style_shape = fit_shape(style_image, params.style_input_size)

        torch.manual_seed(int(params.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(params.seed))

        content_x = self._preprocess(content_image, content_shape)
        style_x = self._preprocess(style_image, style_shape)

        with torch.no_grad():
            contents_y, _ = self._extract(content_x)
            _, styles_y = self._extract(style_x)
            styles_y_gram = [self._gram(feature) for feature in styles_y]
        del style_x

        # 以内容图为主、混入少量噪声作为初始化（notebook 的核心调整 4）
        generated = _SynthesizedImage(content_x.shape).to(self.device)
        noise = torch.randn_like(content_x)
        with torch.no_grad():
            generated.weight.copy_(0.7 * content_x + 0.3 * noise)

        layer_weights = params.layer_weights()
        optimizer = torch.optim.LBFGS(
            [generated.weight],
            lr=float(params.learning_rate),
            max_iter=int(params.max_iter),
            line_search_fn="strong_wolfe",
        )

        history: list[dict[str, float]] = []
        latest = {"content": 0.0, "style": 0.0, "tv": 0.0, "total": 0.0}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                generated.weight.clamp_(self.min_value, self.max_value)

            current = generated()
            contents_y_hat, styles_y_hat = self._extract(current)

            content_loss = sum(
                torch.square(predicted - target.detach()).mean()
                for predicted, target in zip(contents_y_hat, contents_y, strict=True)
            ) * float(params.content_weight)

            style_loss = sum(
                torch.square(self._gram(predicted) - target.detach()).mean()
                * float(params.style_weight)
                * weight
                for predicted, target, weight in zip(
                    styles_y_hat, styles_y_gram, layer_weights, strict=True
                )
            )

            tv_loss = self._tv_loss(current) * float(params.tv_weight)
            total_loss = content_loss + style_loss + tv_loss

            total_loss.backward()

            latest["content"] = float(content_loss.detach())
            latest["style"] = float(style_loss.detach())
            latest["tv"] = float(tv_loss.detach())
            latest["total"] = float(total_loss.detach())
            return total_loss

        for step in range(int(params.num_steps)):
            optimizer.step(closure)

            with torch.no_grad():
                generated.weight.clamp_(self.min_value, self.max_value)

            record = {
                "step": step + 1,
                "content": latest["content"],
                "style": latest["style"],
                "tv": latest["tv"],
                "total": latest["total"],
            }
            history.append(record)

            if progress is not None:
                should_preview = (
                    params.preview_every > 0
                    and (
                        (step + 1) % int(params.preview_every) == 0
                        or step + 1 == int(params.num_steps)
                    )
                )
                preview = self._postprocess(generated.weight) if should_preview else None
                progress(step + 1, int(params.num_steps), preview)

        final_image = self._postprocess(generated.weight)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        return StylizeResult(
            image=final_image,
            history=history,
            elapsed=elapsed,
            content_shape=content_shape,
            style_shape=style_shape,
            layer_weights=tuple(layer_weights),
            device_name=self.device_name,
        )
