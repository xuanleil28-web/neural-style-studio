# SPDX-License-Identifier: MIT
"""VGG19 + L-BFGS neural style transfer engine for images and short videos."""




from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF

try:  # OpenCV is only needed for video
    import cv2
except ImportError:  # pragma: no cover - exercised only without OpenCV
    cv2 = None

try:  # Prefer bundled ffmpeg for browser-compatible H.264
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - fallback to OpenCV mp4v
    imageio_ffmpeg = None

__version__ = "0.2.0"

# ImageNet normalization used by the pretrained VGG19.
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

# VGG19 layers used for style: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1.
STYLE_LAYERS: tuple[int, ...] = (1, 6, 11, 20, 29)
# Content layer: relu4_2.
CONTENT_LAYERS: tuple[int, ...] = (22,)

# Base style-layer weights from the notebook: shallow layers carry color, deep layers carry large-scale strokes.
BASE_STYLE_LAYER_WEIGHTS: tuple[float, ...] = (0.05, 0.10, 0.50, 2.50, 1.00)

ProgressCallback = Callable[[int, int, Image.Image | None], None]
VideoProgressCallback = Callable[[str, int, int, Image.Image | None], None]

FLOW_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": {"levels": 2, "winsize": 11, "iterations": 2, "poly_n": 5, "poly_sigma": 1.1},
    "balanced": {"levels": 3, "winsize": 15, "iterations": 3, "poly_n": 5, "poly_sigma": 1.2},
    "quality": {"levels": 4, "winsize": 21, "iterations": 4, "poly_n": 7, "poly_sigma": 1.5},
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "Video requires OpenCV. Install opencv-python-headless (see requirements.txt)."
        )


def open_rgb_image(source) -> Image.Image:
    """Open an image, apply EXIF orientation, flatten alpha, and return RGB."""
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
    """Resize to keep the long edge <= max_size; return (height, width)."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(1.0, max_size / float(max(width, height)))
    if min(width, height) * scale < min_side:
        scale = min_side / float(min(width, height))

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_height, new_width


def get_video_shape(width: int, height: int, max_size: int = 480) -> tuple[int, int]:
    """Return even (height, width) for video encoding, with a minimum short side."""

    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions.")
    scale = min(1.0, max_size / float(max(width, height)))
    if min(width, height) * scale < 16:
        scale = 16.0 / float(min(width, height))

    new_width = max(2, int(round(width * scale)) // 2 * 2)
    new_height = max(2, int(round(height * scale)) // 2 * 2)
    return new_height, new_width


@dataclass(frozen=True)
class StyleParams:
    """All tunable parameters for one stylization run."""


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
        """Map brush scale to the style-image long edge (512 -> 224 px)."""

        return int(round(512 - 288 * _clamp(self.brush_scale, 0.0, 1.0)))

    def layer_weights(self) -> list[float]:
        """Redistribute the five style-layer weights from color/brush sliders."""
        color = max(0.0, float(self.color_strength))
        brush = max(0.0, float(self.brush_strength))

        weights = [
            BASE_STYLE_LAYER_WEIGHTS[0] * color,
            BASE_STYLE_LAYER_WEIGHTS[1] * color,
            # relu3_1 blends color and texture, so mix both controls.
            BASE_STYLE_LAYER_WEIGHTS[2] * (0.35 * color + 0.65 * brush),
            BASE_STYLE_LAYER_WEIGHTS[3] * brush,
            BASE_STYLE_LAYER_WEIGHTS[4] * brush,
        ]
        # Avoid an all-zero style loss when both controls are zero.
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


@dataclass(frozen=True)
class VideoInfo:
    """Result of probing a video; the first frame is resized for processing."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    first_frame: Image.Image


@dataclass(frozen=True)
class VideoParams:
    """Video parameters; style.num_steps/output_size are overridden by video fields."""

    style: StyleParams = field(default_factory=StyleParams)
    max_frames: int = 36
    video_size: int = 480
    first_frame_steps: int = 35
    frame_steps: int = 8
    temporal_weight: float = 200.0
    flow_preset: str = "balanced"
    occlusion_sensitivity: float = 0.2
    preview_every: int = 5

    def flow_settings(self) -> dict[str, float | int]:
        return dict(FLOW_PRESETS.get(self.flow_preset, FLOW_PRESETS["balanced"]))

    def signature(self) -> tuple:
        return (
            self.style.signature(),
            int(self.max_frames),
            int(self.video_size),
            int(self.first_frame_steps),
            int(self.frame_steps),
            round(self.temporal_weight, 3),
            self.flow_preset,
            round(self.occlusion_sensitivity, 3),
        )


@dataclass
class VideoResult:
    video_bytes: bytes
    first_frame: Image.Image
    width: int
    height: int
    fps: float
    frame_count: int
    elapsed: float
    codec: str
    history: list[dict[str, float]] = field(default_factory=list)
    device_name: str = ""
    temporal_weight: float = 0.0
    source_frame_count: int = 0


class _SynthesizedImage(nn.Module):
    """Image being optimized, equivalent to SynthesizedImage in the notebook."""

    def __init__(self, image_shape: Sequence[int]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*image_shape))

    def forward(self) -> torch.Tensor:
        return self.weight


class _VideoSink:
    """Write H.264 via imageio-ffmpeg, falling back to OpenCV mp4v."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self._generator = None
        self._writer = None
        self.codec = "mp4v"

        if imageio_ffmpeg is not None:
            try:
                generator = imageio_ffmpeg.write_frames(
                    str(path),
                    (width, height),
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    fps=fps,
                    codec="libx264",
                    quality=6,
                    macro_block_size=2,
                    output_params=["-movflags", "+faststart"],
                    ffmpeg_log_level="error",
                )
                generator.send(None)
                self._generator = generator
                self.codec = "H.264"
                return
            except Exception:
                self._generator = None

        if cv2 is None:
            raise RuntimeError("Video encoding requires OpenCV or imageio-ffmpeg.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create a video encoder.")
        self._writer = writer

    def write(self, rgb_frame: np.ndarray) -> None:
        if self._generator is not None:
            self._generator.send(np.ascontiguousarray(rgb_frame).tobytes())
        elif self._writer is not None:
            self._writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._generator is not None:
            try:
                self._generator.close()
            finally:
                self._generator = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def probe_video(path: str | Path, max_size: int = 480) -> VideoInfo:
    """Read basic video metadata and the first frame at processing resolution."""
    _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("The video contains no readable frames.")

        target_h, target_w = get_video_shape(width, height, max_size)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        first_frame = Image.fromarray(rgb, mode="RGB")
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            first_frame=first_frame,
        )
    finally:
        capture.release()


class StyleTransferEngine:
    """Load VGG19 once and reuse it for image or video stylization."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the cuda device cannot be used.")

        # Keep only layers up to relu5_1; drop the unused tail of VGG19.
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = vgg.features

        layers: list[nn.Module] = []
        for index in range(max(CONTENT_LAYERS + STYLE_LAYERS) + 1):
            layer = features[index]
            if isinstance(layer, nn.MaxPool2d):
                # Average pooling produces softer results than max pooling.
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

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    def _preprocess(self, image: Image.Image, shape: tuple[int, int]) -> torch.Tensor:
        height, width = shape
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        tensor = TF.pil_to_tensor(resized).float().div_(255.0)
        tensor = TF.normalize(tensor, mean=RGB_MEAN, std=RGB_STD)
        return tensor.unsqueeze(0).to(self.device)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        image = tensor.detach()[0].to("cpu").permute(1, 2, 0)
        mean = self.mean.cpu().flatten().view(1, 1, 3)
        std = self.std.cpu().flatten().view(1, 1, 3)
        image = image * std + mean
        image = image.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        return np.ascontiguousarray(image.numpy())

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        return Image.fromarray(self._tensor_to_numpy(tensor), mode="RGB")

    def _numpy_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        return (tensor - self.mean) / self.std

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

    def _prepare_style(
        self, style_image: Image.Image, style_input_size: int
    ) -> tuple[list[torch.Tensor], tuple[int, int]]:
        style_shape = fit_shape(style_image, style_input_size)
        style_x = self._preprocess(style_image, style_shape)
        with torch.no_grad():
            _, styles_y = self._extract(style_x)
            style_grams = [self._gram(feature) for feature in styles_y]
        return style_grams, style_shape

    def _content_features(self, content_x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            contents_y, _ = self._extract(content_x)
        return contents_y

    def _optimize(
        self,
        content_x: torch.Tensor,
        contents_y: list[torch.Tensor],
        style_grams: list[torch.Tensor],
        params: StyleParams,
        steps: int,
        init: torch.Tensor,
        *,
        temporal: tuple[torch.Tensor, torch.Tensor] | None = None,
        temporal_weight: float = 0.0,
        progress: ProgressCallback | None = None,
        preview_every: int = 0,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        """Run L-BFGS optimization from a given normalized initialization."""
        generated = _SynthesizedImage(content_x.shape).to(self.device)
        with torch.no_grad():
            generated.weight.copy_(init)

        layer_weights = params.layer_weights()
        optimizer = torch.optim.LBFGS(
            [generated.weight],
            lr=float(params.learning_rate),
            max_iter=int(params.max_iter),
            line_search_fn="strong_wolfe",
        )

        history: list[dict[str, float]] = []
        latest = {"content": 0.0, "style": 0.0, "tv": 0.0, "temporal": 0.0, "total": 0.0}

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
                    styles_y_hat, style_grams, layer_weights, strict=True
                )
            )

            tv_loss = self._tv_loss(current) * float(params.tv_weight)
            total_loss = content_loss + style_loss + tv_loss

            temporal_loss = torch.zeros((), device=self.device)
            if temporal is not None and temporal_weight > 0:
                warped_prev, mask = temporal
                temporal_loss = (
                    torch.mean(mask * torch.square(current - warped_prev.detach()))
                    * float(temporal_weight)
                )
                total_loss = total_loss + temporal_loss

            total_loss.backward()

            latest["content"] = float(content_loss.detach())
            latest["style"] = float(style_loss.detach())
            latest["tv"] = float(tv_loss.detach())
            latest["temporal"] = float(temporal_loss.detach())
            latest["total"] = float(total_loss.detach())
            return total_loss

        for step in range(int(steps)):
            optimizer.step(closure)

            with torch.no_grad():
                generated.weight.clamp_(self.min_value, self.max_value)

            history.append(
                {
                    "step": step + 1,
                    "content": latest["content"],
                    "style": latest["style"],
                    "tv": latest["tv"],
                    "temporal": latest["temporal"],
                    "total": latest["total"],
                }
            )

            if progress is not None:
                should_preview = (
                    preview_every > 0
                    and ((step + 1) % int(preview_every) == 0 or step + 1 == int(steps))
                )
                preview = self._postprocess(generated.weight) if should_preview else None
                progress(step + 1, int(steps), preview)

        return generated.weight.detach(), history

    def stylize(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        params: StyleParams | None = None,
        progress: ProgressCallback | None = None,
    ) -> StylizeResult:
        """Stylize an image and return the result."""

        params = params or StyleParams()
        if params.num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if params.output_size < 128:
            raise ValueError("output_size is too small.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()

        content_shape = fit_shape(content_image, params.output_size)
        content_x = self._preprocess(content_image, content_shape)
        contents_y = self._content_features(content_x)
        style_grams, style_shape = self._prepare_style(
            style_image, params.style_input_size
        )

        self._set_seed(params.seed)
        noise = torch.randn_like(content_x)
        init = 0.7 * content_x + 0.3 * noise

        optimized, history = self._optimize(
            content_x,
            contents_y,
            style_grams,
            params,
            steps=int(params.num_steps),
            init=init,
            progress=progress,
            preview_every=int(params.preview_every),
        )
        final_image = self._postprocess(optimized)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        return StylizeResult(
            image=final_image,
            history=history,
            elapsed=elapsed,
            content_shape=content_shape,
            style_shape=style_shape,
            layer_weights=tuple(params.layer_weights()),
            device_name=self.device_name,
        )

    def preview_video_first_frame(
        self,
        video_path: str | Path,
        style_image: Image.Image,
        params: VideoParams,
        progress: ProgressCallback | None = None,
    ) -> StylizeResult:
        """Render only the first video frame for preview."""
        _require_cv2()
        if params.first_frame_steps < 1:
            raise ValueError("first_frame_steps must be >= 1.")
        if params.video_size < 16:
            raise ValueError("video_size is too small.")
        info = probe_video(video_path, params.video_size)
        content_image = info.first_frame
        content_shape = (content_image.height, content_image.width)
        content_x = self._preprocess(content_image, content_shape)
        contents_y = self._content_features(content_x)
        style_grams, style_shape = self._prepare_style(
            style_image, params.style.style_input_size
        )

        self._set_seed(params.style.seed)
        noise = torch.randn_like(content_x)
        init = 0.7 * content_x + 0.3 * noise

        started_at = time.perf_counter()
        optimized, history = self._optimize(
            content_x,
            contents_y,
            style_grams,
            params.style,
            steps=int(params.first_frame_steps),
            init=init,
            progress=progress,
            preview_every=int(params.style.preview_every),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return StylizeResult(
            image=self._postprocess(optimized),
            history=history,
            elapsed=time.perf_counter() - started_at,
            content_shape=content_shape,
            style_shape=style_shape,
            layer_weights=tuple(params.style.layer_weights()),
            device_name=self.device_name,
        )

    def _flow_warp(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        prev_stylized_rgb: np.ndarray,
        params: VideoParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp the previous stylized frame with optical flow and return a confidence mask."""
        settings = params.flow_settings()
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        flow_bw = cv2.calcOpticalFlowFarneback(
            curr_gray,
            prev_gray,
            None,
            0.5,
            int(settings["levels"]),
            int(settings["winsize"]),
            int(settings["iterations"]),
            int(settings["poly_n"]),
            float(settings["poly_sigma"]),
            0,
        )
        flow_fw = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5,
            int(settings["levels"]),
            int(settings["winsize"]),
            int(settings["iterations"]),
            int(settings["poly_n"]),
            float(settings["poly_sigma"]),
            0,
        )

        height, width = flow_bw.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = (grid_x + flow_bw[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow_bw[:, :, 1]).astype(np.float32)

        warped = cv2.remap(
            prev_stylized_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        warped_flow_fw = cv2.remap(
            flow_fw,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        flow_diff = np.linalg.norm(flow_bw + warped_flow_fw, axis=2)
        flow_diff = np.nan_to_num(flow_diff, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.exp(-float(params.occlusion_sensitivity) * flow_diff).astype(np.float32)
        return warped, np.clip(mask, 0.0, 1.0)

    def process_video(
        self,
        video_path: str | Path,
        style_image: Image.Image,
        params: VideoParams,
        progress: VideoProgressCallback | None = None,
    ) -> VideoResult:
        """Stylize a video frame by frame and return browser-playable MP4 bytes."""
        _require_cv2()
        if params.max_frames < 1:
            raise ValueError("max_frames must be >= 1.")
        if params.first_frame_steps < 1 or params.frame_steps < 1:
            raise ValueError("first_frame_steps / frame_steps must be >= 1.")
        if params.video_size < 16:
            raise ValueError("video_size is too small.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fd, temp_name = tempfile.mkstemp(prefix="nss_video_", suffix=".mp4")
        os.close(fd)
        temp_path = Path(temp_name)
        sink: _VideoSink | None = None
        try:
            source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
            fps = max(1.0, min(60.0, fps))
            source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_h, frame_w = get_video_shape(source_w, source_h, params.video_size)
            total = int(params.max_frames)
            if source_frames > 0:
                total = min(total, source_frames)
            total = max(1, total)

            style_grams, _ = self._prepare_style(
                style_image, params.style.style_input_size
            )
            sink = _VideoSink(temp_path, frame_w, frame_h, fps)

            prev_bgr: np.ndarray | None = None
            prev_stylized: np.ndarray | None = None
            first_frame: Image.Image | None = None
            history: list[dict[str, float]] = []
            processed = 0

            if progress is not None:
                progress("first_frame", 0, total, None)

            for frame_index in range(total):
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    break
                frame_bgr = cv2.resize(
                    frame_bgr, (frame_w, frame_h), interpolation=cv2.INTER_AREA
                )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                content_x = self._numpy_to_tensor(frame_rgb)
                contents_y = self._content_features(content_x)

                if frame_index == 0 or prev_bgr is None or prev_stylized is None:
                    self._set_seed(params.style.seed)
                    noise = torch.randn_like(content_x)
                    init = 0.7 * content_x + 0.3 * noise
                    temporal = None
                    steps = int(params.first_frame_steps)
                else:
                    warped_rgb, mask_np = self._flow_warp(
                        prev_bgr, frame_bgr, prev_stylized, params
                    )
                    warped_tensor = self._numpy_to_tensor(warped_rgb)
                    mask_tensor = torch.from_numpy(mask_np).to(self.device)
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                    self._set_seed(int(params.style.seed) + frame_index)
                    noise = torch.randn_like(content_x)
                    blended = mask_tensor * warped_tensor + (1.0 - mask_tensor) * content_x
                    init = 0.95 * blended + 0.05 * noise
                    temporal = (warped_tensor, mask_tensor)
                    steps = int(params.frame_steps)

                optimized, frame_history = self._optimize(
                    content_x,
                    contents_y,
                    style_grams,
                    params.style,
                    steps=steps,
                    init=init,
                    temporal=temporal,
                    temporal_weight=float(params.temporal_weight),
                )
                out_rgb = self._tensor_to_numpy(optimized)
                sink.write(out_rgb)
                processed += 1

                if first_frame is None:
                    first_frame = Image.fromarray(out_rgb.copy(), mode="RGB")

                for record in frame_history:
                    history.append({**record, "frame": processed})
                prev_bgr = frame_bgr.copy()
                prev_stylized = out_rgb

                if progress is not None:
                    should_preview = (
                        params.preview_every > 0
                        and (
                            processed % int(params.preview_every) == 0
                            or processed == total
                        )
                    )
                    preview = Image.fromarray(out_rgb, mode="RGB") if should_preview else None
                    progress("frame", processed, total, preview)

            if processed == 0 or first_frame is None:
                raise ValueError("The video contains no frames to process.")

            if progress is not None:
                progress("encode", processed, total, None)
            codec = sink.codec
            sink.close()
            sink = None
            video_bytes = temp_path.read_bytes()
            if not video_bytes:
                raise RuntimeError("Video encoding produced no data.")

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return VideoResult(
                video_bytes=video_bytes,
                first_frame=first_frame,
                width=frame_w,
                height=frame_h,
                fps=fps,
                frame_count=processed,
                elapsed=time.perf_counter() - started_at,
                codec=codec,
                history=history,
                device_name=self.device_name,
                temporal_weight=float(params.temporal_weight),
                source_frame_count=source_frames,
            )
        finally:
            capture.release()
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    pass
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
