#!/usr/bin/env python3
"""Print environment information relevant to reproducing the app."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> int:
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")

    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"PyTorch:    NOT AVAILABLE ({exc})")
        return 1

    print(f"torch:      {torch.__version__}")
    print(f"torchvision:{torchvision.__version__}")
    print(f"CUDA build: {torch.version.cuda or 'none'}")
    print(f"CUDA ready: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    try:
        import streamlit

        print(f"Streamlit:  {streamlit.__version__}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Streamlit:  NOT AVAILABLE ({exc})")

    try:
        import cv2

        print(f"OpenCV:     {cv2.__version__}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"OpenCV:     NOT AVAILABLE ({exc}) — video mode disabled")

    try:
        import imageio_ffmpeg

        print(f"ffmpeg:     {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"ffmpeg:     NOT AVAILABLE ({exc}) — H.264 encoding will fall back to mp4v")

    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "vgg19-dcbb9e9d.pth"
    print(f"VGG19 cache:{checkpoint} ({'found' if checkpoint.exists() else 'missing, will download'})")

    for asset in ("demo_content.png", "demo_style.png"):
        path = Path(__file__).resolve().parents[1] / "assets" / asset
        print(f"Asset:      {path} ({'found' if path.exists() else 'missing'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
