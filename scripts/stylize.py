#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Command-line style transfer, useful for reproducible headless runs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stylize a content image with a style image using VGG19 + L-BFGS."
    )
    parser.add_argument("--content", default=str(ROOT / "assets" / "demo_content.png"))
    parser.add_argument("--style", default=str(ROOT / "assets" / "demo_style.png"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "stylize.png"))
    parser.add_argument("--steps", type=int, default=40, help="outer L-BFGS rounds (default: 40)")
    parser.add_argument("--size", type=int, default=512, help="long edge of the output (default: 512)")
    parser.add_argument("--style-weight", type=float, default=3.0e6)
    parser.add_argument("--content-weight", type=float, default=0.5)
    parser.add_argument("--color", type=float, default=1.0, help="color strength (0-2)")
    parser.add_argument("--brush", type=float, default=1.0, help="brush strength (0-2)")
    parser.add_argument("--brush-scale", type=float, default=0.55, help="brush scale (0-1)")
    parser.add_argument("--tv", type=float, default=0.2, help="TV smoothness weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from style_transfer import StyleParams, StyleTransferEngine, open_rgb_image

    content_path = Path(args.content)
    style_path = Path(args.style)
    if not content_path.exists():
        raise SystemExit(f"Content image not found: {content_path}")
    if not style_path.exists():
        raise SystemExit(f"Style image not found: {style_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = None if args.device == "auto" else args.device
    engine = StyleTransferEngine(device=device)
    content = open_rgb_image(content_path)
    style = open_rgb_image(style_path)
    params = StyleParams(
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        color_strength=args.color,
        brush_strength=args.brush,
        brush_scale=args.brush_scale,
        tv_weight=args.tv,
        num_steps=args.steps,
        output_size=args.size,
        seed=args.seed,
    )

    print(f"Device: {engine.device_name}")
    print(f"Content: {content_path} ({content.size})")
    print(f"Style:   {style_path} ({style.size})")
    print(f"Params:  {params}")

    started = time.perf_counter()

    def progress(done: int, total: int, _preview) -> None:
        print(f"\rOptimizing: {done:3d}/{total} rounds", end="", flush=True)

    result = engine.stylize(content, style, params, progress=progress)
    result.image.save(output_path)

    print()
    print(f"Saved: {output_path} ({result.image.width}x{result.image.height})")
    print(f"Time:  {time.perf_counter() - started:.1f}s")
    print(f"Final loss: {result.history[-1]['total']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
