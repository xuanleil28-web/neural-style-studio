#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the images used in README.md (requires VGG19 weights)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BG = (247, 243, 236)
INK = (35, 33, 43)
MUTED = (109, 102, 119)

FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_triptych(content: Image.Image, style: Image.Image, result: Image.Image) -> Image.Image:
    cell, label_height, pad, gap = 512, 42, 24, 18
    canvas = Image.new("RGB", (pad * 2 + cell * 3 + gap * 2, pad * 2 + label_height + cell), BG)
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(17)
    caption_font = load_font(13)

    panels = (
        ("CONTENT", "NASA astronaut · public domain", content),
        ("STYLE", "Van Gogh · public domain", style),
        ("RESULT", "40 rounds · 512 px", result),
    )
    for index, (title, subtitle, image) in enumerate(panels):
        x = pad + index * (cell + gap)
        draw.text((x, pad), title, fill=INK, font=label_font)
        draw.text((x, pad + 21), subtitle, fill=MUTED, font=caption_font)
        thumbnail = ImageOps.contain(image.convert("RGB"), (cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(
            thumbnail,
            (x + (cell - thumbnail.width) // 2, pad + label_height + (cell - thumbnail.height) // 2),
        )
    return canvas


def make_parameter_grid(images: list[tuple[str, Image.Image]]) -> Image.Image:
    cell, label_height, pad, gap = 384, 34, 20, 16
    canvas = Image.new("RGB", (pad * 2 + cell * 2 + gap, pad * 2 + (label_height + cell) * 2 + gap), BG)
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)

    for index, (label, image) in enumerate(images):
        x = pad + (index % 2) * (cell + gap)
        y = pad + (index // 2) * (label_height + cell + gap)
        draw.text((x, y), label, fill=INK, font=font)
        canvas.paste(image.convert("RGB"), (x, y + label_height))
    return canvas


def main() -> int:
    from style_transfer import StyleParams, StyleTransferEngine, open_rgb_image

    assets = ROOT / "assets"
    docs = ROOT / "docs" / "images"
    docs.mkdir(parents=True, exist_ok=True)

    content = open_rgb_image(assets / "demo_content.png")
    style = open_rgb_image(assets / "demo_style.png")
    engine = StyleTransferEngine()
    print("Loaded", engine.device_name, flush=True)

    print("Rendering default result (40 rounds, 512 px)...", flush=True)
    default_result = engine.stylize(content, style, StyleParams(num_steps=40, output_size=512))
    default_result.image.save(docs / "example_result.png", optimize=True)

    triptych = make_triptych(content, style, default_result.image)
    triptych.save(docs / "example_triptych.png", optimize=True)

    print("Rendering parameter examples (15 rounds, 384 px)...", flush=True)
    base = StyleParams(num_steps=15, output_size=384, preview_every=0)
    variants = [
        ("Color strength 0.0", replace(base, color_strength=0.0)),
        ("Color strength 2.0", replace(base, color_strength=2.0)),
        ("Brush scale 0.0 (fine)", replace(base, brush_scale=0.0)),
        ("Brush scale 1.0 (coarse)", replace(base, brush_scale=1.0)),
    ]
    panels = []
    for label, params in variants:
        result = engine.stylize(content, style, params)
        panels.append((label, result.image))
        print("  done:", label, flush=True)

    grid = make_parameter_grid(panels)
    grid.save(docs / "parameter_effects.png", optimize=True)
    print("Saved docs images to", docs, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
