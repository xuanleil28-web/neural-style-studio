# Neural Style Studio

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**An interactive, reproducible neural style transfer app.** Upload a content photo and a
style painting, tune the color and brushstroke parameters with sliders, and download the
stylized result. Built with VGG19, Gram matrices, and L-BFGS — packaged from a course
notebook into a small application that other people can run.

[中文说明](README.zh-CN.md)

![Content, style, and result](docs/images/example_triptych.png)

> Content: NASA astronaut portrait (public domain). Style: Vincent van Gogh,
> *Wheat Field with Cypresses* (1889, public domain). Result: default settings,
> 40 rounds, 512 px. See [asset credits](assets/CREDITS.md).

## Why this project

Most neural style transfer demos are either a notebook with hard-coded paths or a
black-box web service. This project keeps the original optimization-based algorithm but
makes it:

- **Interactive** — every important loss weight and input scale is a slider.
- **Inspectable** — the engine is a standalone Python module, not buried in UI code.
- **Reproducible** — fixed seed, documented defaults, tested environment, and a CLI.
- **Portable** — runs locally, in Docker, or on CPU; no `d2l` or `matplotlib` dependency.
- **Reusable** — the demo images are public domain, and the code is MIT licensed.

## Features

- Upload content/style images (PNG, JPG, JPEG, WEBP, BMP); transparent images are
  composited on white.
- 7 core sliders: style strength, color strength, brush strength, brush scale,
  content preservation, TV smoothness, and optimization rounds.
- 4 one-click presets: Van Gogh impasto, color burst, fine brush, and quick preview.
- Live progress, intermediate previews, final loss chart, and parameter report.
- Download the result as PNG.
- Automatic CUDA acceleration; CPU fallback with a warning.
- Headless CLI and Docker support.
- Unit tests plus an optional real-engine smoke test.
- GitHub Actions CI for lint, tests, and Docker build.

## Quick start

### Requirements

- Python 3.10 or newer
- PyTorch 2.0+ and torchvision 0.15+
- Optional: an NVIDIA GPU with a CUDA-enabled PyTorch build
- ~600 MB free disk space for the VGG19 weights on first run

### Local install

```bash
git clone https://github.com/<your-account>/neural-style-studio.git
cd neural-style-studio

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# CPU-only PyTorch (replace with a CUDA wheel from pytorch.org if you have a GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py
```

Open <http://127.0.0.1:8501>.

On Windows you can also double-click `run_app.bat`. On macOS/Linux:

```bash
chmod +x run_app.sh
./run_app.sh
```

The scripts prefer `.venv` and otherwise use `python`/`python3`.

The first run downloads the 548 MB VGG19 checkpoint to
`~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` and caches it for later runs.

### Docker (CPU)

```bash
docker compose up --build
# open http://127.0.0.1:8501
```

or without Compose:

```bash
docker build -t neural-style-studio .
docker run --rm -p 8501:8501 \
  -v neural-style-cache:/home/appuser/.cache/torch \
  neural-style-studio
```

The Docker image installs CPU-only PyTorch to keep it smaller. For GPU inference,
install a CUDA-enabled PyTorch locally (or adapt the base image and install the matching
CUDA wheel). CPU inference works but is significantly slower.

### CLI

```bash
# Quick preview: 10 rounds, 384 px
python scripts/stylize.py --steps 10 --size 384 --output outputs/preview.png

# Stronger color, coarser brush
python scripts/stylize.py --steps 30 --size 512 \
  --color 1.5 --brush 1.2 --brush-scale 0.7 \
  --output outputs/colorful.png

# Force CPU
python scripts/stylize.py --device cpu --steps 5 --size 256
```

Run `python scripts/stylize.py --help` for all options.

### Check your environment

```bash
python scripts/check_env.py
```

It prints Python, torch, torchvision, Streamlit, CUDA availability, GPU name, and
whether the VGG19 checkpoint and demo assets are present.

## What the sliders do

The UI is currently in Simplified Chinese. The table below maps each label to English.

| UI label | English | Range / default | Engine parameter | Effect |
| --- | --- | --- | --- | --- |
| 风格强度 | Style strength | 0.10–2.00 / 1.00 | `style_weight = 3e6 × value` | Global style intensity; too high can overpower content structure |
| 色彩强度 | Color strength | 0.00–2.00 / 1.00 | multiplier on `relu1_1` / `relu2_1` weights | Shallow features carry color; raise it to move closer to the style palette |
| 笔触强度 | Brush strength | 0.00–2.00 / 1.00 | multiplier on `relu4_1` / `relu5_1` weights | Deep features carry large-scale texture; raise it for impasto/swirl strokes |
| 笔触尺度 | Brush scale | 0.00–1.00 / 0.55 | style input long edge, 512 → 224 px | Smaller style input means larger effective brush strokes |
| 内容保留 | Content preservation | 0.05–2.00 / 0.50 | `content_weight` | Higher keeps mountains, roads, faces, and other structure clearer |
| 画面平滑 | Smoothness | 0.00–1.00 / 0.20 | `tv_weight` | Higher is smoother; lower keeps grain/noise |
| 迭代轮数 | Optimization rounds | 10–60 / 40 | `num_steps` | Each round runs 10 L-BFGS inner iterations |
| 输出长边 | Output long edge | 384 / 512 / 640 px | `output_size` | Output resolution; lower it if VRAM is tight |
| 随机种子 | Random seed | 0–999999 / 42 | `seed` | Controls the noise initialization; fixed seed improves reproducibility |

The five style layers use the notebook's base weights:

```text
relu1_1  relu2_1  relu3_1  relu4_1  relu5_1
  0.05     0.10     0.50     2.50     1.00
```

`relu3_1` sits between color and texture, so it is mixed as
`0.35 × color_strength + 0.65 × brush_strength`.

### Parameter examples

![Parameter comparison](docs/images/parameter_effects.png)

Generated with the same content/style pair at 15 rounds and 384 px. The top row changes
only color strength; the bottom row changes only brush scale. This is a good way to see
that the two controls affect different parts of the style representation.

### Presets

| Preset (UI) | English | Intent |
| --- | --- | --- |
| 梵高·厚涂 | Van Gogh impasto | Stronger deep-layer weights and a smaller style input for bold strokes |
| 色彩冲击 | Color burst | Higher color weight and global style strength |
| 细腻笔触 | Fine brush | Larger style input, fewer rounds, more content preservation |
| 快速预览 | Quick preview | 15 rounds at 384 px for fast iteration |

## How it works

1. A pretrained VGG19 extracts features. The implementation keeps layers up to `relu5_1`,
   uses `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1` as style layers, and
   `relu4_2` as the content layer.
2. Max-pooling is replaced with average-pooling, and ReLUs are made non-inplace.
3. Style is represented by Gram matrices:

   ```text
   G = F Fᵀ / (C × N)
   ```

4. The synthesized image minimizes:

   ```text
   L = content_weight × L_content
     + style_weight × Σ layer_weight_l × L_style_l
     + tv_weight × L_TV
   ```

   Both `L_content` and `L_style` are mean squared errors; `L_TV` is total variation.
5. The image starts at `0.7 × content + 0.3 × Gaussian noise` and is optimized in
   normalized VGG input space with L-BFGS (`strong_wolfe`, 10 inner iterations per round).
6. Pixels are clamped to the valid normalized range before each closure and after each
   round. The result is denormalized and converted back to PIL.

This is the original optimization-based style transfer formulation: each output image
requires its own optimization. It is not a feed-forward stylization network, so generation
takes seconds rather than milliseconds.

## Reproducibility

### Tested environment

| Component | Version |
| --- | --- |
| OS | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Streamlit | 1.45.1 |
| Pillow | 11.1.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB |

The code also runs on CPU. CI uses Python 3.10 and 3.12 with CPU PyTorch.

### Performance

Approximate timings on the machine above:

| Configuration | Time |
| --- | --- |
| 10 rounds, 384 px | ~3 s |
| 15 rounds, 512 px (square) | ~10 s |
| 40 rounds, 512 px (square) | ~27 s |
| 40 rounds, 512 px (portrait 288×512) | ~18 s |

CPU inference is much slower (minutes for 40 rounds). Reduce rounds or resolution first.

### Determinism

`seed` fixes the noise initialization and is exposed in the UI and CLI. Results are
visually reproducible on the same hardware and library versions, but GPU convolution
algorithms are not guaranteed to be bit-exact across drivers, GPUs, or PyTorch versions.

The VGG19 checkpoint is identified by torchvision's `VGG19_Weights.DEFAULT` and is
downloaded to the standard torch cache directory.

## Project structure

```text
neural-style-studio/
├─ app.py                      # Streamlit UI, sliders, presets, result rendering
├─ style_transfer.py           # Standalone VGG19 + L-BFGS engine
├─ assets/
│  ├─ demo_content.png         # NASA astronaut, public domain
│  ├─ demo_style.png           # Van Gogh, public domain
│  └─ CREDITS.md
├─ scripts/
│  ├─ stylize.py               # Headless CLI
│  ├─ check_env.py             # Environment diagnostic
│  └─ make_docs_assets.py      # Regenerates README images
├─ tests/                      # pytest suite
├─ docs/images/                # README images
├─ .github/workflows/ci.yml    # lint + tests + Docker build
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ requirements-dev.txt
├─ pyproject.toml
└─ LICENSE
```

## Development

```bash
pip install -r requirements-dev.txt

pytest -q          # fast tests, skips the model download
ruff check .       # lint

# Real engine smoke test (loads VGG19, runs one optimization round)
# Windows PowerShell:
$env:RUN_ENGINE_TESTS = "1"; pytest -q
# macOS/Linux:
RUN_ENGINE_TESTS=1 pytest -q

# Regenerate README images (GPU recommended)
python scripts/make_docs_assets.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup details and PR expectations.

## Troubleshooting

**`No module named streamlit` / `No module named torch`**
You are probably using a different interpreter than the one where dependencies were
installed. Activate `.venv` or use `python -m streamlit run app.py`.

**First run seems stuck**
The VGG19 checkpoint is 548 MB. Check the download progress in the terminal. If your
network cannot reach PyTorch's servers, download `vgg19-dcbb9e9d.pth` manually and place
it in `~/.cache/torch/hub/checkpoints/`.

**CUDA out of memory**
Lower the output long edge to 384 px, reduce rounds, or close other GPU processes. The
error dialog also suggests this.

**Port 8501 is already in use**
Use another port:

```bash
streamlit run app.py --server.port 8502
```

**The result is too smooth or too noisy**
Increase `画面平滑` (TV weight) for smoother output, or decrease it for more grain.
Increase `内容保留` if structure is lost; increase `笔触强度` / `笔触尺度` for bolder
strokes.

**Docker build is slow**
The CPU PyTorch wheel is large. The first build downloads it once; subsequent builds
reuse Docker layer cache.

## License and credits

- Code: [MIT](LICENSE).
- Demo content: NASA astronaut portrait, public domain.
- Demo style: Vincent van Gogh, *Wheat Field with Cypresses*, 1889, public domain.
- Documentation images are generated from those assets.
- See [assets/CREDITS.md](assets/CREDITS.md) for details. The demo images are not
  covered by the MIT license.

Algorithm inspired by the d2l *Style Transfer* notebook and the original neural style
transfer work:

- Gatys, Ecker, and Bethge. *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.
- Simonyan and Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015.

Built with [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/),
[Streamlit](https://streamlit.io/), and [Pillow](https://python-pillow.org/).

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
first, and do not add image assets without a documented license.
