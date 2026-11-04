# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-04

### Added

- Video style transfer page with first-frame preview and MP4 download.
- Farneback forward/backward optical flow with an occlusion confidence mask.
- Temporal consistency loss and optical-flow warm start between frames.
- Video controls: first-frame rounds, per-frame rounds, temporal weight, max frames,
  resolution, flow quality, occlusion sensitivity, and random seed.
- H.264 encoding through `imageio-ffmpeg`, with OpenCV `mp4v` fallback.
- Shared engine cache and Streamlit navigation for image/video pages.
- Video parameter, render, and real-engine smoke tests.
- Video demo GIF and updated documentation.

### Changed

- `StyleTransferEngine` now shares its optimization loop between images and video.
- Upload limit increased to 200 MB for short video clips.

## [0.1.0] - 2026-11-03

### Added

- Initial Streamlit application for VGG19 + Gram matrix + L-BFGS style transfer.
- Content/style image upload with built-in public-domain demo assets.
- Sliders for style strength, color strength, brush strength, brush scale,
  content preservation, TV smoothness, and optimization rounds.
- Four presets: Van Gogh impasto, color burst, fine brush, and quick preview.
- Live progress, intermediate previews, loss chart, PNG download, and parameter report.
- Reusable `StyleTransferEngine` and `StyleParams` API.
- Headless CLI (`scripts/stylize.py`) and environment diagnostic (`scripts/check_env.py`).
- Tests for image handling, shape fitting, and parameter mapping.
- Dockerfile and Docker Compose setup.
- GitHub Actions CI for linting, tests, and Docker build.
