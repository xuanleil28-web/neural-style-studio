# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
