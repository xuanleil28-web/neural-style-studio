# Contributing

Thanks for taking the time to improve Neural Style Studio.

## Development setup

```bash
git clone https://github.com/<your-account>/neural-style-studio.git
cd neural-style-studio

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

For CUDA, install the PyTorch wheel that matches your driver instead of the CPU wheel.

## Run the app

```bash
streamlit run app.py
```

Or use `run_app.bat` / `./run_app.sh`.

## Tests and lint

```bash
pytest -q
ruff check .
```

The default test suite is fast and does not download model weights. To run the real
engine smoke tests (load VGG19; one image round and a short optical-flow video):

```bash
# Windows PowerShell
$env:RUN_ENGINE_TESTS = "1"; pytest -q

# macOS/Linux
RUN_ENGINE_TESTS=1 pytest -q
```

## Pull requests

- Keep changes focused and describe the user-visible behavior.
- Add or update tests for parameter mapping, image handling, or engine behavior.
- Run `pytest` and `ruff check .` before opening a PR.
- If you add demo images, include the source, author, and license in `assets/CREDITS.md`.
  Do not add images with unclear redistribution rights.
- Update `CHANGELOG.md` for user-visible changes.

## Code style

- Python 3.10+, type hints where useful.
- 4-space indentation, 100-character soft line limit.
- Keep the engine (`style_transfer.py`) independent from Streamlit so it can be used
  from scripts and tests.
- Prefer small, explicit functions over clever abstractions.

## Reporting issues

Use the GitHub issue templates. For bug reports, include:

- operating system and Python version,
- `python scripts/check_env.py` output,
- the parameters you used,
- the full error message or a screenshot.
