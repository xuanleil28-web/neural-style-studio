from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_without_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()

    assert not app.exception
    assert len(app.slider) == 7
    assert len(app.select_slider) == 1
    assert len(app.number_input) == 1
    labels = [button.label for button in app.button]
    assert "开始风格化" in labels
    assert "快速预览" in labels
