from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_video_page_renders_without_upload():
    app = AppTest.from_file(str(ROOT / "tests" / "video_page_app.py"), default_timeout=60)
    app.run()

    assert not app.exception
    assert len(app.slider) == 11
    assert len(app.select_slider) == 1
    assert len(app.selectbox) == 1
    labels = [button.label for button in app.button]
    assert "稳定" in labels
    assert "速度" in labels
    assert "质量" in labels
