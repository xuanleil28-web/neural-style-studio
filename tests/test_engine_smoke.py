import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from style_transfer import StyleParams, StyleTransferEngine, VideoParams, probe_video

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("RUN_ENGINE_TESTS") != "1",
    reason="set RUN_ENGINE_TESTS=1 to load VGG19 and run a real optimization step",
)
def test_engine_runs_one_optimization_round():
    engine = StyleTransferEngine()
    content = Image.effect_noise((96, 72), 40).convert("RGB")
    style = Image.effect_noise((64, 64), 60).convert("RGB")

    result = engine.stylize(
        content,
        style,
        StyleParams(num_steps=1, output_size=128, preview_every=0),
    )

    assert result.image.mode == "RGB"
    assert result.image.size == (96, 72)
    assert len(result.history) == 1
    assert result.history[0]["total"] >= 0
    assert result.layer_weights == pytest.approx((0.05, 0.10, 0.50, 2.50, 1.00))


@pytest.mark.skipif(
    os.environ.get("RUN_ENGINE_TESTS") != "1",
    reason="set RUN_ENGINE_TESTS=1 to load VGG19 and run a real optimization step",
)
def test_video_pipeline_with_optical_flow():
    cv2 = pytest.importorskip("cv2")
    output_dir = ROOT / "outputs" / "test_video_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.mp4"

    try:
        writer = cv2.VideoWriter(
            str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 72)
        )
        assert writer.isOpened()
        for index in range(6):
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            frame[:, :, 1] = 20 + index * 20
            cv2.circle(frame, (20 + index * 8, 36), 14, (220, 120, 40), -1)
            writer.write(frame)
        writer.release()

        info = probe_video(input_path, 96)
        assert info.first_frame.size == (96, 72)

        engine = StyleTransferEngine()
        style = Image.effect_noise((64, 64), 50).convert("RGB")
        params = VideoParams(
            style=StyleParams(num_steps=1),
            max_frames=3,
            video_size=96,
            first_frame_steps=1,
            frame_steps=1,
            temporal_weight=50.0,
            preview_every=0,
        )
        result = engine.process_video(input_path, style, params)

        assert result.frame_count == 3
        assert result.width % 2 == 0 and result.height % 2 == 0
        assert result.codec in {"H.264", "mp4v"}
        assert result.video_bytes[4:8] == b"ftyp"
        assert len(result.history) == 3
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
