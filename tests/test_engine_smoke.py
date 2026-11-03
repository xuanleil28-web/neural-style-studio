import os

import pytest
from PIL import Image

from style_transfer import StyleParams, StyleTransferEngine

pytestmark = pytest.mark.slow


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
