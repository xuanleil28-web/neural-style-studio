import pytest

from style_transfer import FLOW_PRESETS, StyleParams, VideoParams, get_video_shape


@pytest.mark.parametrize(
    ("width", "height", "max_size", "expected"),
    [
        (1920, 1080, 480, (270, 480)),
        (1080, 1920, 480, (480, 270)),
        (1000, 10, 512, (16, 1600)),
        (64, 64, 480, (64, 64)),
        (101, 99, 480, (98, 100)),
    ],
)
def test_video_shape_is_even_and_bounded(width, height, max_size, expected):
    height_out, width_out = get_video_shape(width, height, max_size)
    assert height_out % 2 == 0
    assert width_out % 2 == 0
    assert (height_out, width_out) == expected


def test_video_defaults():
    params = VideoParams()
    assert params.max_frames == 36
    assert params.first_frame_steps == 35
    assert params.frame_steps == 8
    assert params.temporal_weight == 200.0
    assert params.video_size == 480
    assert params.style.style_input_size == 354


def test_flow_presets_and_fallback():
    assert set(FLOW_PRESETS) == {"fast", "balanced", "quality"}
    balanced = VideoParams(flow_preset="balanced").flow_settings()
    assert balanced["levels"] == 3
    assert balanced["winsize"] == 15
    assert VideoParams(flow_preset="missing").flow_settings() == balanced


def test_video_signature_changes_with_temporal_parameters():
    first = VideoParams()
    second = VideoParams(temporal_weight=300.0)
    assert first.signature() != second.signature()
    assert first.signature() == VideoParams().signature()


def test_style_params_are_embedded():
    style = StyleParams(color_strength=1.5, brush_scale=0.2)
    video = VideoParams(style=style)
    assert video.style is style
    assert video.style.style_input_size == 454
