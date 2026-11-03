import pytest
from PIL import Image

from style_transfer import StyleParams, fit_shape


@pytest.mark.parametrize(
    ("size", "max_size", "expected"),
    [
        ((800, 400), 512, (256, 512)),
        ((400, 800), 512, (512, 256)),
        ((100, 50), 512, (50, 100)),
        ((512, 512), 512, (512, 512)),
        ((1200, 900), 384, (288, 384)),
    ],
)
def test_fit_shape_keeps_aspect_ratio(size, max_size, expected):
    image = Image.new("RGB", size)
    assert fit_shape(image, max_size) == expected


def test_fit_shape_does_not_upscale_small_images():
    image = Image.new("RGB", (64, 32))
    assert fit_shape(image, 512) == (32, 64)


def test_fit_shape_guarantees_minimum_short_side():
    image = Image.new("RGB", (1000, 10))
    height, width = fit_shape(image, 512)
    assert min(height, width) >= 16
    assert width > height


def test_default_style_input_size_matches_notebook():
    params = StyleParams()
    assert params.style_input_size == 354


@pytest.mark.parametrize(
    ("brush_scale", "expected"),
    [(0.0, 512), (0.55, 354), (1.0, 224), (-1.0, 512), (2.0, 224)],
)
def test_style_input_size_is_clamped(brush_scale, expected):
    assert StyleParams(brush_scale=brush_scale).style_input_size == expected


def test_default_layer_weights_match_notebook():
    weights = StyleParams().layer_weights()
    assert weights == pytest.approx([0.05, 0.10, 0.50, 2.50, 1.00])


def test_color_and_brush_controls_are_independent():
    color_only = StyleParams(color_strength=2.0, brush_strength=1.0).layer_weights()
    brush_only = StyleParams(color_strength=1.0, brush_strength=2.0).layer_weights()

    assert color_only[0] == pytest.approx(0.10)
    assert color_only[1] == pytest.approx(0.20)
    assert color_only[3] == pytest.approx(2.50)
    assert brush_only[0] == pytest.approx(0.05)
    assert brush_only[3] == pytest.approx(5.00)


def test_zero_controls_fall_back_to_tiny_weights():
    weights = StyleParams(color_strength=0.0, brush_strength=0.0).layer_weights()
    assert all(weight == pytest.approx(1e-6) for weight in weights)


def test_signature_changes_with_parameters():
    first = StyleParams()
    second = StyleParams(seed=43)
    assert first.signature() != second.signature()
    assert first.signature() == StyleParams().signature()
