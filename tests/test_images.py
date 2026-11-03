from io import BytesIO

from PIL import Image

from style_transfer import open_rgb_image


def image_bytes(image: Image.Image, **save_kwargs) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    buffer.seek(0)
    return buffer


def test_rgba_images_are_composited_on_white():
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    result = open_rgb_image(image_bytes(image))

    assert result.mode == "RGB"
    red, green, blue = result.getpixel((0, 0))
    assert red == 255
    assert 120 <= green <= 135
    assert 120 <= blue <= 135


def test_palette_transparency_is_composited_on_white():
    image = Image.new("P", (4, 4), 0)
    image.putpalette([255, 0, 0, 0, 0, 0] + [0] * 762)
    image.info["transparency"] = 0
    result = open_rgb_image(image_bytes(image))

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (10, 20), (30, 60, 90))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)

    result = open_rgb_image(buffer)
    assert result.size == (20, 10)
