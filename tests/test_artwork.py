import struct

from autotagger.artwork import image_dimensions
from autotagger.providers.itunes import _upscale_artwork, artwork_url_ladder


def test_upscale_artwork_url():
    url = "https://is1-ssl.mzstatic.com/image/thumb/Music/x/y/z/source/100x100bb.jpg"
    assert _upscale_artwork(url).endswith("/3000x3000bb.jpg")


def test_artwork_ladder_is_ordered_and_unique():
    url = "https://example.invalid/a/100x100bb.jpg"
    ladder = artwork_url_ladder(url)
    assert ladder[0].endswith("3000x3000bb.jpg")
    assert len(ladder) == len(set(ladder))
    assert any(u.endswith("600x600bb.jpg") for u in ladder)


def test_png_dimensions():
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 480)
    assert image_dimensions(png) == (640, 480)


def test_jpeg_dimensions():
    # SOI, a dummy APP0 segment, then SOF0 carrying 1200x1200.
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
        + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 1200, 1200)
        + b"\x03" + b"\x00" * 9
    )
    assert image_dimensions(jpeg) == (1200, 1200)


def test_unknown_bytes():
    assert image_dimensions(b"not an image at all") is None
