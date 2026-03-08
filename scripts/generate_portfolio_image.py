from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
LANDSCAPE_PATH = ASSETS_DIR / "healthcare_ml_linkedin_banner.png"
SQUARE_PATH = ASSETS_DIR / "healthcare_ml_linkedin_square.png"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str]
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _draw_vertical_gradient(img: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for y in range(h):
        t = y / max(1, h - 1)
        color = (_lerp(top[0], bottom[0], t), _lerp(top[1], bottom[1], t), _lerp(top[2], bottom[2], t))
        draw.line([(0, y), (w, y)], fill=color)


def _rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2 if outline else 0)


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) // 2
    y = top + (bottom - top - th) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=4)


def _draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
    head = 10
    draw.polygon([(x2, y2), (x2 - head, y2 - head), (x2 - head, y2 + head)], fill=color)


def _draw_pipeline_cards(
    draw: ImageDraw.ImageDraw,
    cards: Iterable[str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    gap: int,
    card_font: ImageFont.ImageFont,
) -> None:
    labels = list(cards)
    box_w = (w - gap * (len(labels) - 1)) // len(labels)

    for i, label in enumerate(labels):
        left = x0 + i * (box_w + gap)
        box = (left, y0, left + box_w, y0 + h)
        _rounded_box(draw, box, radius=16, fill=(235, 246, 255), outline=(89, 149, 199))
        _draw_center_text(draw, box, label, card_font, fill=(20, 45, 75))

        if i < len(labels) - 1:
            _draw_arrow(draw, left + box_w + 8, y0 + h // 2, left + box_w + gap - 8, y0 + h // 2, (210, 235, 255))


def _draw_metric_cards(
    draw: ImageDraw.ImageDraw,
    cards: list[tuple[str, str]],
    x0: int,
    y0: int,
    w: int,
    h: int,
    gap: int,
    title_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
) -> None:
    box_w = (w - gap * (len(cards) - 1)) // len(cards)
    for i, (title, value) in enumerate(cards):
        left = x0 + i * (box_w + gap)
        box = (left, y0, left + box_w, y0 + h)
        _rounded_box(draw, box, radius=14, fill=(16, 59, 92), outline=(48, 122, 176))
        draw.text((left + 16, y0 + 14), title, font=title_font, fill=(175, 220, 250))
        draw.text((left + 16, y0 + 46), value, font=value_font, fill=(243, 251, 255))


def _draw_showcase(image: Image.Image, square: bool) -> None:
    draw = ImageDraw.Draw(image)
    w, h = image.size

    _draw_vertical_gradient(image, (13, 37, 68), (18, 82, 121))

    for bubble in [(0.11, 0.2, 260), (0.88, 0.14, 210), (0.78, 0.84, 240), (0.22, 0.85, 180)]:
        cx = int(bubble[0] * w)
        cy = int(bubble[1] * h)
        r = bubble[2] if not square else int(bubble[2] * 0.75)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 111, 155))

    title_font = _load_font(66 if not square else 54, bold=True)
    subtitle_font = _load_font(34 if not square else 28)
    body_font = _load_font(24 if not square else 20)
    metric_title_font = _load_font(22 if not square else 18, bold=True)
    metric_value_font = _load_font(30 if not square else 24, bold=True)
    footer_font = _load_font(20 if not square else 16)

    draw.text((68 if not square else 54, 46), "Healthcare ML on Google Cloud", font=title_font, fill=(236, 249, 255))
    draw.text(
        (68 if not square else 54, 124 if not square else 114),
        "End-to-End Readmission Risk System",
        font=subtitle_font,
        fill=(177, 221, 244),
    )

    pipeline_x = 68 if not square else 54
    pipeline_y = 210 if not square else 200
    pipeline_w = w - (136 if not square else 108)
    pipeline_h = 128 if not square else 104

    pipeline_cards = [
        "Synthetic\nData",
        "BigQuery\nWarehouse",
        "Feature\nSQL",
        "Vertex AI\nPipeline",
        "Endpoint +\nCloud Run",
        "Drift\nMonitoring",
    ]
    _draw_pipeline_cards(
        draw,
        pipeline_cards,
        x0=pipeline_x,
        y0=pipeline_y,
        w=pipeline_w,
        h=pipeline_h,
        gap=14 if not square else 10,
        card_font=body_font,
    )

    metric_y = 390 if not square else 360
    metric_h = 126 if not square else 112
    metric_cards = [
        ("Model Quality", "Test ROC-AUC 0.75"),
        ("Deployment", "Vertex Endpoint + API"),
        ("MLOps", "Terraform + Cloud Build"),
    ]
    _draw_metric_cards(
        draw,
        metric_cards,
        x0=pipeline_x,
        y0=metric_y,
        w=pipeline_w,
        h=metric_h,
        gap=16 if not square else 12,
        title_font=metric_title_font,
        value_font=metric_value_font,
    )

    notes_y = 555 if not square else 500
    note_box = (pipeline_x, notes_y, pipeline_x + pipeline_w, notes_y + (112 if not square else 98))
    _rounded_box(draw, note_box, radius=14, fill=(11, 43, 72), outline=(46, 117, 169))
    _draw_center_text(
        draw,
        note_box,
        "Professional ML Engineer Portfolio Case Study | Synthetic, non-PHI healthcare data",
        _load_font(26 if not square else 20),
        fill=(203, 235, 255),
    )

    draw.text(
        (pipeline_x, h - (54 if not square else 42)),
        "github.com/heussgod/gcp-healthcare-ml-case-study",
        font=footer_font,
        fill=(181, 220, 242),
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    landscape = Image.new("RGB", (1600, 900), color=(0, 0, 0))
    _draw_showcase(landscape, square=False)
    landscape.save(LANDSCAPE_PATH, format="PNG", optimize=True)

    square = Image.new("RGB", (1080, 1080), color=(0, 0, 0))
    _draw_showcase(square, square=True)
    square.save(SQUARE_PATH, format="PNG", optimize=True)

    print(f"Wrote: {LANDSCAPE_PATH}")
    print(f"Wrote: {SQUARE_PATH}")


if __name__ == "__main__":
    main()
