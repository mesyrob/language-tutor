"""Generate grammar reference cards as PNG images from JSON specs.

Run: uv run --dev python scripts/generate_cards.py

Each card spec describes a grammar table (single-column or two-column).
Output is written to curriculum/images/<slug>.png.

The cards are deliberately simple/clean — no illustrations. To replace any of
these with a more polished hand-curated image, just drop a PNG with the matching
slug name into curriculum/images/.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
OUT = ROOT / "curriculum" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Card dimensions optimized for Telegram phone viewing
W, H = 900, 1100
BG = (252, 247, 230)  # warm cream
FG = (40, 32, 20)
ACCENT = (180, 90, 60)  # terracotta
DIVIDER = (200, 180, 140)

# Try system fonts; fall back to PIL default.
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
        if bold else
        ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_title(draw: ImageDraw.ImageDraw, title: str, y: int = 60) -> int:
    f = font(56, bold=True)
    bbox = draw.textbbox((0, 0), title, font=f)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y), title, fill=FG, font=f)
    return y + 110


def render_two_column(slug: str, title: str, left_label: str, right_label: str, rows: list[list[str]]):
    """rows is a list of [yo_left, value_left, yo_right, value_right] quadruples."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    y = draw_title(draw, f'"{title}"')

    # Labels
    label_f = font(40, bold=True)
    left_x = W // 4
    right_x = 3 * W // 4
    for label_text, cx in ((left_label, left_x), (right_label, right_x)):
        bbox = draw.textbbox((0, 0), label_text, font=label_f)
        lw = bbox[2] - bbox[0]
        draw.text((cx - lw // 2, y), label_text, fill=ACCENT, font=label_f)
    y += 90

    # Dashed vertical divider
    cx = W // 2
    dy = y
    while dy < H - 80:
        draw.rectangle((cx - 2, dy, cx + 2, dy + 12), fill=DIVIDER)
        dy += 22

    pron_f = font(32, bold=True)
    val_f = font(32)
    line_h = 70

    for row in rows:
        yo_l, val_l, yo_r, val_r = row
        # Left side
        text_l = f"{yo_l}"
        draw.text((80, y), text_l, fill=FG, font=pron_f)
        bbox = draw.textbbox((0, 0), text_l, font=pron_f)
        offset = bbox[2] - bbox[0]
        draw.text((80 + offset + 12, y + 4), f"- {val_l}", fill=FG, font=val_f)
        # Right side
        text_r = f"{yo_r}"
        draw.text((W // 2 + 60, y), text_r, fill=FG, font=pron_f)
        bbox = draw.textbbox((0, 0), text_r, font=pron_f)
        offset = bbox[2] - bbox[0]
        draw.text((W // 2 + 60 + offset + 12, y + 4), f"- {val_r}", fill=FG, font=val_f)
        y += line_h

    img.save(OUT / f"{slug}.png", "PNG", optimize=True)
    print(f"  → {slug}.png")


def render_three_column(slug: str, title: str, headers: list[str], rows: list[list[str]]):
    """Three-column conjugation table, used for -ar/-er/-ir style cards."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    y = draw_title(draw, title)

    # Column setup: pronoun column + 3 conjugation columns
    col_widths = [220, 200, 200, 200]
    col_starts = [80]
    for cw in col_widths[:-1]:
        col_starts.append(col_starts[-1] + cw)

    # Headers
    head_f = font(36, bold=True)
    for i, header in enumerate(headers):
        draw.text((col_starts[i + 1] + 30, y), header, fill=ACCENT, font=head_f)
    y += 70

    # Divider line
    draw.rectangle((80, y, W - 80, y + 3), fill=DIVIDER)
    y += 30

    pron_f = font(28, bold=True)
    val_f = font(28)
    line_h = 60

    for row in rows:
        pronoun, *values = row
        draw.text((col_starts[0] + 20, y), pronoun, fill=ACCENT, font=pron_f)
        for i, val in enumerate(values):
            draw.text((col_starts[i + 1] + 30, y), val, fill=FG, font=val_f)
        y += line_h

    img.save(OUT / f"{slug}.png", "PNG", optimize=True)
    print(f"  → {slug}.png")


CARDS = {
    # slug → (renderer, args)
    "ser-vs-estar": (
        render_two_column,
        {
            "title": "Ser & Estar",
            "left_label": "Ser",
            "right_label": "Estar",
            "rows": [
                ["Yo", "Soy", "Yo", "Estoy"],
                ["Tú", "Eres", "Tú", "Estás"],
                ["Él/ella", "Es", "Él/ella", "Está"],
                ["Nosotros", "Somos", "Nosotros", "Estamos"],
                ["Ellos", "Son", "Ellos", "Están"],
            ],
        },
    ),
    "regular-verbs": (
        render_three_column,
        {
            "title": "Spanish Regular Verbs",
            "headers": ["-ar (bailar)", "-er (comer)", "-ir (recibir)"],
            "rows": [
                ["yo",          "bailo",   "como",    "recibo"],
                ["tú",          "bailas",  "comes",   "recibes"],
                ["él/ella",     "baila",   "come",    "recibe"],
                ["nosotros",    "bailamos","comemos", "recibimos"],
                ["ellos",       "bailan",  "comen",   "reciben"],
            ],
        },
    ),
    "tener-conjugation": (
        render_three_column,
        {
            "title": "Tener (to have)",
            "headers": ["present", "preterite", "imperfect"],
            "rows": [
                ["yo",       "tengo",   "tuve",      "tenía"],
                ["tú",       "tienes",  "tuviste",   "tenías"],
                ["él/ella",  "tiene",   "tuvo",      "tenía"],
                ["nosotros", "tenemos", "tuvimos",   "teníamos"],
                ["ellos",    "tienen",  "tuvieron",  "tenían"],
            ],
        },
    ),
    "articles-gender": (
        render_two_column,
        {
            "title": "Articles by gender",
            "left_label": "Masculine",
            "right_label": "Feminine",
            "rows": [
                ["el", "the (sg.)",   "la", "the (sg.)"],
                ["los","the (pl.)",   "las","the (pl.)"],
                ["un", "a/an (sg.)",  "una","a/an (sg.)"],
                ["unos","some (pl.)", "unas","some (pl.)"],
                ["-o", "(usually m.)","-a", "(usually f.)"],
            ],
        },
    ),
    "preterite-vs-imperfect": (
        render_two_column,
        {
            "title": "Preterite vs Imperfect",
            "left_label": "Preterite",
            "right_label": "Imperfect",
            "rows": [
                ["use", "completed past", "use", "ongoing past"],
                ["ej.", "comí (I ate)",   "ej.", "comía (I used to eat)"],
                ["ej.", "fui (I went)",   "ej.", "iba (I was going)"],
                ["RU",  "perfective ел",  "RU",  "imperfective ел"],
                ["with", "ayer / anoche", "with", "siempre / a veces"],
            ],
        },
    ),
    "por-vs-para": (
        render_two_column,
        {
            "title": "Por vs Para",
            "left_label": "Por",
            "right_label": "Para",
            "rows": [
                ["sense", "cause / reason",  "sense", "purpose / goal"],
                ["ej.",   "gracias por...",  "ej.",   "para aprender"],
                ["sense", "through / by",    "sense", "destination"],
                ["ej.",   "por el parque",   "ej.",   "para Madrid"],
                ["sense", "in exchange for", "sense", "deadline / for"],
            ],
        },
    ),
}


def main():
    print(f"Rendering {len(CARDS)} cards to {OUT}/")
    for slug, (renderer, kwargs) in CARDS.items():
        renderer(slug, **kwargs)
    print("Done.")


if __name__ == "__main__":
    main()
