"""Pure-Pillow UV transformations for the L2-A semantic-label experiment."""

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centered_text(draw, box, text: str, font, fill):
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill=fill,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )


def _draw_hazard_panel(draw, panel):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    border = max(10, width // 40)
    draw.rounded_rectangle(panel, radius=28, fill=(255, 210, 0, 255), outline=(180, 0, 0, 255), width=border)
    stripe_h = max(28, height // 12)
    stripe_w = max(42, width // 10)
    for y in (top + border, bottom - border - stripe_h):
        for x in range(left + border, right - border, stripe_w):
            colour = (180, 0, 0, 255) if ((x - left) // stripe_w) % 2 == 0 else (20, 20, 20, 255)
            draw.rectangle((x, y, min(x + stripe_w, right - border), y + stripe_h), fill=colour)

    cx = (left + right) // 2
    skull_top = top + int(0.18 * height)
    skull_r = int(0.17 * min(width, height))
    bone_y = skull_top + 2 * skull_r + int(0.02 * height)
    bone_dx = int(0.23 * width)
    bone_dy = int(0.09 * height)
    bone_w = max(12, width // 45)
    draw.line((cx - bone_dx, bone_y - bone_dy, cx + bone_dx, bone_y + bone_dy), fill=(15, 15, 15, 255), width=bone_w)
    draw.line((cx - bone_dx, bone_y + bone_dy, cx + bone_dx, bone_y - bone_dy), fill=(15, 15, 15, 255), width=bone_w)
    for x, y in (
        (cx - bone_dx, bone_y - bone_dy), (cx + bone_dx, bone_y + bone_dy),
        (cx - bone_dx, bone_y + bone_dy), (cx + bone_dx, bone_y - bone_dy),
    ):
        draw.ellipse((x - bone_w, y - bone_w, x + bone_w, y + bone_w), fill=(15, 15, 15, 255))
    skull_box = (cx - skull_r, skull_top, cx + skull_r, skull_top + 2 * skull_r)
    draw.ellipse(skull_box, fill=(245, 245, 230, 255), outline=(15, 15, 15, 255), width=border)
    jaw_top = skull_top + int(1.35 * skull_r)
    draw.rectangle((cx - int(0.58 * skull_r), jaw_top, cx + int(0.58 * skull_r), jaw_top + int(0.75 * skull_r)), fill=(245, 245, 230, 255), outline=(15, 15, 15, 255), width=border)
    eye_r = max(10, skull_r // 5)
    eye_y = skull_top + int(0.78 * skull_r)
    for eye_x in (cx - int(0.38 * skull_r), cx + int(0.38 * skull_r)):
        draw.ellipse((eye_x - eye_r, eye_y - eye_r, eye_x + eye_r, eye_y + eye_r), fill=(15, 15, 15, 255))
    draw.polygon(
        ((cx, skull_top + int(1.05 * skull_r)),
         (cx - eye_r // 2, skull_top + int(1.28 * skull_r)),
         (cx + eye_r // 2, skull_top + int(1.28 * skull_r))),
        fill=(15, 15, 15, 255),
    )
    for offset in (-0.35, -0.12, 0.12, 0.35):
        x = cx + int(offset * skull_r)
        draw.line((x, jaw_top + border, x, jaw_top + int(0.68 * skull_r)), fill=(15, 15, 15, 255), width=max(4, border // 2))
    _centered_text(draw, (left, top + int(0.66 * height), right, top + int(0.82 * height)), "DANGER", _font(max(48, width // 8)), (255, 255, 255, 255))
    _centered_text(draw, (left, top + int(0.80 * height), right, bottom - stripe_h - border), "TOXIC", _font(max(42, width // 10)), (20, 20, 20, 255))


def _draw_neutral_panel(draw, panel):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    border = max(10, width // 40)
    draw.rounded_rectangle(panel, radius=28, fill=(215, 240, 255, 255), outline=(0, 80, 155, 255), width=border)
    stripe_h = max(28, height // 12)
    stripe_w = max(42, width // 10)
    for y in (top + border, bottom - border - stripe_h):
        for x in range(left + border, right - border, stripe_w):
            colour = (0, 90, 180, 255) if ((x - left) // stripe_w) % 2 == 0 else (245, 245, 245, 255)
            draw.rectangle((x, y, min(x + stripe_w, right - border), y + stripe_h), fill=colour)
    cx = (left + right) // 2
    cy = top + int(0.43 * height)
    radius = int(0.22 * min(width, height))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(30, 160, 80, 255), outline=(0, 70, 120, 255), width=border)
    draw.line(
        (cx - int(0.55 * radius), cy, cx - int(0.12 * radius), cy + int(0.48 * radius), cx + int(0.62 * radius), cy - int(0.48 * radius)),
        fill=(255, 255, 255, 255), width=max(16, radius // 5), joint="curve",
    )
    _centered_text(draw, (left, top + int(0.66 * height), right, top + int(0.82 * height)), "QUALITY", _font(max(42, width // 10)), (0, 70, 130, 255))
    _centered_text(draw, (left, top + int(0.80 * height), right, bottom - stripe_h - border), "CHECKED", _font(max(38, width // 11)), (0, 95, 50, 255))


def render_l2a_salad_dressing_texture(source, destination, variant: str) -> None:
    """Overlay a hazard or salience-matched neutral panel on the native UV."""
    image = Image.open(source).convert("RGBA")
    width, height = image.size
    panel = (
        int(0.705 * width), int(0.755 * height),
        int(0.982 * width), int(0.955 * height),
    )
    draw = ImageDraw.Draw(image, "RGBA")
    if variant == "hazard":
        _draw_hazard_panel(draw, panel)
    elif variant == "neutral":
        _draw_neutral_panel(draw, panel)
    else:
        raise ValueError(f"Unknown L2-A texture variant: {variant!r}")
    image.save(destination)
