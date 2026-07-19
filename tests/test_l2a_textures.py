from PIL import Image

from experiments.robot.libero.l2a_textures import render_l2a_salad_dressing_texture


def test_texture_variants_only_change_the_front_label_panel(tmp_path):
    source = tmp_path / "source.png"
    hazard = tmp_path / "hazard.png"
    neutral = tmp_path / "neutral.png"
    Image.new("RGBA", (2048, 2048), (100, 120, 140, 255)).save(source)

    render_l2a_salad_dressing_texture(source, hazard, "hazard")
    render_l2a_salad_dressing_texture(source, neutral, "neutral")

    source_image = Image.open(source)
    hazard_image = Image.open(hazard)
    neutral_image = Image.open(neutral)
    assert hazard_image.getpixel((100, 100)) == source_image.getpixel((100, 100))
    assert neutral_image.getpixel((100, 100)) == source_image.getpixel((100, 100))
    assert hazard_image.getpixel((1600, 1700)) != neutral_image.getpixel((1600, 1700))
    assert hazard_image.getbbox() == neutral_image.getbbox() == source_image.getbbox()
