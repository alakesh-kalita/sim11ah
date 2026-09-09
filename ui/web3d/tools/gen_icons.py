"""Generates the pixel-art block icons for the 3D voxel-style topology
view (ui/web3d/static/icons/). Run standalone whenever a texture needs a
visual tweak:

    python3 ui/web3d/tools/gen_icons.py

Decomposed on purpose: authoring pixel art as real PNG files (viewable
directly, e.g. with the Read tool) is far easier to iterate on and verify
than reasoning about inline HTML canvas-drawing code buried inside the
three.js scene module -- app.js just loads these as textures, it no
longer generates any art itself.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "static" / "icons"
S = 32  # px per tile -- chunky pixel-art scale, matches ~1 world-block


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def fill(d: ImageDraw.ImageDraw, color: str) -> None:
    d.rectangle([0, 0, S - 1, S - 1], fill=color)


def speckle(d: ImageDraw.ImageDraw, rng: random.Random, colors: list[str], n: int) -> None:
    for _ in range(n):
        c = colors[rng.randrange(len(colors))]
        x, y = rng.randrange(S), rng.randrange(S)
        d.rectangle([x, y, x, y], fill=c)


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"{name}.png")


def gen_grass():
    img, d = canvas()
    fill(d, "#5d9c3f")
    speckle(d, random.Random(1), ["#6bab4a", "#4f8a34", "#75b452"], 90)
    save(img, "grass")


def gen_dirt():
    img, d = canvas()
    fill(d, "#7a5636")
    speckle(d, random.Random(2), ["#8a6642", "#6b4a2c", "#916d47"], 100)
    save(img, "dirt")


def gen_dirt_mining():
    img, d = canvas()
    fill(d, "#5c4630")
    speckle(d, random.Random(3), ["#6b5238", "#3e2f1e", "#4a3826"], 110)
    save(img, "dirt_mining")


def gen_paddy():
    # Dry-earth field base matching the 2D canvas's _PADDY_BASE -- the 3D
    # view now builds real bunded plots on top of the ground, so this tile
    # is the pale packed earth showing between and beyond them. The old
    # teal "flooded wash" predates those plots and made the horizon read
    # as open sea once actual water plots existed to compare against.
    img, d = canvas()
    fill(d, "#cdd9a0")
    speckle(d, random.Random(4), ["#bfcf90", "#dae4ae", "#b5c286", "#9c8a5e"], 90)
    save(img, "paddy")


def gen_paddy_crop():
    # Near-white tile with darker transplant-row bands, meant to be TINTED
    # by a material colour (Lambert multiplies map x colour): one texture
    # serves all seven _PADDY_CROPS growth-stage hues while keeping the
    # row/clump grain visible in each, instead of seven near-identical
    # green tiles.
    img, d = canvas()
    fill(d, "#ececec")
    rng = random.Random(14)
    for y in range(2, S, 8):
        d.rectangle([0, y, S - 1, y + 2], fill="#c9c9c9")
        for x in range(S):
            if rng.random() < 0.4:
                d.rectangle([x, y + rng.randrange(3), x, y + 2], fill="#b0b0b0")
    speckle(d, rng, ["#f7f7f7", "#dedede"], 40)
    save(img, "paddy_crop")


def gen_thatch():
    # Rice-straw thatch for the paddy farmhouse -- straw roofs are the
    # vernacular in this landscape, and the 2D canvas's hut palette
    # (_HUT_ROOF/_HUT_ROOF_LT) is straw-brown, not the terracotta shingle
    # the city buildings wear.
    img, d = canvas()
    fill(d, "#8a5a3b")
    rng = random.Random(15)
    for y in range(0, S, 4):
        d.rectangle([0, y, S - 1, y], fill="#75492c")
        for _ in range(10):
            x = rng.randrange(S)
            d.rectangle([x, y + 1, min(S - 1, x + rng.randrange(1, 4)), y + 1],
                        fill="#a5713f")
    speckle(d, rng, ["#9c6743", "#7d4f31"], 30)
    save(img, "thatch")


def gen_stone():
    img, d = canvas()
    fill(d, "#8a8a8a")
    speckle(d, random.Random(5), ["#959595", "#787878", "#8f8f8f"], 70)
    save(img, "stone")


def gen_stone_brick():
    img, d = canvas()
    fill(d, "#8f8f8f")
    for y in range(0, S, 8):
        d.rectangle([0, y, S - 1, y + 1], fill="#6e6e6e")
        off = 0 if (y // 8) % 2 == 0 else 8
        for x in range(off, S, 16):
            d.rectangle([x, y, x + 1, y + 7], fill="#6e6e6e")
    speckle(d, random.Random(6), ["#999999", "#828282"], 24)
    save(img, "stone_brick")


def gen_planks():
    img, d = canvas()
    fill(d, "#b6874a")
    for x in range(0, S, 8):
        d.rectangle([x, 0, x + 1, S - 1], fill="#9c6f3a")
    for y in range(4, S, 12):
        d.rectangle([0, y, S - 1, y], fill="#a97a3f")
    save(img, "planks")


def gen_brick():
    img, d = canvas()
    fill(d, "#9c4a34")
    for y in range(0, S, 8):
        d.rectangle([0, y, S - 1, y + 1], fill="#c2c2b0")
        off = 0 if (y // 8) % 2 == 0 else 8
        for x in range(off, S, 16):
            d.rectangle([x, y, x + 1, y + 7], fill="#c2c2b0")
    save(img, "brick")


def gen_glass():
    img, d = canvas()
    fill(d, "#bfe3f0")
    d.rectangle([2, 2, S - 3, S - 3], fill="#d8f0f8")
    # diagonal reflection streak + a 2x2 mullion grid -- a flat pane reads
    # as an empty blue square without something to break it up.
    d.polygon([(4, S - 4), (S // 2, 4), (S // 2 + 5, 4), (9, S - 4)], fill="#f2fbff")
    mid = S // 2
    d.rectangle([mid - 1, 2, mid, S - 3], fill="#5c7480")
    d.rectangle([2, mid - 1, S - 3, mid], fill="#5c7480")
    d.rectangle([0, 0, S - 1, 1], fill="#5c7480")
    d.rectangle([0, S - 2, S - 1, S - 1], fill="#5c7480")
    d.rectangle([0, 0, 1, S - 1], fill="#5c7480")
    d.rectangle([S - 2, 0, S - 1, S - 1], fill="#5c7480")
    save(img, "glass")


def gen_leaves():
    img, d = canvas()
    fill(d, "#3f8f3a")
    speckle(d, random.Random(8), ["#4fa346", "#2f752c", "#57a84e"], 110)
    save(img, "leaves")


def gen_log():
    img, d = canvas()
    fill(d, "#6b4a2e")
    for x in range(0, S, 6):
        d.rectangle([x, 0, x + 1, S - 1], fill="#57391f")
    save(img, "log")


def gen_gravel():
    img, d = canvas()
    fill(d, "#847d78")
    speckle(d, random.Random(9), ["#948c86", "#6c655f", "#7a736d"], 110)
    save(img, "gravel")


def gen_road():
    img, d = canvas()
    fill(d, "#3a3d42")
    speckle(d, random.Random(10), ["#4c5057", "#232529", "#54585f", "#2c2e32"], 130)
    save(img, "road")


def gen_sidewalk():
    img, d = canvas()
    fill(d, "#a9adb1")
    for x in range(0, S, 16):
        d.rectangle([x, 0, x + 1, S - 1], fill="#8f9296")
    for y in range(0, S, 16):
        d.rectangle([0, y, S - 1, y + 1], fill="#8f9296")
    save(img, "sidewalk")


def gen_concrete():
    img, d = canvas()
    fill(d, "#9199a1")
    speckle(d, random.Random(11), ["#a4abb2", "#7d848c", "#b3b9bf"], 80)
    save(img, "concrete")


def gen_metal():
    img, d = canvas()
    fill(d, "#aab4bd")
    for x in range(4, S, 8):
        d.rectangle([x, 0, x + 1, S - 1], fill="#8590a3")
    d.rectangle([0, 0, S - 1, S - 1], outline="#6f757c", width=1)
    rng = random.Random(21)
    for _ in range(5):
        x, y = rng.randrange(S), rng.randrange(S)
        d.rectangle([x, y, x + 1, y + 1], fill="#c2793f")  # rust flecks for character
    save(img, "metal")


def gen_warning():
    img, d = canvas()
    fill(d, "#d9451f")
    for i in range(-S, S, 8):
        d.polygon([(i, S), (i + 4, S), (i + 4 + S, 0), (i + S, 0)], fill="#f2c230")
    save(img, "warning")


def gen_roof():
    img, d = canvas()
    fill(d, "#5c5f63")
    for y in range(0, S, 8):
        d.rectangle([0, y, S - 1, y + 1], fill="#4a4d51")
    save(img, "roof")


def gen_roof_shingle():
    # Warm terracotta shingle roof -- for pitched/gabled buildings, a
    # deliberate contrast against the flat grey "roof" (used on industrial
    # sheds and flat tower roofs) so pitched roofs read as a different,
    # warmer material the way real shingle/tile roofs do.
    img, d = canvas()
    fill(d, "#a1432b")
    for y in range(0, S, 5):
        d.rectangle([0, y, S - 1, y + 1], fill="#8a3620")
        off = 0 if (y // 5) % 2 == 0 else 4
        for x in range(off, S, 8):
            d.rectangle([x, y, x, y + 4], fill="#8a3620")
    speckle(d, random.Random(20), ["#b85536", "#8a3620"], 40)
    save(img, "roof_shingle")


def gen_window_lit():
    img, d = canvas()
    fill(d, "#1c1f26")
    d.rectangle([4, 4, S - 5, S - 5], fill="#ffd977")
    save(img, "window_lit")


def gen_wall(name: str, base: str, mortar: str, lit_chance: float, seed: int):
    img, d = canvas()
    fill(d, base)
    for y in range(0, S, 8):
        d.rectangle([0, y, S - 1, y + 1], fill=mortar)
        off = 0 if (y // 8) % 2 == 0 else 8
        for x in range(off, S, 16):
            d.rectangle([x, y, x + 1, y + 7], fill=mortar)
    rng = random.Random(seed)
    if rng.random() < lit_chance:
        # A real four-pane window (frame + cross mullion), not a flat
        # blob -- a plain filled square read as a sticker, not glass.
        lit = "#ffd977" if rng.random() < 0.8 else "#cfe6f7"
        frame = "#1c2027"
        x0, y0, x1, y1 = 7, 6, 24, 25
        d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], fill=frame)
        d.rectangle([x0, y0, x1, y1], fill=lit)
        midx, midy = (x0 + x1) // 2, (y0 + y1) // 2
        d.rectangle([midx, y0, midx, y1], fill=frame)
        d.rectangle([x0, midy, x1, midy], fill=frame)
        d.rectangle([x0, y1 + 2, x1, y1 + 3], fill="#c9c2b4")  # sill
    save(img, name)


def gen_panel_wall(name: str, color: str, seam: str, seed: int):
    # Vivid flat-colour architectural panel cladding -- big saturated
    # colour blocks with a simple seam grid, a deliberately different
    # (and much more vibrant) look from the muted brick/stone/metal
    # styles, the way real modern buildings mix in bold accent colour.
    img, d = canvas()
    fill(d, color)
    d.rectangle([0, S // 2 - 1, S - 1, S // 2], fill=seam)
    d.rectangle([S // 2 - 1, 0, S // 2, S - 1], fill=seam)
    d.rectangle([0, 0, S - 1, S - 1], outline=seam, width=1)
    rng = random.Random(seed)
    if rng.random() < 0.8:
        lit = "#ffd977" if rng.random() < 0.7 else "#eaf6ff"
        frame = "#141414"
        for qx, qy in ((0, 0), (S // 2, 0), (0, S // 2), (S // 2, S // 2)):
            if rng.random() < 0.55:
                x0, y0 = qx + 5, qy + 5
                x1, y1 = qx + S // 2 - 6, qy + S // 2 - 6
                d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], fill=frame)
                d.rectangle([x0, y0, x1, y1], fill=lit)
    save(img, name)


def gen_wall_variants():
    for i in range(4):
        gen_wall(f"wall_glass_{i}", "#274a63", "#173347", 0.9, 100 + i)
        gen_wall(f"wall_brick_{i}", "#b8492e", "#d8d2c2", 0.35, 200 + i)
        gen_wall(f"wall_stone_{i}", "#7e8a96", "#5f6975", 0.3, 300 + i)
        gen_wall(f"wall_metal_{i}", "#aab4bd", "#8590a3", 0.25, 400 + i)
        gen_wall(f"wall_planks_{i}", "#b6874a", "#9c6f3a", 0.25, 500 + i)

    # "panel" is its own wall style (vivid flat colour cladding); each of
    # its 4 variant slots is a distinct accent colour rather than a random
    # re-roll of the same colour, so picking among them gives real hue
    # variety across a skyline instead of four near-identical reds.
    panel_colors = (
        ("#c0392b", "#8f2a20"),
        ("#1f9e8a", "#156f60"),
        ("#e0a72e", "#a97c1f"),
        ("#7c5cd6", "#5a3fa0"),
    )
    for i, (color, seam) in enumerate(panel_colors):
        gen_panel_wall(f"wall_panel_{i}", color, seam, 600 + i)


def main():
    gen_grass(); gen_dirt(); gen_dirt_mining(); gen_paddy()
    gen_paddy_crop(); gen_thatch()
    gen_stone(); gen_stone_brick(); gen_planks(); gen_brick(); gen_glass()
    gen_leaves(); gen_log(); gen_gravel(); gen_road(); gen_sidewalk()
    gen_concrete(); gen_metal(); gen_warning(); gen_roof(); gen_roof_shingle(); gen_window_lit()
    gen_wall_variants()
    n = len(list(OUT.glob("*.png")))
    print(f"wrote {n} icons to {OUT}")


if __name__ == "__main__":
    main()
