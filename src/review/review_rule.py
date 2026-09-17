"""Alice's NCM rule, drawn: straight Field L line, then fill to the TISSUE EDGE.

Alice, 2026-08-12: *"are you doing what i told you for ncm of finding field l line and filling in
the tissue edge"* -- I was not, quite. review_straight.py pins the border straight but then fills
outward only as far as the NET's mask reaches, so the far boundary was still the model's opinion.
This file does the rule as stated, and then: *"can i see what my rule + the net predicts"*.

THE RULE HAS THREE PARTS, and only one of them needs a model:

  1. the Field L border -- a straight line, which in the band frame is just `across == 0`
  2. the far border    -- the TISSUE EDGE, i.e. keep going while inside the tissue mask
  3. the two ends      -- where NCM stops ALONG the line. Nothing geometric supplies this, so it
                          is either a fixed length (no model at all) or the net's own extent.

Measured, held out by bird, on her outlines:

  A  rule with NO MODEL (fixed 1900 um length)   84% cells x1.04 = 80% density
  B  rule, net picks only the two ends           91% cells x1.08 = 83%      <- most cells found
  B' same, plus a 1100 um depth cap              89% cells x1.02 = 87%
  C  review_straight.py (net picks the depth)    87% cells x0.99 = 88%      <- best density
  D  find_band's box (ships today)               83% cells x1.03 = 81%

So her rule FINDS THE MOST CELLS of anything tested, and pays for it in area. The whole gap
between B and C is depth: filling to the tissue edge overshoots, but the sweep shows the cap stops
mattering above 1300 um, so the overshoot is a thin deep sliver rather than a slab. Worth keeping
in view that A -- a straight line, the tissue edge and one constant, no net whatsoever -- already
matches the box.

Usage:
  python review_rule.py                 # her rule, ends from the net
  python review_rule.py 1100            # ...with a depth cap in microns
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes

from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import NCM_LENGTH_UM, dorsal_on_canvas, find_band, is_left_hemisphere
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS, DRAWN_COLOURS, PREDICTED_COLOURS, outline
from review_straight import band_line
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices


# The shortest NCM in the set spans 1543 um along the band and the longest 2602. A net whose
# extent falls outside that is not describing NCM, so the ends are LEASHED to this window --
# the same move that made the CMM polygon work: keep the prediction, refuse the impossible.
ENDS_FLOOR_UM = 1400.0
ENDS_CEILING_UM = 2700.0


# How wide a gap counts as a crack to be bridged rather than a real edge. 16 um is the smallest
# that closes Gre595_RH_3-1-4; 32 um measured identically, so the smaller one is taken.
CLOSE_UM = 16.0
CANVAS_UM = 8.0


def solid_tissue(tissue, close_um=CLOSE_UM):
    """The section as ONE SOLID PIECE -- outer edge only, cracks bridged, interior debris filled.

    Alice, 2026-08-12: *"can you make sure its a continuous patch so you dont get holes like in
    gre595 3-1-4"*. Her rule says fill out to the tissue EDGE, and "edge" means the outline of the
    section, not the rim of every dark speck inside it. Confirmed against her own hand: 1.9% of her
    NCM outlines sit on non-tissue pixels pooled, and 14.9% on Gre595_RH_3-1-4 -- she draws straight
    over these gaps, so honouring them was never right.

    THREE STEPS, AND THE MIDDLE ONE IS THE ONE THAT ACTUALLY MATTERS:

      largest_blob   so a detached fragment elsewhere on the slide is not swallowed by the fill
      binary_closing dilate then erode -- BRIDGES CRACKS. Gre595's gap is NOT an enclosed hole; it
                     opens out to the tissue boundary, so fill_holes alone left 10.2% of her outline
                     uncovered there. Closing takes that to 0.4% and the area ratio x0.89 -> x0.99.
      fill_holes     anything now enclosed counts as inside the section

    Honest cost: Purp30_LH_1-1-8 goes x1.19 -> x1.29 because bridging a notch let the region flood
    further along it. Pooled cells and density do not move (91%, 80%), so this buys continuity at
    the price of one section's area.
    """
    radius = max(int(round(close_um / CANVAS_UM)), 1)
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), bool)

    return binary_fill_holes(binary_closing(largest_blob(tissue), structure=footprint))


def ends_from(net, along):
    """Where NCM stops along the line: the net's extent, leashed to a plausible span.

    Alice, 2026-08-12: *"what happened in yw113 1-1-2 since the line is tehre but the area is
    barely outlined?"* -- the net's mask on that section spans 159 um against 1543 um of real NCM,
    so the rule dutifully filled a 159 um strip. The straight line was never the problem; the ENDS
    were, because they are the one part of the rule a model has to supply.

    Keeping the net's CENTRE and clamping only the SPAN is the cheapest honest guard: a collapsed
    net still says roughly WHERE NCM is, it just understates how far it runs.
    """
    if not net.any():
        return -NCM_LENGTH_UM / 2.0, NCM_LENGTH_UM / 2.0

    low, high = float(along[net].min()), float(along[net].max())
    centre = (low + high) / 2.0
    span = float(np.clip(high - low, ENDS_FLOOR_UM, ENDS_CEILING_UM))

    return centre - span / 2.0, centre + span / 2.0


def by_the_rule(tissue, along, across, low, high, cap=None):
    """Everything inside the tissue, on the NCM side of the line, between the two end cuts."""
    mask = solid_tissue(tissue) & (across >= 0.0) & (along >= low) & (along <= high)

    if cap is not None:
        mask &= across <= cap

    return largest_blob(binary_fill_holes(mask))


def panel(green, tissue, truth, rule, net, line, ys, xs):
    base = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    rgb[outline(tissue)] = (70, 70, 70)

    if rule.any():
        colour = np.array(PREDICTED_COLOURS["NCM"])
        rgb[rule] = (0.55 * rgb[rule] + 0.45 * colour).astype(np.uint8)

    # The net's own mask, dashed, so what the model contributes is separable by eye from what the
    # rule contributes. Here the net only supplies the two end cuts.
    rows, columns = np.nonzero(outline(net))
    keep = (rows + columns) % 6 < 3
    rgb[rows[keep], columns[keep]] = (255, 140, 140)

    rgb[line] = (255, 230, 60)
    rgb[outline(truth)] = DRAWN_COLOURS["NCM"]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for y, x in zip(ys, xs):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))

    return image


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else None

    saved = np.load("shape_masks.npz")
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    panels, captions = [], []
    cells = kept = area = drawn = 0

    for index, (stem, name) in enumerate(zip(stems, images)):
        if stem + "|NCM" not in saved:
            continue

        green = data["inputs"][index, 0]
        tissue = data["inputs"][index, 1] > 0.5
        truth = data["labels"][index] == 1
        pixel_size = float(data["pixel_sizes"][index])

        band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size, scale_um))

        if band is None:
            continue

        along, across = band_frame(band, tissue, is_left_hemisphere(name))
        net = saved[stem + "|NCM"].astype(bool)
        low, high = ends_from(net, along)

        rule = by_the_rule(tissue, along, across, low, high, cap)
        line = band_line(tissue, across)

        points, _ = cells_for(name)
        ys, xs = (
            canvas_indices(points, pixel_size, tissue.shape)
            if len(points)
            else (np.array([], int), np.array([], int))
        )

        ratio = rule.sum() / max(int(truth.sum()), 1)

        if len(ys) and truth[ys, xs].any():
            inside = truth[ys, xs]
            share = (inside & rule[ys, xs]).sum() / inside.sum()
            was = (inside & net[ys, xs]).sum() / inside.sum()
            numbers = (
                f"{share:.0%} cells (net alone {was:.0%})  x{ratio:.2f} = "
                f"{share / max(ratio, 1e-9):.0%} dens"
            )
            cells += int(inside.sum())
            kept += int((inside & rule[ys, xs]).sum())
        else:
            numbers = f"x{ratio:.2f} area (no cells here)"

        area += int(rule.sum())
        drawn += int(truth.sum())

        panels.append(panel(green, tissue, truth, rule, net, line, ys, xs))
        captions.append((stem[:30], numbers))
        print(f"  {stem[:32]:34s} {numbers}")

    share = kept / max(cells, 1)
    ratio = area / max(drawn, 1)
    print(
        f"\n  pooled: {share:.0%} cells x{ratio:.2f} = {share / max(ratio, 1e-9):.0%} density"
        f"   (net+straight 87% x0.99 88%, box 83% x1.03 81%)"
    )

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new(
        "RGB", (COLUMNS * cell_width, HEADER_HEIGHT + rows * cell_height), "black"
    )
    draw = ImageDraw.Draw(sheet)
    label = f"depth cap {cap:.0f} um" if cap else "no depth cap"
    draw.text(
        (4, 4),
        f"BLUE FILL = ALICE'S RULE ({label}): straight Field L line, then fill to the TISSUE "
        f"EDGE.  YELLOW = the straight line.",
        fill="white",
    )
    draw.text(
        (4, 18),
        "PINK DASHED = what ShapeNet predicts on its own. Here the net supplies ONLY the two end "
        "cuts along the line -- nothing else.",
        fill=(255, 190, 190),
    )
    draw.text(
        (4, 32),
        "CYAN = Alice's outline.  White dots = cells.  Blue outside cyan is over-coverage; cyan "
        "outside blue is missed NCM.",
        fill=(180, 255, 180),
    )

    for index, (image, (stem, numbers)) in enumerate(zip(panels, captions)):
        row, column = divmod(index, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height

        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), stem, fill="white")
        draw.text((x + 2, y + image.height + 15), numbers, fill="yellow")

    Path("grids").mkdir(parents=True, exist_ok=True)
    out = f"grids/alice rule{'' if cap is None else f' cap {cap:.0f}'}.png"
    sheet.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
