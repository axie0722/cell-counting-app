"""Look at NCM with its Field L border pinned to a single straight line.

Alice, 2026-08-12: *"for the ncm regions can you make sure the line bordering field l is always
straight"*, then *"can i see how the straight line on ncm looks"*.

ShapeNet predicts every pixel independently, so its band-side edge wanders -- 82 um of spread
in where the border sits along the band. That is not a shape she would ever draw. The fix uses
the one property of find_band's frame that makes this trivial: `across` is microns from the
band's NCM-facing border, so A STRAIGHT BORDER IS SIMPLY A CONSTANT `across`. No line fitting,
no angle to estimate -- the band already supplies the angle.

So: trim the net's mask to `across >= offset`, then for each thin slice ALONG the band, fill
from that constant out to however deep the mask reaches in that slice. The Field L side becomes
exactly straight; the far side stays free, so it still follows the tissue edge and tapers at
the ends -- the two things which_error.py showed a box cannot do.

Measured, held out by bird, against the ragged net:

  ShapeNet as-is (ragged)   84% cells x0.95 = 89% density, 82 um of wobble
  straight AT the band border  87% cells x0.99 = 88% density,  4 um of wobble
  find_band's box              83% cells x1.03 = 81% density,  9 um of wobble

One point of density for three points of cells and a border that is straight to within half a
pixel. Pushing the line deeper into the section trades much harder (15th percentile of the net's
own `across`: 92% density but only 77% of cells), so the band border itself is the place to put
it -- which is also what Alice means by the line bordering Field L.

Usage:
  python training/train_shape.py --members 3 --save shape_masks.npz
  python review/review_straight.py shape_masks.npz
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. This file can be run as its own
# process, and Python then puts only ITS folder on sys.path -- so `import ps6` at the root, or a
# sibling folder's module, would not be found. app_path.py explains the whole arrangement; the line
# before it is there because app_path is at the root, which is not on the path yet either. The
# condition also covers a flat copy of the app, where app_path sits right here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import dorsal_on_canvas, find_band, is_left_hemisphere
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS, DRAWN_COLOURS, PREDICTED_COLOURS, outline
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices

SLICE_UM = 80.0
BORDER_UM = 0.0        # where the straight line goes: 0 = the band's own NCM-facing border


def straighten(mask, tissue, along, across, offset=BORDER_UM, slice_um=SLICE_UM):
    """Pin the band-side border of `mask` to the straight line `across == offset`."""
    trimmed = mask & (across >= offset)
    filled = np.zeros_like(mask)
    slices = np.floor(along / slice_um).astype(int)

    for index in np.unique(slices[trimmed]):
        slab = slices == index
        depth = across[trimmed & slab].max()
        filled |= slab & tissue & (across >= offset) & (across <= depth)

    return largest_blob(binary_fill_holes(filled))


def wobble(mask, along, across, slice_um=SLICE_UM):
    """Spread of the band-side border along the band, in microns. 0 = perfectly straight."""
    if mask.sum() < 50:
        return float("nan")

    slices = np.floor(along / slice_um).astype(int)
    edge = [across[mask & (slices == i)].min() for i in np.unique(slices[mask])]

    return float(np.std(edge))


def band_line(tissue, across, offset=BORDER_UM):
    """The straight line itself, two pixels wide, so it can be seen in the picture."""
    step = 8.0 * 1.5      # canvas is 8 um/px; a line a pixel and a half thick reads clearly

    return tissue & (np.abs(across - offset) <= step)


def panel(section, ragged, straight, line, ys, xs):
    """Straightened fill, the ragged net's outline dashed for comparison, the line in yellow."""
    base = (np.clip(section["green"], 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    rgb[outline(section["tissue"])] = (70, 70, 70)

    if straight.any():
        colour = np.array(PREDICTED_COLOURS["NCM"])
        rgb[straight] = (0.55 * rgb[straight] + 0.45 * colour).astype(np.uint8)

    rows, columns = np.nonzero(outline(ragged))
    keep = (rows + columns) % 6 < 3
    rgb[rows[keep], columns[keep]] = (255, 140, 140)

    rgb[line] = (255, 230, 60)
    rgb[outline(section["labels"] == 1)] = DRAWN_COLOURS["NCM"]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for y, x in zip(ys, xs):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))

    return image


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "shape_masks.npz"
    saved = np.load(path)

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
        labels = data["labels"][index]
        pixel_size = float(data["pixel_sizes"][index])

        band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size, scale_um))

        if band is None:
            continue

        along, across = band_frame(band, tissue, is_left_hemisphere(name))
        ragged = saved[stem + "|NCM"].astype(bool)
        straight = straighten(ragged, tissue, along, across)
        line = band_line(tissue, across)

        points, _ = cells_for(name)
        ys, xs = (
            canvas_indices(points, pixel_size, tissue.shape)
            if len(points)
            else (np.array([], int), np.array([], int))
        )

        section = {"green": green, "tissue": tissue, "labels": labels}
        truth = labels == 1
        ratio = straight.sum() / max(int(truth.sum()), 1)
        before, after = wobble(ragged, along, across), wobble(straight, along, across)

        if len(ys) and truth[ys, xs].any():
            inside = truth[ys, xs]
            share = (inside & straight[ys, xs]).sum() / inside.sum()
            was = (inside & ragged[ys, xs]).sum() / inside.sum()
            numbers = (
                f"{share:.0%} cells (ragged {was:.0%})  x{ratio:.2f} = "
                f"{share / max(ratio, 1e-9):.0%} dens   wobble {before:.0f}->{after:.0f}um"
            )
            cells += int(inside.sum())
            kept += int((inside & straight[ys, xs]).sum())
        else:
            numbers = (
                f"x{ratio:.2f} area (no cells)   wobble {before:.0f}->{after:.0f}um"
            )

        area += int(straight.sum())
        drawn += int(truth.sum())

        panels.append(panel(section, ragged, straight, line, ys, xs))
        captions.append((stem[:30], numbers))
        print(f"  {stem[:32]:34s} {numbers}")

    share = kept / max(cells, 1)
    ratio = area / max(drawn, 1)
    print(
        f"\n  pooled: {share:.0%} cells x{ratio:.2f} = {share / max(ratio, 1e-9):.0%} "
        f"density   (ragged 84% x0.95 89%, box 83% x1.03 81%)"
    )

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new(
        "RGB", (COLUMNS * cell_width, HEADER_HEIGHT + rows * cell_height), "black"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        "BLUE FILL = NCM with its Field L border pinned STRAIGHT, held out by bird.  "
        "YELLOW = the straight line (the band's NCM-facing border).",
        fill="white",
    )
    draw.text(
        (4, 18),
        "PINK DASHED = the same ShapeNet mask BEFORE straightening, for comparison.  "
        "CYAN = Alice's outline.  White dots = cells.",
        fill=(255, 190, 190),
    )
    draw.text(
        (4, 32),
        "Only the Field L side is constrained -- the far boundary is still free, so it keeps "
        "following the tissue edge and tapering at the ends.",
        fill=(180, 255, 180),
    )

    for index, (image, (stem, numbers)) in enumerate(zip(panels, captions)):
        row, column = divmod(index, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height

        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), stem, fill="white")
        draw.text((x + 2, y + image.height + 15), numbers, fill="yellow")

    Path("grids").mkdir(parents=True, exist_ok=True)
    sheet.save("grids/straight border.png")
    print("\nwrote grids/straight border.png")


if __name__ == "__main__":
    main()
