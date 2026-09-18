"""Draw CMM as an actual convex polygon, the way Alice describes it.

Alice, 2026-08-12: *"the cmm region is a convex polygon"*, then, on seeing the raw
ShapeNet fills: *"i mean the cmm shape is still not a polygon?"* -- both right. Raw
ShapeNet predicts every pixel independently, so its CMM is ragged by construction. This
script builds the polygon version and shows it.

A convex polygon IS an intersection of half-planes. So instead of predicting a mask, predict
K numbers: for each of K fixed directions, how far out the boundary sits. Intersect the K
half-planes and the result cannot be anything but a convex polygon -- no blobs, no holes, no
islands, guaranteed by arithmetic rather than by hoping the net behaves.

The directions live in find_band's frame (`across` = microns from the band's NCM-facing
border, `along` = microns along it), so the polygon inherits the band's 7-degree-accurate
angle and never has to learn orientation. That is the mistake that sank train_band.py.

Each offset is PREDICTED, not fixed, because a constant polygon loses to find_band's box
(NCM 77% density, CMM 59%, against 81% / 73%). Two predictors are combined:

  prior_k = a_k * (how far the tissue itself reaches in direction k) + b_k
            one straight line per direction, least-squares over the TRAINING birds only.
            This is what lets CMM scale with the tissue a section happens to offer, which
            band_atlas.py showed is the thing an absolute-micron model cannot express.

  net_k   = the 98th percentile of ShapeNet's own mask along direction k.
            A percentile rather than the max, so one stray finger cannot inflate a side.

  offset_k = prior_k + clip(net_k - prior_k, -LEASH_UM, +LEASH_UM)

The leash is the whole trick. ShapeNet is better on average but occasionally lands a region
in the wrong place; the prior is dull but never wild. Letting the net move each side by at
most LEASH_UM keeps its correction and bounds its damage. LEASH_UM = 200 was picked by
sweeping 0 / 200 / 400 / 800 / 1600 / infinity: CMM density went 70 / 87 / 85 / 84 / 84 / 84,
a clear interior peak rather than a monotone trend, which is what makes it a real setting and
not a fitted knob.

CLIP_TO_TISSUE answers Alice's implicit question. With it False the region is a pure polygon
with straight sides everywhere. With it True the polygon is intersected with the tissue mask,
so its outer boundary follows the brain's edge -- still convex-bounded, but no longer straight
where the tissue curves away. Both are scored below, because which one is "right" is a
question about what she means, not about the data.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

import sys

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
from find_band import dorsal_on_canvas, find_band, is_left_hemisphere, regions_for
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS, DRAWN_COLOURS, PREDICTED_COLOURS, outline
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices

DIRECTIONS = 16
LEASH_UM = 200.0
NET_PERCENTILE = 98.0
MIN_LABEL_PX = 50

# Unit vectors fanned evenly around the compass, in (along, across) order. Sixteen because
# the tightest polygon containing Alice's CMM scores 65 / 80 / 84 / 88 / 92 percent density
# at K = 4 / 8 / 12 / 16 / 24: four sides is far too blunt, and past sixteen each extra side
# buys about one point while adding a number to fit from thirteen training sections.
FAN = [
    (np.cos(2.0 * np.pi * k / DIRECTIONS), np.sin(2.0 * np.pi * k / DIRECTIONS))
    for k in range(DIRECTIONS)
]


def projections(along, across):
    """How far each pixel lies along each of the K directions."""
    return np.stack([dy * along + dx * across for dy, dx in FAN])


def offsets_containing(projected, mask, percentile=100.0):
    """The K numbers describing a mask's convex outline, trimming thin fingers."""
    return np.array(
        [np.percentile(projected[k][mask], percentile) for k in range(DIRECTIONS)]
    )


def rasterise(projected, offsets, tissue, clip_to_tissue):
    """Intersect the K half-planes. Convex by construction."""
    inside = np.ones(tissue.shape, dtype=bool)

    for k, offset in enumerate(offsets):
        inside &= projected[k] <= offset

    if not clip_to_tissue:
        return inside

    return largest_blob(binary_fill_holes(inside & tissue))


def sections():
    """Every section find_band can place a band on, with its frame precomputed."""
    data = np.load(DATA_PATH)
    scale_um = float(data["scale_um"])
    found = []

    for index, name in enumerate(str(p) for p in data["images"]):
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])
        band = find_band(
            data["inputs"][index, 0],
            tissue,
            dorsal_on_canvas(name, pixel_size, scale_um),
        )

        if band is None:
            continue

        dorsal = dorsal_on_canvas(name, pixel_size, scale_um)
        along, across = band_frame(band, tissue, is_left_hemisphere(name))
        projected = projections(along, across)
        points, _ = cells_for(name)
        rows, columns = (
            canvas_indices(points, pixel_size, tissue.shape)
            if len(points)
            else (np.array([], int), np.array([], int))
        )

        found.append(
            {
                "stem": str(data["stems"][index]),
                "bird": str(data["stems"][index]).split("_")[0],
                "green": data["inputs"][index, 0],
                "tissue": tissue,
                "labels": data["labels"][index],
                "projected": projected,
                "reach": np.array(
                    [projected[k][tissue].max() for k in range(DIRECTIONS)]
                ),
                "box": regions_for(band, tissue),
                "rows": rows,
                "columns": columns,
                # Carried so a caller can run the NCM rule and orient the dorsal/ventral cut off
                # the same band this polygon uses, instead of a second find_band pass that could
                # land a fractionally different frame.
                "name": name,
                "pixel_size": pixel_size,
                # The band itself, so a rule can read its CMM-FACING edge. `across` is measured from
                # the NCM-facing edge, so the other one is only recoverable from the band's own
                # offsets -- and that edge is Alice's "straight line on the other side of field l".
                "band": band,
                "along": along,
                "across": across,
                "dorsal": dorsal,
            }
        )

    return found


def fit_prior(training, label):
    """One straight line per direction: offset = a * tissue_reach + b.

    Fitted on training birds only. Slopes are clipped to [0, 1.5] because a negative slope --
    "the further the tissue reaches, the smaller the region" -- is anatomically backwards and
    only ever arises from noise in thirteen points.
    """
    reaches = np.array([s["reach"] for s in training])
    truths = np.array(
        [
            offsets_containing(s["projected"], s["labels"] == label)
            for s in training
        ]
    )

    slopes = np.zeros(DIRECTIONS)
    intercepts = np.zeros(DIRECTIONS)

    for k in range(DIRECTIONS):
        x, y = reaches[:, k], truths[:, k]
        spread = ((x - x.mean()) ** 2).sum()
        slopes[k] = (
            float(np.clip(((x - x.mean()) * (y - y.mean())).sum() / spread, 0.0, 1.5))
            if spread > 0
            else 0.0
        )
        intercepts[k] = y.mean() - slopes[k] * x.mean()

    return slopes, intercepts


def leashed(section, training, label, net_mask, leash_um):
    """The predicted polygon: the prior, allowed to move leash_um toward the net."""
    slopes, intercepts = fit_prior(training, label)
    prior = slopes * section["reach"] + intercepts

    if net_mask is None or net_mask.sum() < 20:
        return prior

    net = offsets_containing(section["projected"], net_mask, NET_PERCENTILE)

    return prior + np.clip(net - prior, -leash_um, leash_um)


def tally(found, label, saved, name, leash_um, clip_to_tissue):
    """Cells kept, area ratio and density, held out by bird."""
    pool = [s for s in found if (s["labels"] == label).sum() >= MIN_LABEL_PX]
    cells = kept = area = drawn = 0
    per_section = {}

    for section in pool:
        training = [s for s in pool if s["bird"] != section["bird"]]
        net = saved.get(f"{section['stem']}|{name}")
        offsets = leashed(section, training, label, net, leash_um)
        region = rasterise(
            section["projected"], offsets, section["tissue"], clip_to_tissue
        )
        truth = section["labels"] == label

        area += int(region.sum())
        drawn += int(truth.sum())
        per_section[section["stem"]] = region

        if len(section["rows"]):
            inside = truth[section["rows"], section["columns"]]
            hit = inside & region[section["rows"], section["columns"]]
            cells += int(inside.sum())
            kept += int(hit.sum())

    share = kept / max(cells, 1)
    ratio = area / max(drawn, 1)

    return share, ratio, share / max(ratio, 1e-9), per_section


def panel(section, regions, ys, xs):
    """Polygon fill, box outline dashed for reference, Alice's outline solid, cells as dots.

    Same layering as review_shape.py's panel, deliberately, so the two sheets can be put side
    by side and the only thing that differs is the region.
    """
    base = (np.clip(section["green"], 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    rgb[outline(section["tissue"])] = (70, 70, 70)

    for name in ("NCM", "CMM"):
        region = regions[name].get(section["stem"])

        if region is not None and region.any():
            colour = np.array(PREDICTED_COLOURS[name])
            rgb[region] = (0.55 * rgb[region] + 0.45 * colour).astype(np.uint8)

    for name in ("NCM", "CMM"):
        rows, columns = np.nonzero(outline(section["box"][name]))
        keep = (rows + columns) % 6 < 3
        rgb[rows[keep], columns[keep]] = (120, 120, 120)

    for name, label in (("NCM", 1), ("CMM", 2)):
        rgb[outline(section["labels"] == label)] = DRAWN_COLOURS[name]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for y, x in zip(ys, xs):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))

    return image


def picture(found, regions, path):
    """One panel per section, laid out like every other review sheet in this project."""
    panels, captions = [], []

    for section in found:
        parts = []

        for name, label in (("NCM", 1), ("CMM", 2)):
            drawn = section["labels"] == label
            region = regions[name].get(section["stem"])

            if region is None or drawn.sum() < MIN_LABEL_PX:
                parts.append(f"{name} --")
                continue

            area = region.sum() / max(int(drawn.sum()), 1)
            ys, xs = section["rows"], section["columns"]

            if len(ys) and drawn[ys, xs].any():
                inside = drawn[ys, xs]
                kept = (inside & region[ys, xs]).sum() / inside.sum()
                boxed = (inside & section["box"][name][ys, xs]).sum() / inside.sum()
                parts.append(
                    f"{name} {kept:.0%} cells (box {boxed:.0%})"
                    f" x{area:.1f} = {kept / max(area, 1e-9):.0%} dens"
                )
            else:
                parts.append(f"{name} x{area:.1f} area (no cells)")

        panels.append(panel(section, regions, section["rows"], section["columns"]))
        captions.append((section["stem"][:30], "   ".join(parts)))
        print(f"  {section['stem'][:32]:34s} {'   '.join(parts)}")

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new(
        "RGB", (COLUMNS * cell_width, HEADER_HEIGHT + rows * cell_height), "black"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        f"FILLED = predicted CONVEX POLYGON, {DIRECTIONS} half-planes in find_band's frame, "
        "clipped to tissue, HELD OUT BY BIRD (NCM blue, CMM orange).",
        fill="white",
    )
    draw.text(
        (4, 18),
        "SOLID = Alice's outlines (NCM cyan, CMM magenta).  DASHED GREY = find_band's box.  "
        f"Every side is straight by construction; the leash lets ShapeNet move each side at "
        f"most {LEASH_UM:g} um.",
        fill=(180, 220, 255),
    )
    draw.text(
        (4, 32),
        "Where the fill looks curved it is the TISSUE MASK cutting the polygon, not a curved "
        "side. Run with CLIP_TO_TISSUE off to see the bare polygon.",
        fill=(180, 255, 180),
    )

    for index, (image, (stem, numbers)) in enumerate(zip(panels, captions)):
        row, column = divmod(index, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height

        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), stem, fill="white")
        draw.text((x + 2, y + image.height + 15), numbers, fill="yellow")

    Path("grids").mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    saved = dict(np.load("shape_masks.npz"))
    found = sections()

    print(f"{len(found)} sections, {DIRECTIONS} half-planes, leash {LEASH_UM:g} um\n")
    print(f"  {'region':<5} {'clipped to tissue':<18} kept  area  dens")

    keep = {}

    for label, name in ((1, "NCM"), (2, "CMM")):
        for clip in (True, False):
            share, ratio, density, per_section = tally(
                found, label, saved, name, LEASH_UM, clip
            )
            print(
                f"  {name:<5} {'yes' if clip else 'no -- pure polygon':<18} "
                f"{share:4.0%} x{ratio:4.2f}  {density:4.0%}"
            )

            if clip:
                keep[name] = per_section

    print("\n  find_band boxes   NCM 83% x1.03 81% | CMM 73% x1.00 73%")
    print("  ShapeNet raw      NCM 84% x0.95 89% | CMM 74% x0.96 78%")

    path = "grids/polygon regions.png"
    print()
    picture(found, keep, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
