"""An atlas of where NCM and CMM sit RELATIVE TO THE FOUND BAND.

Alice, 2026-08-11: "i was just wondering if you could implement the atlas and the find band,
and find band can help since its all relative".

THE PROBLEM THIS SOLVES. An atlas needs a common frame, and the obvious frame -- centre the
tissue, scale by its size, rotate so the dorsal click points up -- does not work. Measured
leave-one-out and scored on cells it got NCM 59% kept / 69% density and CMM 21% / 42%, against
find_band's 83%/83% and 73%/69%; the ruler baseline beats it. The reason is not that NCM moves.
Letting each held-out section CHEAT -- rotate and rescale itself to whatever matched Alice's
outline best -- lifted it only to NCM 80%/92%, CMM 50%/82%, and four of the fourteen sections
wanted a 90 DEGREE correction. The frame was rotated wrong. And the outline cannot fix that: a
plain circle of equal area matches these outlines with IoU 0.82 against 0.85 for a real section,
and 20 degrees of rotation are indistinguishable from the best. The tissue shape is consistent,
as Alice said, but it is nearly featureless, so it carries position and scale and no angle.

THE FIX IS TO BORROW THE ANGLE FROM find_band. Its band angle is good to a median 7 degrees, so
the band is a far better axis than the tissue outline or the dorsal click. Every pixel gets two
coordinates measured against the band instead of against the canvas:

    across  um from the band's NCM-facing border, positive toward NCM
    along   um along the band from the band's own centre, sign-flipped for LH sections

In that frame there is nothing left to align, which is the whole point.

WHAT IT REPLACES. find_band currently fixes NCM at a constant 1300 x 1900 um and CMM at
fractions of the tissue behind its border -- about ten numbers I chose from statistics pooled
over all 14 sections. Here those numbers come from a VOTE: for each (along, across) cell, what
fraction of the other sections called that spot NCM. Same information, but held-out, and it can
express a shape a box cannot.

WHAT IT ACTUALLY BOUGHT: A CHECK ON THE CONSTANTS, NOT A BETTER ANSWER.

Leave-one-out, scored on cells (kept / area ratio / density = the fraction of true cells-per-mm2
a region would report):

    vote        NCM kept  area  dens     CMM kept  area  dens
    0.3            89%   x1.14   77%        78%   x1.41   56%
    0.4            86%   x1.05   82%        65%   x1.00   65%
    0.5            81%   x0.97   83%        51%   x0.68   76%
    find_band      83%   x1.03   81%        73%   x1.00   73%

So NCM is a TIE and CMM is a LOSS. Combining them does not help either: atlas AND box gives NCM
81%/81%, atlas OR box gives CMM 82% kept but only 64% density.

THE RESULT WORTH KEEPING is the footprint. Voting over all 14 sections at 0.5, the atlas puts NCM
between +0 and +1300 um across the band and -1000 to +900 um along it -- a 1300 x 1900 um box.
find_band's hand-picked constants are NCM_DEPTH_UM = 1300 and NCM_LENGTH_UM = 1900. The atlas
rederived them exactly, by a procedure with no boxes in it. That is the real payoff: those two
numbers are no longer my judgement call, and the fact that the free-form vote comes out RECTANGULAR
says a box is the right shape for NCM rather than a convenient one. CMM's footprint is -2100 to
-600 um across, -600 to +600 along, i.e. 1500 x 1200 um -- but a FIXED CMM is known to be wrong
(Alice: small CMMs are real and recurring), which is exactly why the atlas loses on CMM: an atlas
in absolute microns cannot say "however deep this section's tissue happens to go". Making the
frame relative instead (depth and length as fractions of the tissue available) reverses the two:
CMM 65%->70% kept at the same area, NCM density 82%->74%. Two mechanisms, one conclusion.

USE IT AS A REFRESH TOOL, NOT A PIPELINE STAGE. find_band needs no labels, so every section is a
fair test; the atlas needs Alice's outlines, so shipping it would trade that away for a tie. Re-run
this when there are more outlines: if the footprint has moved, the constants need updating.

Usage:
  python regions/band_atlas.py
"""

import numpy as np

import sys
from pathlib import Path

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. This file can be run as its own
# process, and Python then puts only ITS folder on sys.path -- so `import ps6` at the root, or a
# sibling folder's module, would not be found. app_path.py explains the whole arrangement; the line
# before it is there because app_path is at the root, which is not on the path yet either. The
# condition also covers a flat copy of the app, where app_path sits right here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

from ceiling_lines import DATA_PATH
from field_l_band import signed_distance
from find_band import (
    CANVAS_UM,
    along_coordinate,
    dorsal_on_canvas,
    find_band,
    is_left_hemisphere,
    regions_for,
)
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices

# Bin size for the atlas, in microns. 100 um is a deliberate compromise: fine enough to hold a
# border to a fifth of the band's width, coarse enough that 13 sections give every bin a real
# vote rather than one section's pixel.
BIN_UM = 100.0

# How far the atlas reaches. Beyond this a pixel is simply not in either region.
ALONG_LIMIT_UM = 3000.0
ACROSS_LOW_UM = -4000.0
ACROSS_HIGH_UM = 3000.0

# A bin becomes part of a region when this fraction of the sections that HAVE tissue there called
# it that region. 0.5 is the majority; swept in main().
VOTE = 0.5

# A bin with only one or two sections behind it is noise, not a vote.
MIN_VOTERS_PX = 200


def band_frame(band, tissue, mirror):
    """Every tissue pixel's (along, across) in microns, relative to the band.

    across = 0 is the band's NCM-facing border and positive points at NCM; along = 0 is the
    middle of the band. Nothing here refers to the canvas, so two sections photographed at
    different angles land on top of each other.
    """
    normal = np.array([band["normal_y"], band["normal_x"]])

    across = (signed_distance(normal, tissue.shape) - band["ncm_offset"]) * CANVAS_UM

    centre = (band["span_low"] + band["span_high"]) / 2.0
    along = (along_coordinate(band, tissue.shape) - centre) * CANVAS_UM

    # LH sections are mirror images, so their 'along' axis runs the other way anatomically. The
    # normal is already oriented toward NCM, so this is the only handedness bit left.
    return (-along if mirror else along), across


def bin_index(along, across):
    """Which atlas bin each pixel falls in, and which pixels fall inside the atlas at all."""
    inside = (
        (np.abs(along) < ALONG_LIMIT_UM)
        & (across > ACROSS_LOW_UM)
        & (across < ACROSS_HIGH_UM)
    )

    columns = int((2 * ALONG_LIMIT_UM) / BIN_UM)
    rows = int((ACROSS_HIGH_UM - ACROSS_LOW_UM) / BIN_UM)

    ia = np.clip(((along + ALONG_LIMIT_UM) / BIN_UM).astype(int), 0, columns - 1)
    ic = np.clip(((across - ACROSS_LOW_UM) / BIN_UM).astype(int), 0, rows - 1)

    return ia * rows + ic, inside, columns * rows


def sections():
    """Every section that find_band can place, with its band-frame coordinates and its cells."""
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    out = []

    for index, (stem, name) in enumerate(zip(stems, images)):
        green = data["inputs"][index, 0]
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])

        band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size, scale_um))

        if band is None:
            continue

        along, across = band_frame(band, tissue, is_left_hemisphere(stem))
        index_map, inside, bins = bin_index(along, across)

        points, _ = cells_for(name)
        ys, xs = (
            canvas_indices(points, pixel_size, tissue.shape)
            if len(points)
            else (np.array([], int), np.array([], int))
        )

        labels = data["labels"][index]

        out.append(
            {
                "stem": stem,
                "tissue": tissue,
                "band": band,
                "bins": index_map,
                "inside": tissue & inside,
                "n_bins": bins,
                "labels": labels,
                "ys": ys,
                "xs": xs,
            }
        )

    return out


def atlas_from(others, region, n_bins):
    """The vote: fraction of sections calling each bin this region, and how many voted at all."""
    label = 1 if region == "NCM" else 2

    total = np.zeros(n_bins)
    hits = np.zeros(n_bins)

    for section in others:
        keep = section["inside"]

        total += np.bincount(section["bins"][keep], minlength=n_bins)
        hits += np.bincount(
            section["bins"][keep & (section["labels"] == label)], minlength=n_bins
        )

    share = np.divide(hits, total, out=np.zeros_like(total), where=total > 0)

    return share, total


def predict(section, share, voters, vote):
    """Paint the atlas back onto one section's pixels."""
    keep = (share >= vote) & (voters >= MIN_VOTERS_PX)

    return section["inside"] & keep[section["bins"]]


def leave_one_out(secs, vote, clean=True):
    """Score the atlas the only honest way: build it from the OTHERS, test on the held-out one.

    Without this the atlas has already seen the answer it is being graded on, which in this
    project has inflated a number by 0.2 AP before now.
    """
    from scipy.ndimage import binary_fill_holes

    from ceiling_lines import largest_blob

    tally = {"NCM": [0, 0, 0, 0], "CMM": [0, 0, 0, 0]}

    for held in secs:
        others = [s for s in secs if s["stem"] != held["stem"]]

        for region in ("NCM", "CMM"):
            share, voters = atlas_from(others, region, held["n_bins"])
            found = predict(held, share, voters, vote)

            if clean:
                found = largest_blob(binary_fill_holes(found))

            drawn = held["labels"] == (1 if region == "NCM" else 2)

            tally[region][2] += int(found.sum())
            tally[region][3] += int(drawn.sum())

            if len(held["ys"]):
                inside = drawn[held["ys"], held["xs"]]
                tally[region][0] += int(inside.sum())
                tally[region][1] += int(
                    (inside & found[held["ys"], held["xs"]]).sum()
                )

    return tally


def report(tally):
    parts = []

    for region in ("NCM", "CMM"):
        total, kept, area, drawn = tally[region]
        fraction = kept / max(total, 1)
        ratio = area / max(drawn, 1)
        parts.append(
            f"{region} kept {fraction:4.0%} area x{ratio:4.2f}"
            f" dens {fraction / max(ratio, 1e-9):4.0%}"
        )

    return "   ".join(parts)


def footprint(secs, region, vote=VOTE):
    """The atlas's own extent in microns -- the number that checks find_band's constants."""
    share, voters = atlas_from(secs, region, secs[0]["n_bins"])
    rows = int((ACROSS_HIGH_UM - ACROSS_LOW_UM) / BIN_UM)

    keep = ((share >= vote) & (voters >= MIN_VOTERS_PX)).reshape(-1, rows)
    ia, ic = np.nonzero(keep)

    if not len(ia):
        return None

    along = ia * BIN_UM - ALONG_LIMIT_UM
    across = ic * BIN_UM + ACROSS_LOW_UM

    return {
        "bins": int(keep.sum()),
        "area_mm2": keep.sum() * BIN_UM ** 2 / 1e6,
        "across": (float(across.min()), float(across.max() + BIN_UM)),
        "along": (float(along.min()), float(along.max() + BIN_UM)),
    }


def picture(secs, path="grids/band atlas.png"):
    """Draw the two vote maps, so the shape can be looked at rather than trusted."""
    from pathlib import Path

    from PIL import Image, ImageDraw

    rows = int((ACROSS_HIGH_UM - ACROSS_LOW_UM) / BIN_UM)
    scale = 6
    panels = []

    for region in ("NCM", "CMM"):
        share, voters = atlas_from(secs, region, secs[0]["n_bins"])
        grid = share.reshape(-1, rows).T[::-1]          # across upward, NCM at the top
        thin = (voters.reshape(-1, rows).T[::-1] < MIN_VOTERS_PX)

        rgb = np.zeros(grid.shape + (3,), np.uint8)
        rgb[..., 2 if region == "NCM" else 0] = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
        rgb[..., 1] = (np.clip(grid, 0, 1) * 140).astype(np.uint8)
        rgb[thin] = (28, 28, 28)

        image = Image.fromarray(rgb).resize(
            (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
        )
        draw = ImageDraw.Draw(image)

        # The band itself: across 0 is its NCM-facing border, -550 the other side.
        for micron, colour in ((0.0, (255, 255, 255)), (-550.0, (255, 255, 120))):
            y = (ACROSS_HIGH_UM - micron) / BIN_UM * scale
            draw.line([(0, y), (image.width, y)], fill=colour)

        draw.line(
            [(image.width / 2, 0), (image.width / 2, image.height)], fill=(90, 90, 90)
        )
        draw.text((4, 4), f"{region}  vote share over {len(secs)} sections", fill="white")
        panels.append(image)

    sheet = Image.new(
        "RGB",
        (max(p.width for p in panels), sum(p.height for p in panels) + 34),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        "WHERE EACH REGION SITS RELATIVE TO THE FOUND BAND. Up = toward NCM across the band; "
        "left/right = along it, grey line = band centre.",
        fill="white",
    )
    draw.text(
        (4, 18),
        "WHITE line = band's NCM-facing border (across 0).  YELLOW line = its far side (-550 um)."
        "  Dark grey = too few sections to vote.",
        fill=(180, 220, 255),
    )

    y = 34
    for image in panels:
        sheet.paste(image, (0, y))
        y += image.height

    Path("grids").mkdir(parents=True, exist_ok=True)
    sheet.save(path)

    return path


def main():
    secs = sections()
    print(f"{len(secs)} sections placed by find_band\n")

    print("leave-one-out, scored on CELLS:")
    for vote in (0.3, 0.4, 0.5, 0.6):
        print(f"  vote {vote:.1f}   {report(leave_one_out(secs, vote))}")

    print("\n  find_band  NCM kept  83% area x1.03 dens  81%   "
          "CMM kept  73% area x1.00 dens  73%")

    print("\nfootprint -- what the atlas says the regions MEASURE:")
    for region in ("NCM", "CMM"):
        shape = footprint(secs, region)
        if shape is None:
            print(f"  {region}: nothing reached the vote")
            continue
        print(
            f"  {region}  {shape['area_mm2']:.2f} mm2   "
            f"across {shape['across'][0]:+.0f} to {shape['across'][1]:+.0f} um   "
            f"along {shape['along'][0]:+.0f} to {shape['along'][1]:+.0f} um"
        )
    print("  find_band's constants: NCM 1300 deep x 1900 long um")

    print(f"\nwrote {picture(secs)}")


if __name__ == "__main__":
    main()
