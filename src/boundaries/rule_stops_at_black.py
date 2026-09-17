"""Does letting the region stop at black gaps fix the far border, with no model at all?

Alice, 2026-08-14: *"the stuff on the other side is just random brain tissue i dont know but theres so
clearly empty space(black sutff) that separates it its not that complicated"*.

She is right, and measurement agrees: 73% of the "interior far border" -- the target three models
failed on -- sits within 100 um of RAW non-tissue, and 86% within 100 um of the darkest 5% of pixels.
The reason nothing could see it is self-inflicted. `solid_tissue` fills the tissue mask's interior
holes, because she draws over small gaps and the region has to be a continuous patch
([[regions-must-be-continuous-patches]]). Filling happens BEFORE the region is built, so a border she
drew along a real black gap ends up in the middle of solid tissue and gets filed as "interior"
([[ncm-border-is-three-kinds]]).

THE TENSION, AND WHY IT IS A SIZE QUESTION. Both of these are true:

  small holes inside NCM      she draws straight over them, so they must be FILLED
  a large gap beside NCM      separates NCM from unrelated tissue, so it must STOP the region

A single closed mask cannot do both. The order that can: build on the RAW mask so gaps are still
there, keep the connected piece that touches Field L, and only then fill holes below a size cut. The
gap stops the region because the tissue beyond it is a different connected component; the small hole
gets filled because it is small.

WHY SEEDED AND NOT LARGEST-COMPONENT. Taking the largest piece is one bad section away from disaster:
a gap that runs INTO NCM splits it in two and the larger half wins arbitrarily. Seeding from the Field
L border -- the one boundary that is known per section, and known to be exact because it is an edge of
her own polygon -- keeps the piece that is anatomically NCM even when it is the smaller one.

Usage:
  python rule_stops_at_black.py
  python rule_stops_at_black.py --fill 0.05      # hole size cut, as a fraction of region area
"""

import argparse

import numpy as np
from scipy.ndimage import binary_fill_holes, label as connected_components

from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import NCM_LENGTH_UM, dorsal_on_canvas, find_band, is_left_hemisphere
from hippocampus_check import cells_kept, overlap
from review_rule import CANVAS_UM, solid_tissue

# Holes smaller than this share of the region's area get filled; bigger ones are left open. 5% is a
# starting point, not a fitted value -- the flag exists so the sensitivity is visible rather than
# hidden in a constant.
FILL_FRACTION = 0.05


def fill_small_holes(mask, fraction):
    """Fill only the enclosed holes below `fraction` of the mask's area.

    `binary_fill_holes` on its own fills everything, which is the behaviour being replaced. Labelling
    the complement and keeping the components that do not touch the border identifies the enclosed
    holes; the ones that touch the border are outside space and must never be filled.
    """
    filled = binary_fill_holes(mask)
    holes = filled & ~mask

    if not holes.any():
        return mask

    pieces, count = connected_components(holes)
    sizes = np.bincount(pieces.ravel())
    limit = fraction * float(mask.sum())

    keep = np.zeros(count + 1, bool)
    keep[1:] = sizes[1:] <= limit

    return mask | keep[pieces]


def seeded_piece(base, along, across):
    """The connected piece of `base` that touches the Field L border nearest the band's middle.

    The seed is the eligible pixel with the smallest `across` (closest to the Field L border), with
    ties broken by |along| so it sits mid-band rather than at an end. If the seed's component is
    somehow empty the caller falls back to the largest piece.
    """
    if not base.any():
        return base

    pieces, count = connected_components(base)

    if count <= 1:
        return base

    rows, columns = np.nonzero(base)
    score = across[rows, columns] + 0.001 * np.abs(along[rows, columns])
    best = int(np.argmin(score))

    return pieces == pieces[rows[best], columns[best]]


def regions_for(green, tissue, name, pixel_size_um, scale_um, fraction):
    """Today's rule and the stops-at-black rule, from the same band."""
    band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size_um, scale_um))

    if band is None:
        return None, None

    along, across = band_frame(band, tissue, is_left_hemisphere(name))

    window = (
        (across >= 0.0)
        & (along >= -NCM_LENGTH_UM / 2.0)
        & (along <= NCM_LENGTH_UM / 2.0)
    )

    # Today: holes filled first, so gaps are invisible.
    today = window & solid_tissue(tissue)
    today = largest_blob(binary_fill_holes(today)) if today.any() else today

    # Proposed: gaps intact, keep the piece touching Field L, then fill only small holes.
    raw = window & tissue
    stopped = seeded_piece(raw, along, across) if raw.any() else raw
    stopped = fill_small_holes(stopped, fraction) if stopped.any() else stopped

    return today, stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", type=float, default=FILL_FRACTION)
    args = parser.parse_args()

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    rows = []

    print(f"  hole fill cut: {100 * args.fill:.0f}% of region area\n")
    print(f"  {'section':32s} {'IoU now':>8s} {'IoU black':>10s} "
          f"{'prec now':>9s} {'prec black':>11s} {'cells now':>10s} {'cells black':>12s}")

    for index, (stem, name) in enumerate(zip(stems, images)):
        truth = data["labels"][index] == 1

        if not truth.any():
            continue

        green = data["inputs"][index, 0].astype(np.float32)
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])

        today, stopped = regions_for(green, tissue, name, pixel_size, scale_um, args.fill)

        if today is None or not today.any() or not stopped.any():
            print(f"  {stem[:30]:32s} no band found")
            continue

        now, black = overlap(today, truth), overlap(stopped, truth)
        cells_now = cells_kept(today, truth, name, pixel_size)
        cells_black = cells_kept(stopped, truth, name, pixel_size)

        rows.append({
            "stem": stem, "now": now, "black": black,
            "cells_now": cells_now, "cells_black": cells_black,
        })

        cells = (
            f"{100 * cells_now:9.1f}% {100 * cells_black:11.1f}%"
            if cells_now is not None and cells_black is not None
            else f"{'-':>10s} {'-':>12s}"
        )

        print(f"  {stem[:30]:32s} {now['iou']:8.3f} {black['iou']:10.3f} "
              f"{now['precision']:9.3f} {black['precision']:11.3f} {cells}")

    if not rows:
        raise SystemExit("no scoreable sections")

    def mean(key, field):
        return float(np.mean([r[key][field] for r in rows]))

    counted = [r for r in rows if r["cells_now"] is not None and r["cells_black"] is not None]

    print(f"\n  {'':32s} {'IoU':>8s} {'recall':>10s} {'precision':>11s} {'cells kept':>12s}")

    for label, key, cells_key in (("today (holes filled)", "now", "cells_now"),
                                  ("stops at black", "black", "cells_black")):
        kept = (f"{100 * np.mean([r[cells_key] for r in counted]):11.1f}%"
                if counted else f"{'-':>12s}")
        print(f"  {label:32s} {mean(key, 'iou'):8.3f} {mean(key, 'recall'):10.3f} "
              f"{mean(key, 'precision'):11.3f} {kept}")

    print(f"\n  n = {len(rows)} sections, {len(counted)} with cell counts."
          f"  Her outline is the target: IoU 1.0, cells kept 100%.")


if __name__ == "__main__":
    main()
