"""Walk outward from the Field L border and stop at the FIRST black gap each ray crosses.

Alice, 2026-08-14: *"theres so clearly empty space(black sutff) that separates it its not that
complicated"*. With a base-rate control, she is right about the cue: her interior far border sits
within 100 um of raw non-tissue 73% of the time against a 36% base rate anywhere inside NCM. Darkness
in general is not the cue -- 86% against a 67% base rate, nearly worthless.

WHY THE OBVIOUS USE OF IT FAILED. `rule_stops_at_black.py` built the region on the unfilled mask and
kept the connected piece. Worth +0.008 IoU. The reason is that these gaps are partial SLITS, not
walls: connectivity only stops a region if the gap severs the tissue completely, and otherwise the
region flows around the end of the slit and refills everything beyond it.

A slit is directional evidence, so it has to be used directionally. Every ray out from the Field L
border stops at the first gap IT crosses -- a ray that meets the slit stops, a ray that misses it
carries on to the tissue edge. That is also what she does by eye, and it is not a depth cap
(*"definitley dont do the cap"*): nothing here has a maximum distance, the stopping point is wherever
the image puts a gap, and a ray with no gap runs to the edge exactly as today's rule does.

HOW THE RAYS ARE BUILT. The band frame already supplies both coordinates: `across` is distance out
from the Field L border and `along` is position down the band. So a "ray" is one column of a table
indexed (along, across) -- no geometry to construct, and it inherits the band's angle for free. The
region becomes: inside the window, and no gap encountered at a smaller `across` on the same `along`.

THE ONE PARAMETER, FITTED HONESTLY. A gap must be thick enough to be a gap rather than the space
between two cells, which is what killed the lamina detector ([[lamina-cue-failed-three-ways]]).
`--thickness` is that width, chosen leave-one-bird-out so the reported number is not fitted on the
section it is scored on ([[blind-grid-is-the-only-honest-metric]]).

Usage:
  python rule_first_gap.py
  python rule_first_gap.py --thickness 24     # a fixed width instead of the LOBO sweep
"""

import argparse

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, distance_transform_edt

from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import NCM_LENGTH_UM, dorsal_on_canvas, find_band, is_left_hemisphere
from hippocampus_check import cells_kept, overlap
from review_rule import CANVAS_UM, solid_tissue
from rule_stops_at_black import fill_small_holes
from train_shape import bird

# Candidate gap thicknesses in um. A gap has to survive an opening of this width to count, so 0 means
# "any hole in the mask at all" and 40 means "only gaps as wide as a couple of cell diameters".
THICKNESSES = (0.0, 8.0, 16.0, 24.0, 40.0)

# Bin width for the (along, across) table. One canvas pixel, so nothing is smoothed away; the table is
# only a re-indexing of the section, not a downsample.
BIN_UM = CANVAS_UM

# Holes below this share of the region get filled at the end, so small gaps she draws over do not
# punch through the finished region ([[regions-must-be-continuous-patches]]).
FILL_FRACTION = 0.20


def gap_mask(tissue, thickness_um):
    """Non-tissue that is at least `thickness_um` wide, as a barrier mask.

    A morphological opening is the right test for "wide enough": it deletes any structure a disc of
    that radius cannot fit inside, so a one-pixel crack disappears while a real gap survives intact.
    """
    holes = ~tissue

    if thickness_um <= 0.0:
        return holes

    radius = max(int(round(thickness_um / CANVAS_UM / 2.0)), 1)

    return binary_opening(holes, iterations=radius)


def stop_at_first_gap(window, along, across, barrier):
    """Everything in `window` with no barrier pixel at a smaller `across` on the same `along`.

    Implemented as a scatter into an (along, across) table rather than by tracing rays: a cumulative
    maximum down the across axis marks every cell at or beyond the first barrier, and one lookup maps
    the answer back onto the section. Same result as ray tracing, one pass instead of thousands.
    """
    eligible = window & np.isfinite(along) & np.isfinite(across)

    if not eligible.any():
        return window

    rows, columns = np.nonzero(eligible)
    a = np.rint(along[rows, columns] / BIN_UM).astype(np.int64)
    c = np.rint(across[rows, columns] / BIN_UM).astype(np.int64)

    a -= a.min()
    c -= c.min()
    shape = (int(a.max()) + 1, int(c.max()) + 1)

    hit = np.zeros(shape, bool)
    is_barrier = barrier[rows, columns]
    hit[a[is_barrier], c[is_barrier]] = True

    # Cumulative OR outward: True from the first barrier in this ray onward.
    blocked = np.maximum.accumulate(hit, axis=1)

    keep = np.zeros(window.shape, bool)
    keep[rows, columns] = ~blocked[a, c]

    return keep


def regions_for(green, tissue, name, pixel_size_um, scale_um, thickness_um):
    """Today's rule, and the same rule stopped at the first gap on each ray."""
    band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size_um, scale_um))

    if band is None:
        return None, None

    along, across = band_frame(band, tissue, is_left_hemisphere(name))
    solid = solid_tissue(tissue)

    window = (
        (across >= 0.0)
        & (along >= -NCM_LENGTH_UM / 2.0)
        & (along <= NCM_LENGTH_UM / 2.0)
    )

    today = window & solid
    today = largest_blob(binary_fill_holes(today)) if today.any() else today

    # The barrier is a gap in the RAW mask -- the holes `solid_tissue` fills in.
    barrier = gap_mask(tissue, thickness_um) & solid

    stopped = stop_at_first_gap(window & solid, along, across, barrier)
    stopped = fill_small_holes(largest_blob(stopped), FILL_FRACTION) if stopped.any() else stopped

    return today, stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thickness", type=float, default=None,
                        help="fixed gap thickness in um; omit for the leave-one-bird-out sweep")
    args = parser.parse_args()

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    sections = []

    for index, (stem, name) in enumerate(zip(stems, images)):
        truth = data["labels"][index] == 1

        if not truth.any():
            continue

        sections.append({
            "stem": stem,
            "name": name,
            "bird": bird(stem),
            "green": data["inputs"][index, 0].astype(np.float32),
            "tissue": data["inputs"][index, 1] > 0.5,
            "pixel_size": float(data["pixel_sizes"][index]),
            "truth": truth,
        })

    thicknesses = (args.thickness,) if args.thickness is not None else THICKNESSES
    results = {}

    for thickness in thicknesses:
        for section in sections:
            today, stopped = regions_for(
                section["green"], section["tissue"], section["name"],
                section["pixel_size"], scale_um, thickness,
            )

            if today is None or not stopped.any():
                continue

            results[(thickness, section["stem"])] = {
                "today": overlap(today, section["truth"]),
                "stopped": overlap(stopped, section["truth"]),
                "cells_today": cells_kept(today, section["truth"], section["name"], section["pixel_size"]),
                "cells_stopped": cells_kept(stopped, section["truth"], section["name"], section["pixel_size"]),
            }

        scored = [v for (t, _), v in results.items() if t == thickness]
        print(f"  thickness {thickness:5.0f} um   mean IoU {np.mean([s['stopped']['iou'] for s in scored]):.3f}"
              f"   recall {np.mean([s['stopped']['recall'] for s in scored]):.3f}"
              f"   precision {np.mean([s['stopped']['precision'] for s in scored]):.3f}")

    if args.thickness is not None:
        return

    print("\n  LEAVE-ONE-BIRD-OUT: thickness picked on the other birds, scored on the held-out one\n")
    print(f"  {'held-out bird':10s} {'picked':>8s} {'IoU now':>9s} {'IoU gap':>9s} "
          f"{'prec now':>10s} {'prec gap':>10s}")

    rows = []

    for held in sorted({s["bird"] for s in sections}):
        others = [s["stem"] for s in sections if s["bird"] != held]
        mine = [s["stem"] for s in sections if s["bird"] == held]

        best, choice = -1.0, THICKNESSES[0]

        for thickness in THICKNESSES:
            scored = [results[(thickness, stem)] for stem in others if (thickness, stem) in results]

            if not scored:
                continue

            mean_iou = float(np.mean([s["stopped"]["iou"] for s in scored]))

            if mean_iou > best:
                best, choice = mean_iou, thickness

        held_out = [results[(choice, stem)] for stem in mine if (choice, stem) in results]

        if not held_out:
            continue

        row = {
            "bird": held,
            "choice": choice,
            "iou_now": float(np.mean([s["today"]["iou"] for s in held_out])),
            "iou_gap": float(np.mean([s["stopped"]["iou"] for s in held_out])),
            "prec_now": float(np.mean([s["today"]["precision"] for s in held_out])),
            "prec_gap": float(np.mean([s["stopped"]["precision"] for s in held_out])),
            "cells_now": [s["cells_today"] for s in held_out if s["cells_today"] is not None],
            "cells_gap": [s["cells_stopped"] for s in held_out if s["cells_stopped"] is not None],
        }
        rows.append(row)

        print(f"  {held:10s} {choice:7.0f}u {row['iou_now']:9.3f} {row['iou_gap']:9.3f} "
              f"{row['prec_now']:10.3f} {row['prec_gap']:10.3f}")

    cells_now = [v for r in rows for v in r["cells_now"]]
    cells_gap = [v for r in rows for v in r["cells_gap"]]

    print(f"\n  {'mean':10s} {'':8s} {np.mean([r['iou_now'] for r in rows]):9.3f} "
          f"{np.mean([r['iou_gap'] for r in rows]):9.3f} "
          f"{np.mean([r['prec_now'] for r in rows]):10.3f} "
          f"{np.mean([r['prec_gap'] for r in rows]):10.3f}")

    if cells_now and cells_gap:
        print(f"  cells kept: today {100 * np.mean(cells_now):.1f}%  "
              f"first gap {100 * np.mean(cells_gap):.1f}%  (n = {len(cells_now)} sections)")


if __name__ == "__main__":
    main()
