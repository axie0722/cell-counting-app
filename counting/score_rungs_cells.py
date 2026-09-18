"""Does Alice's looser criterion hold up? Score every rung of the ladder on CELLS.

Alice, 2026-08-11: "its fine if its an arbitrary polygon as long as its on the correct side
of field l and not past the tissue".

That is a much weaker requirement than the rotated box the predictor currently outputs -- it
is exactly the BOTTOM rung of the ceiling_lines.py ladder, `"line only"`: the tissue mask
split by the Field L line, with no end cuts and no outer cut. It matters because the two
parameters the network is worst at (where the region stops along the axis) are precisely the
ones this rung does not need. If it is good enough, they can be dropped.

score_region_cells.py already grades the FINAL rung on cells. This grades all four, so the
question "which cut actually earned its parameters" is answered in the unit that matters
rather than in area.

TWO KINDS OF EXTRA CELL, COUNTED SEPARATELY. The earlier metric pooled every cell a region
wrongly contained into one number. That hides a distinction Alice has been explicit about --
"its also fine if the cells arent in the delineated regions because they are still cells and
look the same":

  * a cell she assigned to the OTHER region      -> a real error. Wrong side of Field L, so
                                                    the count for both regions is wrong.
  * a cell she assigned to NEITHER region        -> the region reaching into unlabelled
                                                    brain. Inflates area, and the count with
                                                    it, but it is not a side error.

Pooling them would make a half-plane look as bad as a misplaced box when its failure is a
different and much more tolerable one. Reported as `other` and `unlabelled`.

Every number here uses Alice's own lines, so all four rungs are CEILINGS: what each rung can
do once prediction is perfect. The predictor's own score is score_region_cells.py.

Usage:
  python counting/score_rungs_cells.py            # per-section detail, then the pooled table
  python counting/score_rungs_cells.py --quiet    # pooled table only
"""

import sys

import numpy as np
import pandas as pd

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

from ceiling_lines import DATA_PATH, RUNGS, reconstructions
from score_region_cells import cells_for, to_canvas

# Region name -> its value in the label map, and the value of the region it borders. The
# "other" region is needed twice: to fit the shared Field L line, and to tell a side error
# apart from a reach into unlabelled tissue.
REGIONS = {"NCM": (1, 2), "CMM": (2, 1)}


def canvas_indices(points, pixel_size_um, shape):
    """Cell coordinates as (row, column) integer indices into the region canvas.

    Clipped rather than dropped: a point rounding one pixel outside the canvas is a
    rounding artefact, not a cell that left the section, and dropping it would quietly
    change the denominator the recall is measured against.
    """
    canvas = to_canvas(points, pixel_size_um)
    ys = np.clip(canvas[:, 0].round().astype(int), 0, shape[0] - 1)
    xs = np.clip(canvas[:, 1].round().astype(int), 0, shape[1] - 1)

    return ys, xs


def score_section(labels, tissue, ys, xs):
    """One row per (region, rung) for a single section.

    Returns counts, not fractions. Fractions are only formed at the end, over the pooled
    totals -- averaging per-section percentages would give a section with 7 cells the same
    weight as one with 51, and the sections differ that much (see
    cell-counts-vary-between-sections).
    """
    rows = []

    for name, (value, other_value) in REGIONS.items():
        drawn = labels == value
        other = labels == other_value

        if not drawn.any() or not other.any():
            continue

        # Which of Alice's regions each cell falls in, evaluated once and reused by every
        # rung, so all four rungs are scored against exactly the same cells.
        in_drawn = drawn[ys, xs]
        in_other = other[ys, xs]

        total = int(in_drawn.sum())

        if not total:
            continue

        ladder = reconstructions(drawn, other, tissue)

        for rung in RUNGS:
            inside = ladder[rung][ys, xs]

            rows.append(
                {
                    "region": name,
                    "rung": rung,
                    "cells": total,
                    "kept": int((in_drawn & inside).sum()),
                    # Cells the rung claims that Alice put in the other region: a genuine
                    # side error.
                    "other": int((inside & in_other).sum()),
                    # Cells the rung claims that Alice put in neither region: reach into
                    # unlabelled brain.
                    "unlabelled": int((inside & ~in_drawn & ~in_other).sum()),
                    "claimed": int(inside.sum()),
                    # Area is still worth carrying as a secondary diagnostic: it is the
                    # density denominator, so a rung that keeps every cell but doubles the
                    # area still halves the reported density.
                    "area_ratio": ladder[rung].sum() / max(drawn.sum(), 1),
                }
            )

    return rows


def main():
    quiet = "--quiet" in sys.argv

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(i) for i in data["images"]]
    pixel_sizes = data["pixel_sizes"]

    print(
        "Alice's criterion is the 'line only' rung: tissue, split by Field L, nothing else.\n"
        "Grading every rung on CELLS to see which cuts earn their parameters.\n"
    )

    rows = []

    for index, stem in enumerate(stems):
        points, source = cells_for(images[index])

        if not len(points):
            continue

        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]

        ys, xs = canvas_indices(points, float(pixel_sizes[index]), tissue.shape)

        found = score_section(labels, tissue, ys, xs)

        for row in found:
            row["stem"] = stem
            row["source"] = source

        rows.extend(found)

    table = pd.DataFrame(rows)

    if not len(table):
        raise SystemExit(f"No sections in {DATA_PATH} had cells to grade against.")

    if not quiet:
        print(f"=== per section, 'line only' vs '+ outer cut' ===")
        print(
            f"{'section':30s} {'region':6s} {'cells':>6s} {'line':>6s} "
            f"{'boxed':>6s}   {'line area':>9s}"
        )

        for (stem, region), group in table.groupby(["stem", "region"], sort=True):
            indexed = group.set_index("rung")
            cells = int(indexed["cells"].iloc[0])

            line = indexed.loc["line only"]
            boxed = indexed.loc["+ outer cut"]

            print(
                f"{stem[:30]:30s} {region:6s} {cells:6d} "
                f"{line['kept'] / cells:5.0%} {boxed['kept'] / cells:5.0%}   "
                f"{line['area_ratio']:8.2f}x"
            )

        print()

    print("=== pooled over sections, weighted by cell count ===")
    print(
        f"{'region':6s} {'rung':13s} {'cells':>6s} {'kept':>6s} {'other side':>11s} "
        f"{'unlabelled':>11s} {'area':>7s}"
    )

    for name in REGIONS:
        part = table[table["region"] == name]

        if not len(part):
            continue

        for rung in RUNGS:
            group = part[part["rung"] == rung]
            cells = group["cells"].sum()

            print(
                f"{name:6s} {rung:13s} {cells:6d} {group['kept'].sum() / cells:5.0%} "
                f"{group['other'].sum() / cells:10.0%} "
                f"{group['unlabelled'].sum() / cells:10.0%} "
                f"{(group['area_ratio'] * group['cells']).sum() / cells:6.2f}x"
            )

        print()

    print(
        "'kept'       cells of Alice's that the rung contains -- higher is better.\n"
        "'other side' cells she gave the OTHER region -- a real Field L error.\n"
        "'unlabelled' cells she gave neither region -- reach into unlabelled brain, which\n"
        "             inflates the count and the area but is not a side error.\n"
        "'area'       predicted area over hers: the density denominator.\n"
        "\nAll rungs are built from HER lines, so these are ceilings. If 'line only' keeps\n"
        "nearly as many cells as '+ outer cut' with few other-side errors, the end cuts are\n"
        "not carrying the meaning and the predictor can stop trying to guess them."
    )


if __name__ == "__main__":
    main()
