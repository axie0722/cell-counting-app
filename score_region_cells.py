"""Grade the region model on CELLS, not on area or IoU.

Alice, 2026-08-11: "its more important for the region to be right since thats where all the
cells are". That is the correct metric and the earlier ones were proxies. Area error is the
density DENOMINATOR, but if a region sits in the wrong place the cells inside it are the
wrong cells -- so the numerator is wrong too. A box with the right area in the wrong
location reports a plausible-looking density that is wrong twice over, which is more
dangerous than an obviously bad prediction.

So this asks, of the cells that Alice's region contains, what fraction does the predicted
region contain -- and what fraction of the predicted region's cells do not belong.

Cells come from the counted-cells sidecars where they exist, and from the hand annotations
otherwise, because only 5 sections have been counted. Coordinates are in FULL-RESOLUTION
image pixels while the region canvas is 480x480 at 8 um/px, so every point has to be mapped
onto the canvas -- see to_canvas().

Usage:
  python score_region_cells.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

import ps6
from ceiling_lines import DATA_PATH, reconstructions
from region_boxes import render_box
from train_boxes import boxes_from_outputs, targets_for

CROSSBIRD_PATH = "grids/box crossbird outputs.npy"
CANVAS_UM = 8.0


def to_canvas(points, pixel_size_um):
    """Full-resolution (y, x) to the 480x480 canvas prepare_regions.py builds.

    Sections are resampled to 8 um/px and placed at the TOP-LEFT of the canvas, with no
    centring, so the mapping is a single scale factor and no offset. Getting this wrong
    would silently shift every cell and make a good region look bad.
    """
    return points * (pixel_size_um / CANVAS_UM)


def cells_for(image_path):
    """Cell coordinates for a section, preferring counted cells over hand annotations."""
    counted = Path(f"{Path(image_path).stem} counted cells.csv")

    if counted.exists():
        table = pd.read_csv(counted)
        if len(table):
            return table[["axis-0", "axis-1"]].to_numpy(float), "counted"

    annotations = Path(f"{Path(image_path).stem} cell centers.csv")

    if annotations.exists():
        table = pd.read_csv(annotations)
        if len(table):
            keep = table
            # Only positives: the annotation files carry rejected candidates too.
            if "label" in table.columns:
                keep = table[table["label"] == 1]
            if len(keep):
                return keep[["axis-0", "axis-1"]].to_numpy(float), "hand"

    return np.empty((0, 2)), "none"


def main():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(i) for i in data["images"]]
    pixel_sizes = data["pixel_sizes"]

    predicted = np.load(CROSSBIRD_PATH)
    truth = np.array(
        [targets_for(data["labels"][i], data["inputs"][i, 1] > 0.5) for i in range(len(stems))]
    )

    print(
        "Grading on CELLS: of the cells inside Alice's region, how many does the\n"
        "predicted region contain, and how many of its cells do not belong?\n"
    )
    print(f"{'section':30s} {'src':8s} {'region':5s} {'cells':>6s} {'kept':>6s} "
          f"{'recall':>7s} {'wrong':>7s}")

    rows = []

    for index, stem in enumerate(stems):
        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]

        points, source = cells_for(images[index])

        if not len(points):
            continue

        canvas = to_canvas(points, float(pixel_sizes[index]))
        ys = np.clip(canvas[:, 0].round().astype(int), 0, tissue.shape[0] - 1)
        xs = np.clip(canvas[:, 1].round().astype(int), 0, tissue.shape[1] - 1)

        boxes_pred = boxes_from_outputs(predicted[index])

        for name, value in (("NCM", 1), ("CMM", 2)):
            drawn = labels == value
            model = render_box(boxes_pred[name], tissue)
            ceiling = reconstructions(drawn, labels == (2 if value == 1 else 1), tissue)[
                "+ outer cut"
            ]

            in_drawn = drawn[ys, xs]
            in_model = model[ys, xs]
            in_ceiling = ceiling[ys, xs]

            total = int(in_drawn.sum())

            if not total:
                continue

            kept = int((in_drawn & in_model).sum())
            extra = int((in_model & ~in_drawn).sum())

            rows.append(
                {
                    "stem": stem,
                    "region": name,
                    "cells": total,
                    "kept": kept,
                    "recall": kept / total,
                    "wrong_share": extra / max(int(in_model.sum()), 1),
                    "ceiling_recall": int((in_drawn & in_ceiling).sum()) / total,
                }
            )

            print(
                f"{stem[:30]:30s} {source:8s} {name:5s} {total:6d} {kept:6d} "
                f"{kept / total:6.0%} {extra / max(int(in_model.sum()), 1):6.0%}"
            )

    table = pd.DataFrame(rows)

    if not len(table):
        raise SystemExit("No sections had cells to grade against.")

    print("\n=== pooled over sections, weighted by cell count ===")
    print(f"{'region':6s} {'cells':>7s} {'in right region':>16s} {'contamination':>14s} "
          f"{'ceiling':>8s}")

    for name in ("NCM", "CMM"):
        part = table[table["region"] == name]

        if not len(part):
            continue

        cells = part["cells"].sum()
        kept = part["kept"].sum()

        print(
            f"{name:6s} {cells:7d} {kept / cells:15.0%} "
            f"{part['wrong_share'].mean():13.0%} "
            f"{(part['ceiling_recall'] * part['cells']).sum() / cells:7.0%}"
        )

    print(
        "\n'in right region' is the fraction of Alice's cells the model's region keeps.\n"
        "'ceiling' is the same figure when the region is rebuilt from HER lines, so it is\n"
        "the best this construction can do. The gap between them is the predictor's fault."
    )


if __name__ == "__main__":
    main()
