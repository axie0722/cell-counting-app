"""Each region as tissue intersected with a rotated rectangle: 5 numbers, not 230,400.

WHY A RECTANGLE. The ceiling ladder in ceiling_lines.py builds a region from a Field L
half-plane, two cuts square to it, and an outer cut square to it. Four straight edges, all
mutually perpendicular -- that IS a rotated rectangle, so the ladder's best rung and this
file describe the same shape. Saying "rectangle" instead of "four half-planes" buys two
things:

  - No side ambiguity. The half-plane version had to decide which side of the Field L line
    was the region's interior, and an earlier bug in exactly that sign silently disabled
    the outer cut. |t| <= a is symmetric, so the question never comes up.
  - 5 parameters, in units a network can regress: a centre, two half-lengths, an angle.

The tissue mask still does the work a rectangle cannot: NCM's curved lateral boundary is
the tissue edge, and intersecting recovers it for free. So this is not "approximate the
region by a box" -- it is "the box says where to CUT the tissue".

Angles are carried as (sin 2t, cos 2t) rather than as t. A line has no head or tail, so t
and t+180 are the same line, and a network asked to output degrees has to learn that
351 and -9 are neighbours -- across a discontinuity it cannot represent. Doubling the
angle makes the wrap-around vanish, because 2t advances a full turn as t advances a half
turn.

Usage:
  python regions/region_boxes.py          # verify the box form reproduces the ceiling
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

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

import ps6
from ceiling_lines import (
    DATA_PATH,
    across_axis,
    along_axis,
    extent,
    facing_boundary,
    fit_line,
    largest_blob,
    reconstructions,
)

# The five geometric numbers, plus the angle's two-component encoding, in the order the
# network emits them. Named so nothing depends on remembering the order.
BOX_FIELDS = ["centre_y", "centre_x", "half_along", "half_across", "sin2t", "cos2t"]


def box_from_mask(region, other, tissue):
    """The rectangle that best brackets `region`, oriented along its Field L edge.

    Not a general minimum-area bounding box: the orientation is FIXED to the Field L edge
    rather than chosen to minimise area. That edge is the anatomical landmark Alice draws
    against, so a box aligned to it means the same thing on every section, while a
    minimum-area box could pick a different edge whenever a region is nearly square.
    """
    centre, direction = fit_line(facing_boundary(region, other))

    t = along_axis(tissue.shape, centre, direction)
    across = across_axis(tissue.shape, centre, direction)

    low, high = extent(region, t)
    inner, outer = extent(region, across)

    # Back to canvas coordinates. `direction` is the along unit vector; the across
    # direction is its perpendicular, which is the gradient of `across_axis`.
    perpendicular = np.array([-direction[1], direction[0]])
    middle = (
        centre
        + 0.5 * (low + high) * direction
        + 0.5 * (inner + outer) * perpendicular
    )

    angle = np.arctan2(direction[0], direction[1])

    return {
        "centre_y": float(middle[0]),
        "centre_x": float(middle[1]),
        "half_along": float(0.5 * (high - low)),
        "half_across": float(0.5 * (outer - inner)),
        "sin2t": float(np.sin(2 * angle)),
        "cos2t": float(np.cos(2 * angle)),
    }


def render_box(box, tissue):
    """Rebuild a region mask from the five numbers plus the tissue mask.

    largest_blob after the intersection is what removes the hippocampus: Alice says it is
    separated from NCM by "black stuff in between", so the dark gap already breaks the
    tissue mask there and the hippocampal strip arrives as a smaller component.
    """
    angle = 0.5 * np.arctan2(box["sin2t"], box["cos2t"])
    direction = np.array([np.sin(angle), np.cos(angle)])
    centre = np.array([box["centre_y"], box["centre_x"]])

    t = along_axis(tissue.shape, centre, direction)
    across = across_axis(tissue.shape, centre, direction)

    inside = (np.abs(t) <= box["half_along"]) & (
        np.abs(across) <= box["half_across"]
    )

    return largest_blob(binary_fill_holes(tissue & inside))


def main():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]

    print(
        "Does the rectangle form lose anything against the ceiling ladder's best rung?\n"
        "Both use Alice's own lines, so any gap is the PARAMETERISATION's fault, not\n"
        "prediction's. Checking before training a predictor on these numbers.\n"
    )
    print(f"{'section':30s} {'NCM box':>9} {'NCM rung':>9} {'CMM box':>9} {'CMM rung':>9}")

    gaps = {"NCM": [], "CMM": []}

    for index, stem in enumerate(stems):
        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]
        drawn = {"NCM": labels == 1, "CMM": labels == 2}

        shown = []

        for name, other in (("NCM", "CMM"), ("CMM", "NCM")):
            box = box_from_mask(drawn[name], drawn[other], tissue)
            rebuilt = render_box(box, tissue)
            rung = reconstructions(drawn[name], drawn[other], tissue)["+ outer cut"]

            box_iou = (rebuilt & drawn[name]).sum() / max(
                (rebuilt | drawn[name]).sum(), 1
            )
            rung_iou = (rung & drawn[name]).sum() / max((rung | drawn[name]).sum(), 1)

            gaps[name].append(box_iou - rung_iou)
            shown += [box_iou, rung_iou]

        print(
            f"{stem[:30]:30s} {shown[0]:9.2f} {shown[1]:9.2f} "
            f"{shown[2]:9.2f} {shown[3]:9.2f}"
        )

    print()
    for name in ("NCM", "CMM"):
        difference = np.array(gaps[name])
        print(
            f"{name}: median IoU difference box - rung {np.median(difference):+.3f} "
            f"(worst {difference[np.argmax(np.abs(difference))]:+.3f})"
        )

    print(
        "\nNear zero means the five numbers carry the whole shape, so a predictor that\n"
        "gets them right inherits the ceiling."
    )


if __name__ == "__main__":
    main()
