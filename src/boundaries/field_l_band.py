"""The Field L band: two near-parallel borders with a real gap between them.

score_rungs_cells.py showed the Field L split does essentially all the work of assigning
cells to regions -- 99% of NCM cells and 100% of CMM cells on the correct side, 0% side
errors in every section -- so the borders are the thing worth predicting and the end cuts
are a separate, smaller problem about area.

A BAND, NOT A LINE. An earlier version of this file predicted ONE shared line and split the
tissue with it. Alice, 2026-08-11: "ncm and cmm should not have a shared line". She is right,
and it is anatomy rather than a detail: Field L is a stripe with width, NCM borders one edge
of it and CMM borders the other. One line would push the two regions into contact and
swallow Field L into whichever side won. Measured across the 14 labelled sections, the gap
between her NCM and CMM borders is a median 550 um (296-937), present in every section.

FOUR NUMBERS, THREE DEGREES OF FREEDOM:

    normal            unit vector across Field L, pointing from CMM towards NCM  (2, 1 dof)
    ncm_offset        where NCM's border sits, along the normal                  (1)
    cmm_offset        where CMM's border sits                                    (1)

    NCM      = { p : signed(p) >  ncm_offset }
    Field L  = { p : cmm_offset <= signed(p) <= ncm_offset }   -- belongs to neither
    CMM      = { p : signed(p) <  cmm_offset }

Down from the box model's 10 numbers per section, which matters at 14 labelled sections.

WHY THE NORMAL IS DIRECTED. A line has no head or tail, so train_boxes.py had to encode
angle as (sin 2t, cos 2t) -- raw degrees put a false discontinuity between 351 and -9. A
normal that points at NCM does have a head, so the doubling is unnecessary AND side
assignment comes free instead of being another output to predict.

ONE SHARED NORMAL IS AN ASSUMPTION, AND IT IS TESTED HERE. Alice: "the straight lines of ncm
and cmm regions dont have to be exactly parallel". Measured, they disagree by a median 2.6
degrees (worst 5.4). Small, but "usually small" is not "free", so --tilt scores the same band
with each border keeping its own angle. If the cost is nil the shared normal stays, because
it halves the angle error the predictor can make.

Offsets are measured from the CANVAS CENTRE. A line's distance from a corner swings wildly
as it rotates even when the line barely moves, which would make an easy target look hard.

Usage:
  python field_l_band.py           # shared-normal band vs independent borders, on cells
  python field_l_band.py --bands   # the extracted band parameters, per section
"""

import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

import ps6
from ceiling_lines import DATA_PATH, facing_boundary, fit_line, largest_blob
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices

BAND_FIELDS = ["normal_y", "normal_x", "ncm_offset", "cmm_offset"]


def deepest_point(mask):
    """The pixel furthest inside a mask.

    Used instead of a centroid to answer "which way is NCM": NCM is a curved band, so its
    centroid can fall outside it, and a reference point outside the region would flip the
    normal the wrong way.
    """
    return np.unravel_index(np.argmax(ps6.distance_to_edge(mask)), mask.shape)


def signed_distance(normal, shape):
    """Every pixel's signed distance from the canvas centre, along `normal`.

    One projection serves both borders: they differ only in where they cut it, which is
    exactly what makes the band three degrees of freedom rather than six.
    """
    unit = normal / max(np.linalg.norm(normal), 1e-9)
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])

    return unit[0] * (ys - centre[0]) + unit[1] * (xs - centre[1])


def band_from_labels(labels, shape, tilt=False):
    """Alice's Field L band, as a normal plus two offsets. None if unusable.

    With `tilt`, each border keeps the angle fitted to its own boundary, so the result is
    two independent lines rather than a band -- the arrangement this module's shared normal
    has to justify itself against.

    The shared normal is fitted to BOTH facing boundaries pooled as equals, even though one
    is usually longer. Both are tracings of the same stripe and neither is more
    authoritative, so letting a longer NCM edge outvote CMM's would be arbitrary.
    """
    ncm = labels == 1
    cmm = labels == 2

    if not (ncm.any() and cmm.any()):
        return None

    ncm_edge = facing_boundary(ncm, cmm)
    cmm_edge = facing_boundary(cmm, ncm)

    if ncm_edge.sum() < 2 or cmm_edge.sum() < 2:
        return None

    towards_ncm = np.array(deepest_point(ncm), dtype=float)

    def normal_from(edge):
        centre, direction = fit_line(edge)
        normal = np.array([-direction[1], direction[0]])

        # Either perpendicular is a valid normal; take the one facing NCM so both borders
        # and both variants share one sign convention.
        if normal @ (towards_ncm - centre) < 0:
            normal = -normal

        return normal / max(np.linalg.norm(normal), 1e-9), centre

    if tilt:
        ncm_normal, ncm_centre = normal_from(ncm_edge)
        cmm_normal, cmm_centre = normal_from(cmm_edge)
    else:
        shared, _ = normal_from(ncm_edge | cmm_edge)
        ncm_normal = cmm_normal = shared
        ncm_centre = fit_line(ncm_edge)[0]
        cmm_centre = fit_line(cmm_edge)[0]

    canvas_centre = np.array([shape[0] / 2.0, shape[1] / 2.0])

    band = {
        "normal_y": float(ncm_normal[0]),
        "normal_x": float(ncm_normal[1]),
        "ncm_offset": float(ncm_normal @ (ncm_centre - canvas_centre)),
        "cmm_offset": float(cmm_normal @ (cmm_centre - canvas_centre)),
    }

    if tilt:
        # Only the tilted variant needs a second normal; keeping it out of the shared
        # variant's dict means BAND_FIELDS stays the prediction target.
        band["cmm_normal_y"] = float(cmm_normal[0])
        band["cmm_normal_x"] = float(cmm_normal[1])

    return band


def split_tissue(band, tissue):
    """Tissue cut into (NCM side, CMM side) by the band's two borders.

    The strip between the borders is returned to neither: it is Field L, and a cell in it
    is a cell Alice assigned to neither region.

    largest_blob and a hole fill on each side, as in ceiling_lines.py: each region is one
    piece, so a detached fragment elsewhere on the slide is an error, and the dark gap that
    separates the hippocampus already breaks the tissue mask there.
    """
    ncm_normal = np.array([band["normal_y"], band["normal_x"]], dtype=float)
    ncm_signed = signed_distance(ncm_normal, tissue.shape)

    if "cmm_normal_y" in band:
        cmm_signed = signed_distance(
            np.array([band["cmm_normal_y"], band["cmm_normal_x"]], dtype=float),
            tissue.shape,
        )
    else:
        cmm_signed = ncm_signed

    def finish(mask):
        return largest_blob(binary_fill_holes(tissue & mask))

    return (
        finish(ncm_signed > band["ncm_offset"]),
        finish(cmm_signed < band["cmm_offset"]),
    )


def band_width_um(band, scale_um):
    """How wide the predicted Field L stripe is, in microns.

    Signed on purpose: a NEGATIVE width means the borders have crossed, so the two regions
    overlap. That is anatomically impossible and worth seeing rather than hiding under an
    abs(), because it is the failure mode a predictor with two free offsets can invent.
    """
    return (band["ncm_offset"] - band["cmm_offset"]) * scale_um


def score_split(labels, tissue, ys, xs, band):
    """How a band's split treats the cells, per region.

    Only cells Alice assigned to a region are scored. A cell she put in neither cannot be
    on the wrong side of anything, so counting it would mix a real error with a judgement
    call about where a region stops.
    """
    ncm_side, cmm_side = split_tissue(band, tissue)
    rows = []

    for name, value, predicted in (("NCM", 1, ncm_side), ("CMM", 2, cmm_side)):
        drawn = labels == value

        if not drawn.any():
            continue

        in_drawn = drawn[ys, xs]
        total = int(in_drawn.sum())

        if not total:
            continue

        rows.append(
            {
                "region": name,
                "cells": total,
                "correct_side": int((in_drawn & predicted[ys, xs]).sum()),
                "area_ratio": predicted.sum() / max(drawn.sum(), 1),
            }
        )

    return rows


def main():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(i) for i in data["images"]]
    pixel_sizes = data["pixel_sizes"]
    scale_um = float(data["scale_um"])

    if "--bands" in sys.argv:
        print(
            "Alice's Field L band per section. The normal points from CMM towards NCM,\n"
            "and the offsets say where each border cuts it.\n"
        )
        print(f"{'section':32s} {'normal_y':>9s} {'normal_x':>9s} {'NCM at':>8s} "
              f"{'CMM at':>8s} {'width um':>9s}")

        for index, stem in enumerate(stems):
            band = band_from_labels(data["labels"][index], (480, 480))

            if band is None:
                print(f"{stem[:32]:32s}  no usable boundary")
                continue

            print(
                f"{stem[:32]:32s} {band['normal_y']:9.3f} {band['normal_x']:9.3f} "
                f"{band['ncm_offset']:8.1f} {band['cmm_offset']:8.1f} "
                f"{band_width_um(band, scale_um):9.0f}"
            )

        return

    print(
        "Does one SHARED normal cost anything, versus each border keeping its own angle?\n"
        "Both are built from Alice's outlines, so both are ceilings -- this is about what\n"
        "the predictor should be asked to output, not about accuracy yet.\n"
    )
    print(f"{'section':30s} {'region':6s} {'cells':>6s} {'shared':>7s} {'tilted':>7s} "
          f"{'width um':>9s}")

    rows = []

    for index, stem in enumerate(stems):
        points, source = cells_for(images[index])

        if not len(points):
            continue

        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]

        shared = band_from_labels(labels, tissue.shape, tilt=False)
        tilted = band_from_labels(labels, tissue.shape, tilt=True)

        if shared is None or tilted is None:
            continue

        ys, xs = canvas_indices(points, float(pixel_sizes[index]), tissue.shape)

        shared_rows = {r["region"]: r for r in score_split(labels, tissue, ys, xs, shared)}
        tilted_rows = {r["region"]: r for r in score_split(labels, tissue, ys, xs, tilted)}

        for name in ("NCM", "CMM"):
            if name not in shared_rows or name not in tilted_rows:
                continue

            cells = shared_rows[name]["cells"]

            rows.append(
                {
                    "stem": stem,
                    "region": name,
                    "cells": cells,
                    "shared_correct": shared_rows[name]["correct_side"],
                    "tilted_correct": tilted_rows[name]["correct_side"],
                    "shared_area": shared_rows[name]["area_ratio"],
                    "tilted_area": tilted_rows[name]["area_ratio"],
                    "width_um": band_width_um(shared, scale_um),
                }
            )

            print(
                f"{stem[:30]:30s} {name:6s} {cells:6d} "
                f"{shared_rows[name]['correct_side'] / cells:6.0%} "
                f"{tilted_rows[name]['correct_side'] / cells:6.0%} "
                f"{band_width_um(shared, scale_um):9.0f}"
            )

    table = pd.DataFrame(rows)

    if not len(table):
        raise SystemExit("No sections had both cells and a usable Field L band.")

    print("\n=== pooled, weighted by cell count ===")
    print(f"{'region':6s} {'cells':>6s} {'shared':>8s} {'tilted':>8s} "
          f"{'shared area':>12s} {'tilted area':>12s}")

    for name in ("NCM", "CMM"):
        part = table[table["region"] == name]

        if not len(part):
            continue

        cells = part["cells"].sum()

        print(
            f"{name:6s} {cells:6d} {part['shared_correct'].sum() / cells:7.0%} "
            f"{part['tilted_correct'].sum() / cells:7.0%} "
            f"{(part['shared_area'] * part['cells']).sum() / cells:11.2f}x "
            f"{(part['tilted_area'] * part['cells']).sum() / cells:11.2f}x"
        )

    widths = table.drop_duplicates("stem")["width_um"]

    print(
        f"\nField L band width: median {widths.median():.0f} um, "
        f"range {widths.min():.0f}-{widths.max():.0f} um.\n"
        "If shared matches tilted, one normal carries both borders and the predictor\n"
        "outputs four numbers with three degrees of freedom."
    )


if __name__ == "__main__":
    main()
