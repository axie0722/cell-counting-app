"""Can NCM be rebuilt from one line plus the tissue mask? Measure the ceiling first.

Alice, 2026-08-10: "the segmentation should be straight lines down the sides of the cmm and
ncm that border field l, and for ncm should just border the tissue", and separately "the
straight lines of ncm and cmm regions dont have to be exactly parallel".

Measured against her 14 drawings, that description holds: NCM's single longest polygon edge
is 38% of its whole perimeter (median), 80% of the REST of its boundary lies on the tissue
edge, and CMM is a 6-vertex polygon whose longest edge sits a median 4 degrees off NCM's
(so nearly parallel, but independent -- they get separate lines here).

WHY THIS FILE EXISTS AND WHAT IT DELIBERATELY DOES NOT DO. The plan has two parts: predict
the lines, then rebuild the regions from them. Building the predictor first would confound
the two -- a 30% area error could mean the prediction was poor OR that rebuilding-from-lines
cannot represent Alice's regions at all. So this hands the reconstruction the TRUE lines,
taken from her own drawings. Whatever error survives is the FLOOR: the best the approach can
ever achieve, with prediction made perfect. If the floor is bad the idea is dead, and it cost
minutes rather than a training run.

The general habit: when a plan has a learned part and a fixed part, test the fixed part with
perfect inputs first. It is the cheapest way to falsify the whole design.

Usage:
  python ceiling_lines.py
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

import ps6

DATA_PATH = "ps6_region_maps.npz"

# Which slice of a region's boundary counts as "facing the other region". Taken as the
# third of boundary pixels closest to the other region, rather than a fixed distance,
# because the gap between the two polygons varies almost 3x across sections.
FACING_FRACTION = 0.35


def boundary(mask):
    return mask & (ps6.distance_to_edge(mask) <= 1)


def facing_boundary(mask, other):
    """The part of `mask`'s boundary that looks towards `other`.

    This is the Field L side -- the straight edge Alice draws. Isolating it matters because
    a line fitted to the WHOLE boundary would be pulled around by the tissue-following
    side, which is not straight and not supposed to be.
    """
    edge = boundary(mask)
    to_other = distance_transform_edt(~other)
    cutoff = np.percentile(to_other[edge], FACING_FRACTION * 100)

    return edge & (to_other <= cutoff)


def fit_line(mask):
    """Total-least-squares line through a set of pixels: a point and a unit direction.

    Ordinary least squares fits y as a function of x and blows up when a line is near
    vertical -- which several of these are, since the Field L axis runs diagonally. Taking
    the first principal component of the point cloud instead has no preferred axis: it
    finds the direction of greatest spread whatever the orientation.
    """
    ys, xs = np.nonzero(mask)
    points = np.column_stack([ys, xs]).astype(np.float64)
    centre = points.mean(axis=0)

    # The principal direction is the leading right-singular vector of the centred cloud.
    _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)

    return centre, vectors[0]


def side_of_line(shape, centre, direction, keep_point):
    """Boolean mask of the half-plane containing `keep_point`.

    The sign of the cross product says which side of the line a point falls on, so
    comparing every pixel's sign to a known interior point's sign picks the correct half
    without needing to reason about angle conventions.
    """
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    cross = direction[0] * (xs - centre[1]) - direction[1] * (ys - centre[0])

    reference = direction[0] * (keep_point[1] - centre[1]) - direction[1] * (
        keep_point[0] - centre[0]
    )

    return (cross > 0) if reference > 0 else (cross < 0)


def along_axis(shape, centre, direction):
    """Each pixel's signed distance ALONG `direction`, measured from `centre`.

    This is the same dot product `side_of_line` takes the sign of, kept as a magnitude
    instead. The sign answers "which side of the line", the magnitude answers "how far
    along it" -- one projection, two different questions. Distance along the Field L axis
    is what "where does NCM stop" means, so the end cuts are just thresholds on this.
    """
    ys, xs = np.mgrid[: shape[0], : shape[1]]

    return direction[0] * (ys - centre[0]) + direction[1] * (xs - centre[1])


def across_axis(shape, centre, direction):
    """Each pixel's signed perpendicular distance FROM the line.

    The exact mirror of `along_axis`: dot product measures along a unit direction, cross
    product measures across it. Both come straight from the fitted line, so no distance
    transform is involved -- an earlier version used `distance_transform_edt(~inner)`
    here, which measures distance TO the half-plane and is therefore zero at every pixel
    inside it. That made the outer cut silently keep everything.

    The sign is arbitrary until `side_of_line` fixes which half is the region's own, so
    callers multiply by that.
    """
    ys, xs = np.mgrid[: shape[0], : shape[1]]

    return direction[0] * (xs - centre[1]) - direction[1] * (ys - centre[0])


def extent(mask, t):
    """How far a region reaches along the projection, ignoring stray pixels.

    Percentiles rather than min/max: a single mislabelled pixel at the far end would drag
    a true extreme by hundreds of microns, and the cut it implies is then wrong for the
    whole region. 0.5% trims that without moving a genuine boundary measurably.
    """
    values = t[mask]

    return np.percentile(values, 0.5), np.percentile(values, 99.5)


def largest_blob(mask):
    """The biggest connected piece. NCM is one region, so extra pieces are errors.

    This is also what removes the hippocampus for free: Alice says it is separated from NCM
    by "black stuff in between", so the dark gap already breaks the tissue mask there and
    the hippocampal strip becomes its own smaller component.
    """
    if not mask.any():
        return mask

    pieces, count = label(mask)

    if count <= 1:
        return mask

    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0

    return pieces == sizes.argmax()


def reconstructions(region, other, tissue):
    """Rebuild one region several ways, from least to most geometry.

    Returned as a ladder rather than a single answer because the useful question is not
    "does this work" but "which constraint bought the improvement". A single final number
    cannot say whether the end cuts mattered or the Field L line was doing all the work,
    and that decides what the predictor has to output.

    Every parameter here comes from Alice's own drawing, so each rung is a CEILING: the
    best that rung can do once prediction is perfect.
    """
    centre, direction = fit_line(facing_boundary(region, other))

    # An interior point far from the line, so the side test is unambiguous: the pixel
    # deepest inside the region.
    deepest = np.unravel_index(
        np.argmax(distance_transform_edt(region)), region.shape
    )
    inner = side_of_line(tissue.shape, centre, direction, deepest)

    # Along the Field L axis: where the region starts and stops.
    t = along_axis(tissue.shape, centre, direction)
    low, high = extent(region, t)

    # Across it: how far the region reaches away from Field L. Flipped so positive always
    # means the region's own side, which makes the outer cut a single upper bound rather
    # than a sign the caller has to reason about.
    across = across_axis(tissue.shape, centre, direction)
    if np.median(across[region]) < 0:
        across = -across

    _, width = extent(region, across)

    def finish(mask):
        return largest_blob(binary_fill_holes(tissue & mask))

    return {
        "line only": finish(inner),
        "+ far end": finish(inner & (t <= high)),
        "+ both ends": finish(inner & (t >= low) & (t <= high)),
        "+ outer cut": finish(inner & (t >= low) & (t <= high) & (across <= width)),
    }


def compare(name, drawn, rebuilt, area_scale):
    error = rebuilt.sum() / drawn.sum() - 1
    iou = (rebuilt & drawn).sum() / max((rebuilt | drawn).sum(), 1)

    return {"rung": name, "error": error, "iou": iou}


RUNGS = ["line only", "+ far end", "+ both ends", "+ outer cut"]


def main():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    scale_um = float(data["scale_um"])
    area_scale = scale_um**2 / 1e6

    print(
        "Rebuilding each region from Alice's OWN lines plus the tissue mask, adding one\n"
        "constraint at a time. Every line is taken from her drawing, so each column is a\n"
        "FLOOR: the best that set of constraints can reach with prediction made perfect.\n"
    )

    results = {"NCM": {rung: [] for rung in RUNGS}, "CMM": {rung: [] for rung in RUNGS}}

    for index, stem in enumerate(stems):
        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]
        ncm, cmm = labels == 1, labels == 2

        for name, region, other in (("NCM", ncm, cmm), ("CMM", cmm, ncm)):
            built = reconstructions(region, other, tissue)

            for rung, mask in built.items():
                results[name][rung].append(
                    compare(rung, region, mask, area_scale)
                )

    for name in ("NCM", "CMM"):
        print(f"=== {name} ===")
        print(f"{'constraints':16s} {'median |err|':>13} {'bias':>7} {'median IoU':>11}")

        for rung in RUNGS:
            errors = np.array([r["error"] for r in results[name][rung]])
            ious = np.array([r["iou"] for r in results[name][rung]])

            print(
                f"{rung:16s} {np.median(np.abs(errors)):12.0%} "
                f"{errors.mean():+7.0%} {np.median(ious):11.2f}"
            )

        print()

    print(
        "For comparison, the per-pixel U-Net managed 50% median area error (cross-bird)\n"
        "/ 43% (best case), with 487,891 parameters instead of about five numbers."
    )


if __name__ == "__main__":
    main()
