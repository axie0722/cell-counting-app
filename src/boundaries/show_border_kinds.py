"""Split Alice's NCM outline into the three KINDS of border it is made of, and draw them.

Alice, 2026-08-12: *"the main issue is that it can't identify some of ncm border that is still
technically in the middle of the tissue but there is a visible separation between that border and
the other part of the brain slice. i want the model to be able to identify that based on my
annotations"*.

WHY MEASURE BEFORE TRAINING. Her rule already supplies two of the three borders and they are the
ones that work: the straight Field L line, and "keep going to the tissue edge". The third kind --
where she stops in the MIDDLE of the tissue because another structure begins -- has no rule at all,
and it is the only part a net needs to supply. The printed share of INTERIOR is the honest size of
that problem, and the picture is there to check the definition against her eye before it becomes a
loss function.

  TISSUE EDGE   within EDGE_UM of non-tissue         -- her rule already gets this for free
  FIELD L LINE  the LONG STRAIGHT RUN she drew        -- her rule already gets this for free
  INTERIOR      everything else                       -- THE TARGET. Nothing predicts this today.

HOW THE FIELD L LINE IS FOUND, AND THE TWO WAYS I GOT IT WRONG FIRST. Alice, 2026-08-13: *"now that
im thinking about it the first time you showed me the red regions weren't bad it was just you marked
a lot of the field l lines as red so can't you just use those but just say that the long straight
line should be yellow? i don't think the model was necessary"* -- and she is right on both counts.

    attempt 1   distance to find_band's line          mislabelled 42% of her Field L border as red,
                                                      because find_band's line sits up to 332 um off
    attempt 2   distance to a 1-D histogram peak       still 54% of red had Field L just outside it,
                                                      because it assumes her line is parallel to the
                                                      band; tilt and curve leak out of the slack
    attempt 3   the direction the border FACES         got that to 27%, still wrong, and it threw
                                                      away 63% of the target to do it
    this one    the LONGEST EDGE OF HER POLYGON        exact, per section, no threshold at all

The last one is not a new idea: `ps6.field_l_direction` already relies on it, because she traces the
curved tissue edge as many short segments and the Field L border as ONE long straight run. So the
answer was in her drawing the whole time and the three attempts above were all inference from a
rasterised mask that had thrown the vertices away. When a derived label keeps failing, check whether
the raw annotation still holds the thing being derived.

Usage:
  python show_border_kinds.py
  python show_border_kinds.py 100        # a stricter EDGE_UM, in microns
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt

import ps6
from ceiling_lines import DATA_PATH
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS
from review_rule import CANVAS_UM, solid_tissue

# How close to non-tissue counts as "she drew on the tissue edge". Her outlines sit a median of 18 um
# inside the true edge, so this has to be a few pixels of slack rather than zero -- but not so much
# that a genuine interior border 200 um in gets absorbed into the easy case.
EDGE_UM = 60.0

# Slack around her drawn Field L run. This is now only drawing jitter, not a search window, because
# the run itself is exact -- it is an edge of her own polygon.
LINE_UM = 60.0

KINDS = {
    "edge": (90, 200, 255),      # blue   -- tissue edge, already solved
    "line": (255, 230, 60),      # yellow -- her long straight Field L run, already solved
    "interior": (255, 60, 60),   # red    -- THE TARGET
}


def field_l_run(vertices):
    """Her Field L border as a two-point segment, plus how clearly it stands out.

    The longest edge of the polygon IS the Field L border -- see the module docstring. The ratio to
    the second-longest edge is returned as well, not for a decision but as a warning light: a section
    where Field L was traced as two or three segments would show a ratio near 1, and its yellow would
    then cover only part of the run.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)

    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    order = np.argsort(-lengths)

    best = int(order[0])
    ratio = lengths[best] / max(lengths[order[1]], 1e-9)

    return np.stack([vertices[best], vertices[(best + 1) % len(vertices)]]), ratio


def distance_to_segment(rows, columns, start, end):
    """Perpendicular distance from each point to a SEGMENT, in pixels.

    A segment rather than an infinite line, because the Field L border stops where she stopped
    drawing it; past its ends the boundary is tissue edge or interior, and an infinite line would
    paint those yellow too. The clip to [0, 1] is what makes it a segment: it pins the closest point
    to inside the run instead of letting it slide off the end.
    """
    points = np.stack([rows, columns], axis=1).astype(np.float64) - start
    along = end - start
    length_squared = float(along @ along)

    if length_squared <= 0:
        return np.linalg.norm(points, axis=1)

    fraction = np.clip(points @ along / length_squared, 0.0, 1.0)

    return np.linalg.norm(points - fraction[:, None] * along, axis=1)


def border_kinds(truth, tissue, field_l=None):
    """Every pixel of her outline, labelled 'edge', 'line' or 'interior'.

    `distance_transform_edt(solid)` gives, for each pixel inside the section, how far it is from the
    nearest non-tissue pixel -- so reading it at the boundary pixels says how close she drew to the
    tissue edge, in one pass and without any loop.

    `field_l` is her drawn Field L run as two points IN CANVAS PIXELS. Passing None leaves the yellow
    kind empty, which is only useful for showing what the tissue-edge test alone explains.
    """
    solid = solid_tissue(tissue)

    # Her boundary: the drawn pixels that touch the outside of the drawn region.
    boundary = truth & ~binary_erosion(truth, iterations=1, border_value=0)

    to_edge = distance_transform_edt(solid) * CANVAS_UM
    on_edge = boundary & (to_edge <= EDGE_UM)

    on_line = np.zeros_like(boundary)

    if field_l is not None:
        # Tested only on pixels the tissue edge does not already explain, so a section whose Field L
        # run happens to graze the outline's edge cannot lose blue to yellow.
        rows, columns = np.nonzero(boundary & ~on_edge)

        if rows.size:
            near = distance_to_segment(rows, columns, field_l[0], field_l[1]) * CANVAS_UM <= LINE_UM
            on_line[rows[near], columns[near]] = True

    interior = boundary & ~on_edge & ~on_line

    return {"edge": on_edge, "line": on_line, "interior": interior}, boundary


def canvas_field_l(image_name, pixel_size_um, scale_um):
    """Her Field L run in CANVAS pixels, or (None, nan) if this section has no drawn outline.

    Her vertices are in original image pixels; every mask in DATA_PATH is on the 8 um canvas. One
    multiply converts between them, and getting it wrong would put the segment somewhere off the
    section entirely -- which is why the ratio and the picture are both printed.
    """
    outlines = ps6.load_region_vertices(ps6.region_path(image_name))

    if "NCM" not in outlines or len(outlines["NCM"]) < 3:
        return None, float("nan")

    segment, ratio = field_l_run(outlines["NCM"])

    return segment * (pixel_size_um / scale_um), ratio


def panel(green, tissue, kinds, field_l=None):
    base = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    solid = solid_tissue(tissue)
    rgb[solid & ~binary_erosion(solid, iterations=1, border_value=0)] = (70, 70, 70)

    # Interior last so it is never painted over by the two easy kinds.
    for name in ("edge", "line", "interior"):
        mask = kinds[name]
        if mask.any():
            rgb[mask] = KINDS[name]

    image = Image.fromarray(rgb)

    # The run itself, faint and dashed, so an obviously wrong segment is caught by eye.
    if field_l is not None:
        draw = ImageDraw.Draw(image)
        (y0, x0), (y1, x1) = field_l
        for step in np.arange(0, 1, 0.02):
            if int(step * 50) % 2:
                continue
            draw.point([(x0 + (x1 - x0) * step, y0 + (y1 - y0) * step)], fill=(120, 110, 30))

    return image


def main():
    global EDGE_UM

    if len(sys.argv) > 1:
        EDGE_UM = float(sys.argv[1])

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    panels, captions, ratios = [], [], []
    totals = {name: 0 for name in KINDS}
    all_boundary = 0

    print(f"edge slack {EDGE_UM:.0f} um, line slack {LINE_UM:.0f} um\n")

    for index, (stem, name) in enumerate(zip(stems, images)):
        green = data["inputs"][index, 0]
        tissue = data["inputs"][index, 1] > 0.5
        truth = data["labels"][index] == 1
        pixel_size = float(data["pixel_sizes"][index])

        if not truth.any():
            continue

        field_l, ratio = canvas_field_l(name, pixel_size, scale_um)
        kinds, boundary = border_kinds(truth, tissue, field_l)

        counts = {key: int(mask.sum()) for key, mask in kinds.items()}
        total = max(int(boundary.sum()), 1)

        for key, count in counts.items():
            totals[key] += count
        all_boundary += total

        # How deep the interior border sits, since that is what a net would have to output.
        depths = (distance_transform_edt(solid_tissue(tissue)) * CANVAS_UM)[kinds["interior"]]
        depth_text = f"  {np.median(depths):.0f} um in (max {depths.max():.0f})" if depths.size else ""

        share = counts["interior"] / total
        flag = "  <-- Field L not one clear run" if ratio < 2.0 else ""
        print(
            f"  {stem[:32]:34s} interior {share:5.0%}   edge {counts['edge'] / total:4.0%}   "
            f"line {counts['line'] / total:4.0%}   longest edge {ratio:4.1f}x{depth_text}{flag}"
        )

        ratios.append(ratio)
        panels.append(panel(green, tissue, kinds, field_l))
        captions.append((stem[:30], f"interior {share:.0%}  longest edge {ratio:.1f}x"))

    print(f"\n  pooled: interior {totals['interior'] / max(all_boundary, 1):.0%}   "
          f"edge {totals['edge'] / max(all_boundary, 1):.0%}   "
          f"line {totals['line'] / max(all_boundary, 1):.0%}")
    print(f"  longest NCM edge is {np.min(ratios):.1f}x to {np.max(ratios):.1f}x the second-longest "
          f"(median {np.median(ratios):.1f}x)")

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new("RGB", (COLUMNS * cell_width, HEADER_HEIGHT + rows * cell_height), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        "ALICE'S OWN NCM OUTLINE, split by WHAT KIND OF BORDER each piece is.",
        fill="white",
    )
    draw.text(
        (4, 18),
        "RED = INTERIOR: she stopped in the middle of the tissue. Nothing in the pipeline predicts "
        "this -- it is the thing to learn.",
        fill=(255, 120, 120),
    )
    draw.text(
        (4, 32),
        f"YELLOW = the LONG STRAIGHT RUN she drew, i.e. Field L, taken straight off her polygon.  "
        f"BLUE = tissue edge (within {EDGE_UM:.0f} um).  Both already handled.",
        fill=(255, 230, 60),
    )

    for position, (image, (title, note)) in enumerate(zip(panels, captions)):
        row, column = divmod(position, COLUMNS)
        x = column * cell_width
        y = HEADER_HEIGHT + row * cell_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 2), title, fill="white")
        draw.text((x + 4, y + image.height + 14), note, fill=(255, 150, 150))

    out = Path("grids") / "border kinds.png"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
