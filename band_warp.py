"""Resample every section into the FOUND BAND's frame, so all 14 look pose-identical.

Alice, 2026-08-11: "is there a way to combine find_band with a model? i think the main problem is
the region outlines i feel like the general area its landing in is pretty good".

She is right, and which_error.py measured it. Replacing find_band's box with a PERFECT box --
true angle, true border, true depth, true centre, true length, all read off her own outlines --
keeps only 74% of her NCM cells, against the real finder's 83%. Placement is nearly solved (true
angle is worth 4 points, true border 1). THE BOX SHAPE IS NOW THE BINDING CONSTRAINT, so the job
left for a model is shape, not position.

WHY WARP FIRST INSTEAD OF FEEDING THE RAW IMAGE. train_band.py did the latter and collapsed to
one constant vertical line for all 14 sections (region-nets-collapse-to-a-constant-line): with 14
examples there is not enough signal to learn rotation and shape at once. So hand the pose over
for free. After this warp:

    row    = across the band. A FIXED row is the NCM-facing border; NCM is above it, Field L in
             the 550 um below it, CMM below that.
    column = along the band, zero at the band's centre.
    LH sections are mirrored, so left and right hemispheres stack.

A net trained here cannot learn rotation, because there is no rotation left to learn. That is the
point. band_atlas.py is the non-parametric version of the same idea and already ties the box on
NCM (86% of cells at x1.05 area against 83% at x1.03), which is the number to beat.

Nearest-neighbour sampling, deliberately: the labels and tissue mask are categorical, and the
green channel is going to a net that reads texture, so inventing in-between values would be
worse than a slightly blocky edge.

Usage:
  python band_warp.py        # warp all 14 and draw the check sheet
"""

from pathlib import Path

import numpy as np

from ceiling_lines import DATA_PATH
from find_band import BAND_WIDTH_UM, CANVAS_UM, dorsal_on_canvas, find_band, is_left_hemisphere

# The canonical frame. 20 um/px is 2.5x coarser than the 8 um canvas -- regions are 1000+ um
# objects, so this loses nothing that matters and makes the net small enough to train on a laptop.
FRAME_UM = 20.0

# Reach, in microns, measured from the band's NCM-facing border (across = 0) and its centre
# (along = 0). Chosen to hold every section's tissue: the deepest NCM in the set reaches 2180 um
# and the deepest CMM -3120.
ACROSS_LOW_UM = -4000.0
ACROSS_HIGH_UM = 2600.0
ALONG_HALF_UM = 3200.0

ROWS = int((ACROSS_HIGH_UM - ACROSS_LOW_UM) / FRAME_UM)
COLUMNS = int(2 * ALONG_HALF_UM / FRAME_UM)


def frame_grid():
    """The (across, along) microns of every pixel of the canonical frame.

    Row 0 is the top of the picture and holds the HIGHEST across, so that NCM appears above the
    band the way it does in the atlas picture and the way anyone would draw it.
    """
    across = ACROSS_HIGH_UM - (np.arange(ROWS) + 0.5) * FRAME_UM
    along = -ALONG_HALF_UM + (np.arange(COLUMNS) + 0.5) * FRAME_UM

    return np.meshgrid(across, along, indexing="ij")


def sample_points(band, shape, mirror):
    """Where each canonical pixel comes from in the section, as (y, x) canvas indices.

    The inverse of band_atlas.band_frame: that turns canvas pixels into band coordinates, this
    turns band coordinates back into canvas pixels, so the two can never drift apart.
    """
    across, along = frame_grid()

    normal = np.array([band["normal_y"], band["normal_x"]])
    tangent = np.array([-normal[1], normal[0]])

    if mirror:
        along = -along

    # signed_distance measures from the canvas CENTRE along each axis, so inverting it is just
    # centre + across * normal + along * tangent, in canvas pixels.
    centre = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])

    a = across / CANVAS_UM + band["ncm_offset"]
    b = along / CANVAS_UM + (band["span_low"] + band["span_high"]) / 2.0

    ys = centre[0] + a * normal[0] + b * tangent[0]
    xs = centre[1] + a * normal[1] + b * tangent[1]

    return ys, xs


def valley_map(green, tissue):
    """How much darker a pixel is than its surroundings -- the dark-lamina channel.

    Alice, 2026-08-12: "theres a black line separating the cmm region from another part of the
    brain that you should be able to identify." She is right, and nothing else in the input can
    see it: the tissue mask only separates tissue from background, so a region can cross a real
    anatomical border and still be "inside tissue".

    A GLOBAL DARKNESS THRESHOLD CANNOT FIND IT -- tried, and it flags the whole dim flank because
    NCM itself is dim. The line is dark only RELATIVE TO ITS NEIGHBOURS, so this is a difference
    of Gaussians: blur wide, blur narrow, subtract. Positive means "darker than the local
    background", whatever that background's level is. General rule worth keeping: for a
    curvilinear feature use local contrast, never an absolute level.

    Two details that decide whether the line survives to the net:

      * Computed on the CANVAS (8 um/px), not in this 20 um/px frame. The lamina is only 1-2
        canvas pixels wide, so a frame-resolution DoG has nothing left to detect.
      * Then THICKENED with a maximum filter, because warp() samples nearest-neighbour. A
        one-pixel line sampled onto a 2.5x coarser grid is mostly missed; a three-pixel ridge
        lands every time. Thickening loses precision the net cannot use anyway.

    Scaled by a robust spread rather than the max, so one bright speck cannot flatten the map.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    fine = gaussian_filter(green.astype(np.float32), 1.5)
    wide = gaussian_filter(green.astype(np.float32), 14.0)
    valley = wide - fine

    inside = valley[tissue]
    spread = np.percentile(inside, 99) - np.percentile(inside, 50) if inside.size else 1.0
    valley = np.clip(valley / max(float(spread), 1e-6), 0.0, 2.0)

    return maximum_filter(valley * tissue, size=3)


def warp(image, band, mirror, fill=0):
    """Pull one array into the canonical frame. Anything off the edge becomes `fill`."""
    ys, xs = sample_points(band, image.shape, mirror)

    iy = np.rint(ys).astype(int)
    ix = np.rint(xs).astype(int)

    inside = (iy >= 0) & (iy < image.shape[0]) & (ix >= 0) & (ix < image.shape[1])

    out = np.full((ROWS, COLUMNS), fill, dtype=image.dtype)
    out[inside] = image[iy[inside], ix[inside]]

    return out


def unwarp(frame_mask, band, shape, mirror):
    """Push a canonical-frame mask back onto the section -- the step that closes the loop.

    Done by mapping the SECTION's pixels into the frame and reading the mask there, not by
    scattering the frame's pixels outward: scattering leaves holes wherever the frame is coarser
    than the canvas, and this frame is 2.5x coarser.
    """
    from band_atlas import band_frame

    along, across = band_frame(band, np.ones(shape, bool), mirror)

    row = np.rint((ACROSS_HIGH_UM - across) / FRAME_UM - 0.5).astype(int)
    column = np.rint((along + ALONG_HALF_UM) / FRAME_UM - 0.5).astype(int)

    inside = (row >= 0) & (row < ROWS) & (column >= 0) & (column < COLUMNS)

    out = np.zeros(shape, bool)
    out[inside] = frame_mask[row[inside], column[inside]]

    return out


def warped_sections():
    """Every section, warped: green, tissue, labels, plus what is needed to get back."""
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    out = []

    for index, (stem, name) in enumerate(zip(stems, images)):
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])
        band = find_band(
            data["inputs"][index, 0], tissue, dorsal_on_canvas(name, pixel_size, scale_um)
        )

        if band is None:
            continue

        mirror = is_left_hemisphere(stem)

        out.append(
            {
                "stem": stem,
                "image": name,
                "pixel_size": pixel_size,
                "band": band,
                "mirror": mirror,
                "shape": tissue.shape,
                "green": warp(data["inputs"][index, 0], band, mirror, fill=0.0),
                "valley": warp(
                    valley_map(data["inputs"][index, 0], tissue), band, mirror, fill=0.0
                ),
                "tissue": warp(tissue.astype(np.uint8), band, mirror) > 0,
                "labels": warp(data["labels"][index].astype(np.uint8), band, mirror),
                "canvas_tissue": tissue,
                "canvas_labels": data["labels"][index],
            }
        )

    return out


def band_rows():
    """The two rows the band occupies: its NCM-facing border, and its far side."""
    return (
        int(ACROSS_HIGH_UM / FRAME_UM),
        int((ACROSS_HIGH_UM + BAND_WIDTH_UM) / FRAME_UM),
    )


def panel(section):
    from PIL import Image, ImageDraw

    from review_predictions import DRAWN_COLOURS, outline

    base = (np.clip(section["green"], 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    rgb[outline(section["tissue"])] = (70, 70, 70)

    for name, value in (("NCM", 1), ("CMM", 2)):
        rgb[outline(section["labels"] == value)] = DRAWN_COLOURS[name]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    border, far = band_rows()
    draw.line([(0, border), (COLUMNS, border)], fill=(255, 255, 255))
    draw.line([(0, far), (COLUMNS, far)], fill=(230, 200, 60))
    draw.line([(COLUMNS // 2, 0), (COLUMNS // 2, ROWS)], fill=(90, 90, 90))

    return image


def main():
    from PIL import Image, ImageDraw

    sections = warped_sections()
    panels = [panel(s) for s in sections]

    columns = 5
    rows = int(np.ceil(len(panels) / columns))
    width, height = panels[0].width, panels[0].height + 16

    sheet = Image.new("RGB", (columns * width, 34 + rows * height), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        "EVERY SECTION IN THE FOUND BAND'S FRAME. WHITE line = the band's NCM-facing border, "
        "same row in all 14 by construction. YELLOW = its far side.",
        fill="white",
    )
    draw.text(
        (4, 18),
        "Cyan = Alice's NCM, magenta = her CMM. If the warp is right these outlines nearly "
        "stack -- and what is left over is the shape a model has to learn.",
        fill=(180, 220, 255),
    )

    for index, image in enumerate(panels):
        row, column = divmod(index, columns)
        x, y = column * width, 34 + row * height
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), sections[index]["stem"][:34], fill="yellow")

    Path("grids").mkdir(parents=True, exist_ok=True)
    sheet.save("grids/band frame.png")

    # How well do the outlines stack? Spread of each region's extent, in the frame.
    print("  region  border row      depth um            length um")
    for name, value in (("NCM", 1), ("CMM", 2)):
        depths, lengths, borders = [], [], []

        for section in sections:
            drawn = section["labels"] == value
            if not drawn.any():
                continue
            rs, cs = np.nonzero(drawn)
            top, bottom = np.percentile(rs, (5, 95))
            left, right = np.percentile(cs, (5, 95))
            borders.append(bottom if name == "NCM" else top)
            depths.append((bottom - top) * FRAME_UM)
            lengths.append((right - left) * FRAME_UM)

        print(
            f"  {name}     {np.median(borders):5.0f} +-{np.std(borders):4.1f}   "
            f"{np.median(depths):5.0f} +-{np.std(depths):4.0f}      "
            f"{np.median(lengths):5.0f} +-{np.std(lengths):4.0f}"
        )

    print(f"\n  band's border row is {band_rows()[0]} by construction")
    print("\nwrote grids/band frame.png")


if __name__ == "__main__":
    main()
