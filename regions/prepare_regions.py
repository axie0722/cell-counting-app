"""Turn the 14 drawn sections into a small cached array for region segmentation.

Alice, 2026-08-09: "ok can we do the segmentation of regions now then". Drawing NCM and
CMM by hand is the slowest step left in counting, and unlike cell clicking it is one
smooth outline per region rather than hundreds of decisions -- which is exactly the shape
a segmentation model can learn from 14 examples.

Two labels, not three. `ps6.resolve_regions` already cuts NCM into dNCM and vNCM at the
dorsal marker using the Field L rule, so predicting NCM and CMM yields all three
countable regions, and the subdivision keeps coming from a stated rule rather than from a
second thing that can be wrong. All 14 sections have a dorsal marker (checked), so
nothing blocks that path.

RESOLUTION IS PHYSICAL, NOT A PIXEL COUNT. Resampling every section to a fixed pixel
size would be the obvious move and would be wrong here: the sections are scanned at
0.3088-0.3395 um/px AND genuinely differ in size (NCM spans 0.95-2.63 mm2, a 2.8x real
range). Normalising to a fixed pixel count magnifies the small brains and shrinks the
large ones, erasing a real difference the model should see and inventing a fake one. So
every section is resampled to PIXELS_PER_UM microns per pixel and padded into a common
canvas, leaving physical size intact.

8 um/px sounds drastic -- ~26x coarser than the originals. It is not, for this task. A
region is one smooth blob ~1.5 mm across, so it is still ~190 px wide after resampling,
while 8 um of boundary error sits far below hand-drawing variability: redrawing the CMM
outlines on 2026-08-09 moved the median CMM/NCM area ratio from 0.25 to 0.38, i.e.
hundreds of microns of honest disagreement with the earlier hand. The model cannot be
graded finer than the labels, and this resolution is comfortably finer.

Two input channels: percentile-normalised green, and the tissue mask. Green is normalised
per section for the reason train_cnn.standardise_batch discards brightness -- it is what
lets one model span sections imaged at different exposures. The tissue mask earns its
place because with only 14 examples the silhouette of the section is strong evidence
about where NCM sits, and it is free.

Usage:
  python regions/prepare_regions.py
  python regions/prepare_regions.py --scale 12      # coarser, for a quick experiment
"""

import sys

import numpy as np
from skimage.draw import polygon2mask
from skimage.transform import resize

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
from review_regions import sections

OUTPUT_PATH = "ps6_region_maps.npz"

# Microns per pixel of the resampled canvas. See the module docstring: chosen to sit well
# below hand-drawing variability while making the whole dataset small enough to train on
# CPU in minutes.
PIXELS_PER_UM = 8.0

# Square canvas, in resampled pixels. The largest section is 3.66 mm across, which is 458
# px at 8 um; 480 leaves margin without a second pass. Assertions below catch any future
# section that outgrows it rather than silently cropping a brain.
CANVAS = 480

# Percentiles for the green channel stretch. The high end is not 100 because these
# sections have a thin very-bright tail (dust, edge artifacts) that would otherwise
# compress all the tissue into the bottom of the range -- the same reason
# review_regions.thumbnail and false_positive_gallery.stretch use percentiles.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.5

# Label values in the returned mask. Background is 0 so a plain sum counts region pixels.
LABELS = {"NCM": 1, "CMM": 2}

# NOT LOWERED HERE, ON PURPOSE. ps6.TISSUE_THRESHOLD's 0.10 is absolute and does discard real tissue
# on dim sections (15.9% of Alice's drawn NCM on Gre595_RH_3-1-4, measured in
# check_tissue_threshold.py). The tempting fix is a lower cutoff for the region path. It is the wrong
# fix: Alice, 2026-08-14: *"i thought i told you to remove the tissue mask?"* -- and she is right, the
# region rule should read the image, not a global binary mask. Lowering the threshold would also only
# help in ONE direction. See rule_two_borders.py for the two-sided evidence.


def bird_for_image(image_path):
    """The BIRD name for a section, which is what a holdout must split on.

    Deliberately looked up in ps6.BIRDS rather than parsed from the filename: several
    sections of one animal share a name on purpose, and ps6.BIRDS is where that fact is
    recorded. A name parsed from the path would put two sections of one brain on opposite
    sides of a holdout and report memorisation as generalisation.
    """
    for bird in ps6.BIRDS:
        if bird["image"] == image_path:
            return bird["name"]

    raise KeyError(
        f"{image_path} is not registered in ps6.BIRDS, so its bird is unknown and it "
        f"cannot be held out safely. Add it there first."
    )


def resample(array, pixel_size_um, scale_um, order):
    """Resample to `scale_um` microns per pixel, preserving physical size."""
    factor = pixel_size_um / scale_um
    height, width = array.shape[:2]
    target = (max(1, round(height * factor)), max(1, round(width * factor)))

    return resize(
        array.astype(np.float32),
        target,
        order=order,
        mode="constant",
        cval=0.0,
        anti_aliasing=order > 0,
    )


def pad_into(array, canvas):
    """Place an array at the top-left of a `canvas` x `canvas` square.

    Top-left rather than centred, and this is a real choice. Centring would make the
    padding depend on section size, so the same anatomy would land at a different canvas
    position depending on how big the brain is -- a spurious cue. Top-left keeps the
    origin fixed and lets the model learn position relative to a corner, with size
    carried honestly by how much of the canvas the tissue fills.
    """
    if array.shape[0] > canvas or array.shape[1] > canvas:
        raise ValueError(
            f"section is {array.shape[0]}x{array.shape[1]} at this scale, larger than "
            f"the {canvas}x{canvas} canvas. Raise CANVAS or PIXELS_PER_UM."
        )

    out = np.zeros((canvas, canvas), dtype=array.dtype)
    out[: array.shape[0], : array.shape[1]] = array

    return out


def prepare_section(section, scale_um, canvas):
    """One section as (2-channel input, label map, metadata)."""
    image_path = section["image"]
    pixel_size = ps6.pixel_size_um(image_path) or ps6.REFERENCE_PIXEL_SIZE_UM

    _, green = ps6.load_green(image_path)
    shape = green.shape

    # Tissue at FULL resolution, then downsampled -- not computed on the small image.
    # ps6.TISSUE_SIGMA is a pixel distance tuned at the original scale, so thresholding
    # after the downsample would apply a smoothing radius ~26x too large in microns.
    tissue = ps6.compute_tissue(green, ps6.geometry_for(image_path))

    low, high = np.percentile(green[tissue] if tissue.any() else green,
                              [LOW_PERCENTILE, HIGH_PERCENTILE])
    stretched = (np.clip(green, low, high) - low) / max(1e-6, high - low)
    del green

    small_green = resample(stretched, pixel_size, scale_um, order=1)
    del stretched

    small_tissue = resample(tissue, pixel_size, scale_um, order=1)
    del tissue

    # Rasterise the polygons at FULL resolution then downsample the mask by area, rather
    # than rasterising scaled vertices. Scaling the vertices first would quantise each
    # one to the coarse grid and can pinch a narrow neck of the outline shut; averaging a
    # full-resolution mask instead gives the true fraction of each coarse pixel covered.
    labels = np.zeros(small_green.shape[:2], dtype=np.uint8)
    coverage = {}

    for name, value in LABELS.items():
        vertices = section["polygons"].get(name)

        if vertices is None or len(vertices) < 3:
            raise ValueError(f"{section['stem']} has no usable {name} polygon")

        mask = polygon2mask(shape, vertices)
        fraction = resample(mask, pixel_size, scale_um, order=1)
        del mask

        coverage[name] = fraction

    # Assigned by which region covers the pixel more, so the mutually exclusive labels
    # stay mutually exclusive. NCM and CMM were measured to have zero overlap at full
    # resolution on all 14 sections, so this only ever arbitrates pixels straddling the
    # two boundaries where they touch.
    stacked = np.stack([coverage[name] for name in LABELS], axis=0)
    winner = stacked.argmax(axis=0)
    covered = stacked.max(axis=0) >= 0.5

    for index, name in enumerate(LABELS):
        labels[covered & (winner == index)] = LABELS[name]

    inputs = np.stack(
        [pad_into(small_green, canvas), pad_into(small_tissue, canvas)], axis=0
    )
    label_map = pad_into(labels, canvas)

    # Area in mm2 from the ORIGINAL polygons, not from the resampled mask: this is the
    # number the lab's densities divide by, so it should not inherit resampling error.
    # It also gives training a check that the coarse mask did not lose area.
    from review_regions import polygon_area

    truth_mm2 = {
        name: polygon_area(section["polygons"][name]) * pixel_size**2 / 1e6
        for name in LABELS
    }
    coarse_mm2 = {
        name: float((label_map == value).sum()) * scale_um**2 / 1e6
        for name, value in LABELS.items()
    }

    return (
        inputs.astype(np.float32),
        label_map,
        {
            "stem": section["stem"],
            "image": image_path,
            "bird": bird_for_image(image_path),
            "pixel_size_um": pixel_size,
            "height": small_green.shape[0],
            "width": small_green.shape[1],
            "truth_mm2": truth_mm2,
            "coarse_mm2": coarse_mm2,
        },
    )


def main():
    scale_um = next(
        (
            float(sys.argv[i + 1])
            for i, a in enumerate(sys.argv)
            if a == "--scale" and i + 1 < len(sys.argv)
        ),
        PIXELS_PER_UM,
    )
    canvas = int(round(CANVAS * PIXELS_PER_UM / scale_um)) if scale_um != PIXELS_PER_UM \
        else CANVAS

    found = sections()

    if not found:
        raise SystemExit("No *regions.csv files found.")

    print(
        f"{len(found)} sections at {scale_um:g} um/px on a {canvas}x{canvas} canvas\n"
    )

    inputs, labels, meta = [], [], []

    for section in found:
        one_input, label_map, info = prepare_section(section, scale_um, canvas)

        inputs.append(one_input)
        labels.append(label_map)
        meta.append(info)

        # Area preserved by the resampling is worth printing per section, because it is
        # the one way this step can silently corrupt the target the lab cares about.
        drift = {
            name: info["coarse_mm2"][name] / info["truth_mm2"][name] - 1
            for name in LABELS
        }
        print(
            f"  {info['stem'][:34]:36s} {info['bird']:7s} "
            f"{info['height']:3d}x{info['width']:3d}  "
            f"NCM {info['truth_mm2']['NCM']:.3f} mm2 ({drift['NCM']:+.1%})  "
            f"CMM {info['truth_mm2']['CMM']:.3f} mm2 ({drift['CMM']:+.1%})",
            flush=True,
        )

    inputs = np.stack(inputs)
    labels = np.stack(labels)

    np.savez_compressed(
        OUTPUT_PATH,
        inputs=inputs,
        labels=labels,
        stems=np.array([m["stem"] for m in meta]),
        images=np.array([m["image"] for m in meta]),
        birds=np.array([m["bird"] for m in meta]),
        heights=np.array([m["height"] for m in meta]),
        widths=np.array([m["width"] for m in meta]),
        pixel_sizes=np.array([m["pixel_size_um"] for m in meta]),
        ncm_mm2=np.array([m["truth_mm2"]["NCM"] for m in meta]),
        cmm_mm2=np.array([m["truth_mm2"]["CMM"] for m in meta]),
        scale_um=np.array(scale_um),
    )

    size_mb = inputs.nbytes / 1e6
    print(
        f"\nwrote {OUTPUT_PATH}: inputs {inputs.shape} ({size_mb:.0f} MB in memory), "
        f"labels {labels.shape}"
    )

    birds = sorted({m["bird"] for m in meta})
    print(f"{len(birds)} birds for leave-one-bird-out: {birds}")

    worst = max(
        (abs(m["coarse_mm2"][n] / m["truth_mm2"][n] - 1), m["stem"], n)
        for m in meta
        for n in LABELS
    )
    print(
        f"worst area drift from resampling: {worst[0]:.1%} ({worst[2]} of {worst[1][:30]})"
    )


if __name__ == "__main__":
    main()
