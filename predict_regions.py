"""Predicted NCM and CMM outlines for one section, as editable polygons.

Alice, 2026-08-19: *"can you implement this into the app so that when the user presses draw outlines you open
these predicted outlines, but also give them the option to edit/redraw them if they are bad"*. This is the part
that turns the rule into polygons; `draw_regions.py` is the part that puts them in front of you.

TWO DIFFERENT RULES, one per region: NCM from `rule_ncm.rule_border` (fill outward to the first wide gap, far
border read off the image, about 0.81 IoU) and CMM from `rule_cmm_rays.rays` (a fitted shape, because no cue
marks its far border on most sections, about 0.53). Swapping them scores about 0.34 -- see `region_masks`.

IT NO LONGER WAITS FOR THE DORSAL CLICK. Everything the rule does happens in the Field L band's own frame, and
which side of that band is NCM is one bit that the image does not carry -- no cue predicts it better than
answering "the same side" every time ([[diag_dorsal_bit]]). So with no click, `default_dorsal` GUESSES that
side, says it guessed, and `draw_regions.py` offers Ctrl-F to flip it: a swapped NCM and CMM is unmistakable
on sight, so the eye that is already looking at the section is the best classifier available. Alice, 2026-08-19:
*"can you activate the annotaions without the dorsal click?"*. The click is still what decides dNCM from vNCM,
which nothing here guesses and no outline would reveal.

THREE THINGS HAVE TO MATCH THE TRAINING CANVAS EXACTLY or every fitted constant shifts under the rule:
  * the green channel is percentile-stretched (1st to 99.5th, over the tissue) BEFORE resampling;
  * resampling is to 8 um/px, and the tissue mask is computed at FULL resolution first, because the smoothing
    radius in `ps6.compute_tissue` is a pixel distance tuned at the original scale;
  * the result is zero-padded into a 480x480 square at the TOP LEFT -- and this one is easy to miss, because
    the rule reads percentiles over the WHOLE array (`dark_percent`), so padding changes its darkness floor.

Usage:
  python predict_regions.py "Gre595_RH_3-1-4_NCM_Slide 1.TIF"     # write nothing, just report
  python predict_regions.py Gre595_RH_3-1-4 --check               # score against her own outlines
"""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage.measure import approximate_polygon, find_contours

import ps6
from band_atlas import band_frame
from find_band import band_axis, bright_patch, find_band, is_left_hemisphere
from prepare_regions import CANVAS, HIGH_PERCENTILE, LOW_PERCENTILE, PIXELS_PER_UM, pad_into, resample
from rule_cmm_rays import rays
from rule_ncm import rule_border

SCALE_UM = PIXELS_PER_UM
LABELS = {"NCM": 1, "CMM": 2}
# BESIDE THIS FILE, for the same reason as count_cells.MODEL_PATH: these fitted numbers are part
# of the program, not of the folder someone happens to be standing in. This one is worse than a
# missing model, though, and that is why it is worth a paragraph.
#
# fitted_constants below treats "the file is not there" as "then fit it from the labelled corpus",
# which is right on this machine and impossible anywhere else -- a stranger has no labelled
# sections, so a bare relative name would turn "drawing outlines on another computer" into a
# FileNotFoundError from deep inside review_polygon, in a napari window that then closes. The file
# is 535 bytes; it ships with the code, and now it is found regardless of where the app was
# started from. See package_app.py, which will not build a package without it.
CONSTANTS_PATH = str(Path(__file__).resolve().parent / "region_constants.json")

# HOW ROUGH THE TRACED OUTLINE MAY BE, in canvas pixels, so ~16 um. The rule's border is a staircase of
# per-ray stops; tracing it literally gives hundreds of vertices, and a polygon with hundreds of vertices is
# not editable by hand -- which is the whole point of putting it in napari. Alice can drag ~30.
TOLERANCE_PX = 2.0

# THE BIGGEST ARRAY THE CANVAS MAY BE BUILT FROM, in megapixels, before it is decimated by an integer stride.
# Alice, 2026-08-19: *"for all the images that need regions the napari window doesn't open properly and then
# crasehs"* -- and her whole computer went down with it. Two uploads in birds/ are 243 and 217 megapixels,
# three to seven times every section ever processed, and a full-resolution float32 plane of that is ~1 GB
# before the blur, the tissue mask, and the stretch each take another.
#
# 100 KEEPS EVERY MEASURED SECTION AT STRIDE 1. The largest labelled section is 88 Mpx, so the whole corpus
# the constants were fitted on is untouched -- decimation only ever happens on something bigger than anything
# that has been graded, which is exactly where the alternative is a crash.
WORKING_MEGAPIXELS = 100.0


def working_step(shape):
    """The stride to decimate by before any full-resolution work. 1 for every section ever measured."""
    return max(1, int(np.ceil(np.sqrt(shape[0] * shape[1] / 1e6 / WORKING_MEGAPIXELS))))


def too_big_for_canvas(shape, pixel_size):
    """The reason this section cannot go on the canvas at all, or "" if it fits.

    CHECKED BEFORE ANY WORK. `pad_into` raises when the section is larger than the canvas, and that exception
    used to land at module level in `draw_regions.py`, so the window appeared and the process died with the
    reason printed to a terminal the app hides. The two NCL uploads are 4.3 x 5.3 mm; the canvas is 3.84 mm
    square. They are also a different brain region, so there is nothing here to predict for them anyway.

    RAISING `CANVAS` IS NOT THE FIX. `dark_percent` and every constant fitted through it read percentiles over
    the WHOLE array, so a bigger canvas moves the darkness floor under all of them ([[a-relative-floor-beats-
    absolute]] is fitted against that floor). A section this size needs its own fit, not a bigger frame.
    """
    if max(shape[0], shape[1]) * pixel_size / SCALE_UM <= CANVAS:
        return ""

    return (f"this section is {shape[0] * pixel_size / 1000:.1f} x {shape[1] * pixel_size / 1000:.1f} mm, "
            f"bigger than the {CANVAS * SCALE_UM / 1000:.1f} mm square the rule was fitted in, so its "
            f"outlines have to be drawn by hand")


def fitted_constants(label_value, refresh=False):
    """CMM's shape constants, fitted once over every labelled section and cached to disk.

    NCM does not appear here: its rule reads its border off the image and has no fitted shape.

    Cached because fitting means loading the whole labelled corpus and placing a band on all fifteen
    sections, which is half a minute for numbers that do not change between runs. NO hold-out here, unlike
    every grading script: a section being annotated is not in the corpus, so there is nothing to hold out.
    Opening one of the fifteen labelled sections is the exception -- its own outlines helped fit these
    constants, so treat what appears on it as flattering.
    """
    cache = json.loads(Path(CONSTANTS_PATH).read_text()) if Path(CONSTANTS_PATH).exists() else {}
    key = str(label_value)

    if key in cache and not refresh:
        return cache[key]

    from polygon_solid import MIN_LABEL_PX
    from review_polygon import sections
    from rule_trapezoid import fit

    print(f"  fitting the {label_value} constants over the labelled sections (once, then cached)...")
    pool = [s for s in sections() if (s["labels"] == label_value).sum() >= MIN_LABEL_PX]
    cache[key] = fit(pool, label_value)
    Path(CONSTANTS_PATH).write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"  wrote {CONSTANTS_PATH} from {len(pool)} sections")

    return cache[key]


def canvas_for(image_path, green=None):
    """The section on the 8 um canvas the rule was fitted on: stretched green, tissue, and its pixel size.

    `green` is the FULL-RESOLUTION green channel when the caller already has it -- `draw_regions.py` has just
    loaded it to show you the image, and re-reading the TIF costs about a minute for an identical array.

    RAISES ValueError, with a sentence fit to show someone, when the section cannot go on the canvas. Callers
    in front of a window must catch it: this runs before napari's event loop starts, so an exception here
    closes the window instead of reporting anything.

    BOUNDED MEMORY, at a cost of nothing on any section ever measured. Everything below works on a decimated
    VIEW (no copy) when the section is over `WORKING_MEGAPIXELS`, with the tissue blur radius divided by the
    same stride -- `tissue_sigma` is a pixel distance, so leaving it alone would smooth 26x too far in microns
    (the mistake `prepare_regions.prepare_section` warns about, in the opposite direction). The stretch is
    then done in place and in float32, which is three fewer full-size temporaries than `(clip - low) / span`.
    """
    pixel_size = ps6.pixel_size_um(image_path) or ps6.REFERENCE_PIXEL_SIZE_UM
    green = ps6.load_green(image_path)[1] if green is None else green

    refusal = too_big_for_canvas(green.shape, pixel_size)

    if refusal:
        raise ValueError(refusal)

    step = working_step(green.shape)
    working = green[::step, ::step]
    working_um = pixel_size * step

    geometry = ps6.geometry_for(image_path)
    geometry.tissue_sigma = geometry.tissue_sigma / step

    if step > 1:
        print(f"  {green.shape[0]}x{green.shape[1]} px is over {WORKING_MEGAPIXELS:.0f} Mpx, so the canvas is "
              f"built from every {step}th pixel ({working.shape[0]}x{working.shape[1]})")

    tissue = ps6.compute_tissue(working, geometry)

    low, high = np.percentile(working[tissue] if tissue.any() else working,
                             [LOW_PERCENTILE, HIGH_PERCENTILE])
    stretched = working.astype(np.float32)
    np.clip(stretched, float(low), float(high), out=stretched)
    stretched -= float(low)
    stretched /= max(1e-6, float(high - low))
    del green, working

    small_green = pad_into(resample(stretched, working_um, SCALE_UM, order=1), CANVAS)
    small_tissue = pad_into(resample(tissue, working_um, SCALE_UM, order=1), CANVAS)

    return small_green, small_tissue > 0.5, pixel_size


def default_dorsal(green, tissue, flip=False):
    """A stand-in for the dorsal click, so the outlines can appear BEFORE you make it. None if no band.

    Alice, 2026-08-19: *"can you activate the annotaions without the dorsal click?"*. The click carries two
    bits and only one of them decides these outlines: which side of Field L is NCM. So this returns a point
    far out on the side `band_axis` happens to call positive, and `flip` returns the other one.

    IT IS A GUESS AND IT IS NAMED ONE. On the fifteen labelled sections that side is right on 13
    ([[diag_dorsal_bit]]), but so is answering "the same side" every time, and no cue beat it -- the sign is
    near-constant in this corpus rather than predictable. A swap puts NCM's outline on CMM, which is obvious
    on sight in cyan and magenta, so the flip key is the honest interface: your eye is the classifier.

    THE OTHER BIT IS NOT GUESSED ANYWHERE. Which END is dorsal splits NCM into dNCM and vNCM, an error no
    outline reveals, so counting still requires the real click.

    FAR ENOUGH OUT THAT THE SIGN DECIDES IT. `orient` compares the point's projection against the patch's own
    middle, and `signed_distance` is measured from the canvas centre, so every middle is within half a canvas
    of zero -- a point 1000 px out cannot land on the wrong side of one.
    """
    found = bright_patch(green, tissue)

    if found is None:
        return None

    normal = band_axis(*found)
    centre = np.array([tissue.shape[0] / 2.0, tissue.shape[1] / 2.0])

    return centre + (-1.0 if flip else 1.0) * normal * 1000.0


def region_masks(section):
    """The two shipped rules, on a section already in the band's frame.

    THEY ARE NOT THE SAME RULE, which is the thing to know before touching this. NCM fills outward to the
    first wide gap and its far border is read from the image (`rule_ncm`). CMM has a fitted SHAPE -- radii off
    a centre, with a straight Field L side -- because no cue marks its far border on most sections
    (`rule_cmm_rays`, [[cmm-is-a-polygon-ncm-is-not]]). Running NCM's rule for CMM, or the reverse, scores
    about 0.34 instead of about 0.8, so a swap here is not a subtle error.
    """
    return {"NCM": rule_border(section)[0],
            "CMM": rays(section, fitted_constants(LABELS["CMM"]), LABELS["CMM"])}


def masks_for(image_path, dorsal_point=None, canvas=None, flip=False):
    """(masks, why, guessed): the rule's NCM and CMM masks on the canvas, `None` if the band cannot be placed.

    `guessed` is True when there was no click and `default_dorsal` supplied the side, because a caller that
    does not pass that on to the person looking at the screen is showing a coin flip as a fact.
    """
    green, tissue, pixel_size = canvas if canvas is not None else canvas_for(image_path)
    point = ps6.load_dorsal_point(ps6.dorsal_path(image_path)) if dorsal_point is None else dorsal_point
    guessed = point is None

    if guessed:
        # ALREADY ON THE CANVAS, so it is not scaled below -- the click is in image pixels, this is not.
        point = default_dorsal(green, tissue, flip)
    else:
        # The same single scale factor as `find_band.dorsal_on_canvas`: the canvas is padded TOP-LEFT, so
        # there is no offset to undo, only a scale.
        point = np.asarray(point, float) * (pixel_size / SCALE_UM)

    if point is None:
        return None, "no Field L patch on this section, so there is no side to guess", guessed

    band = find_band(green, tissue, point)

    if band is None:
        return None, "the Field L band could not be placed on this section", guessed

    along, across = band_frame(band, tissue, is_left_hemisphere(Path(image_path).name))

    return region_masks({"green": green, "tissue": tissue, "along": along, "across": across}), "", guessed


def polygon_of(mask, pixel_size, tolerance_px=TOLERANCE_PX):
    """The outline of `mask` as vertices in FULL-RESOLUTION image pixels, simplified enough to edit by hand.

    The largest contour only. A predicted region can leave a speck behind, and handing napari a second
    polygon that looks like a mistake is worse than dropping it -- the mask itself is not what gets saved.
    """
    contours = find_contours(mask.astype(float), 0.5)

    if not contours:
        return None

    traced = approximate_polygon(max(contours, key=len), tolerance=tolerance_px)

    # `approximate_polygon` repeats the first vertex to close the ring; napari closes polygons itself, and a
    # duplicated vertex sits on top of another one where it cannot be grabbed.
    if len(traced) > 1 and np.allclose(traced[0], traced[-1]):
        traced = traced[:-1]

    return traced * (SCALE_UM / pixel_size) if len(traced) >= 3 else None


def predict(image_path, dorsal_point=None, canvas=None, flip=False):
    """`{"NCM": vertices, "CMM": vertices}` in image pixels, a message when empty, and whether it guessed."""
    masks, why, guessed = masks_for(image_path, dorsal_point, canvas, flip)

    if masks is None:
        return {}, why, guessed

    _, _, pixel_size = canvas if canvas is not None else canvas_for(image_path)
    out = {}

    for name, mask in masks.items():
        vertices = polygon_of(mask, pixel_size) if mask is not None and mask.any() else None

        if vertices is not None:
            out[name] = vertices

    return out, "" if out else "the rule produced no region on this section", guessed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="a path, a filename, or any unique fragment of one")
    parser.add_argument("--check", action="store_true", help="score the prediction against her own outlines")
    parser.add_argument("--refresh", action="store_true", help="re-fit the cached constants")
    parser.add_argument("--flip", action="store_true",
                        help="the OTHER side is NCM (only used when there is no dorsal click to say)")
    args = parser.parse_args()

    from draw_helpers import resolve_image

    image_path = resolve_image(args.image)

    # CMM ONLY: NCM's rule has no fitted shape to cache, it reads its border off the image every time.
    if args.refresh:
        fitted_constants(LABELS["CMM"], refresh=True)

    canvas = canvas_for(image_path)
    predicted, why, guessed = predict(image_path, canvas=canvas, flip=args.flip)

    if not predicted:
        raise SystemExit(f"  no prediction: {why}")

    for name, vertices in predicted.items():
        print(f"  {name}: {len(vertices)} vertices")

    if guessed:
        print(f"  NO DORSAL CLICK on this section, so which side is NCM was GUESSED"
              f"{' (flipped)' if args.flip else ''} -- see default_dorsal")

    if not args.check:
        return

    # AGAINST HER OWN OUTLINES, rasterised on the same canvas, so this checks the whole path -- canvas,
    # padding, band, rule, contour, simplification -- and not just the rule.
    from skimage.draw import polygon2mask

    green, tissue, pixel_size = canvas
    masks, _, _ = masks_for(image_path, canvas=canvas, flip=args.flip)
    hers = ps6.load_region_vertices(ps6.region_path(image_path))

    print(f"\n  {'region':6s} {'IoU vs hers':>12s} {'as a polygon':>13s}")

    for name in LABELS:
        drawn = hers.get(name)

        if drawn is None or len(drawn) < 3 or name not in predicted:
            print(f"  {name:6s} {'she drew none' if drawn is None else 'no prediction':>12s}")
            continue

        scale = pixel_size / SCALE_UM
        truth = polygon2mask(green.shape, np.asarray(drawn, float) * scale)
        simple = polygon2mask(green.shape, np.asarray(predicted[name], float) * scale)

        def iou(a):
            return float((a & truth).sum()) / max(float((a | truth).sum()), 1.0)

        print(f"  {name:6s} {iou(masks[name]):12.3f} {iou(simple):13.3f}")


if __name__ == "__main__":
    main()
