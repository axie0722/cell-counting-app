"""CMM as Alice describes it: a straight line on the far side of Field L, then a polygon off that line.

Alice, 2026-08-18: *"can you predict the cmm region yourself by drawing a straight line on the other
side of field l and then drawing a polygon that stays within the appropriate borders"*, and then
*"you don't have to follow my outlines exactly since i told you they were a bit arbitrary ... like
depending on what else is there"*.

WHY THIS REPLACES THE RADII POLYGON. `polygon_solid` hangs 16 fitted radii off find_band's BOX
CENTROID, and the centroid is the error: 177/123 um from her centroid, worth +0.077 IoU if it were
perfect, and no correction for it exists ([[cmm-centre-has-no-predictor]]). A line has no centroid to
get wrong. find_band already measures BOTH edges of the bright Field L patch -- `ncm_offset`, which
the NCM rule stops on, and `cmm_offset`, the side CMM is on -- so her line costs nothing.

WHAT `diag_cmm_sides.py` MEASURED, and it is why this shape and not another:

  near border   4 um rms about a straight line, 2 um on 11 of 15 sections. She draws it with a ruler.
                It sits 125 um INSIDE the bright patch (same sign on 14 of 15), so the line is
                `cmm_offset` plus one constant.
  the two ends  NOT perpendicular to the band. `end high` tilts one way on 15 of 15 sections and
                `end low` the other way on 13 of 15: CMM is a TRAPEZOID, widest against the band and
                narrowing away from it.
  far border    60 um rms and no cue covers it -- the expensive side, and the one she calls arbitrary
                and dependent on what is next to it. So it is the only side allowed to be decided by
                the image: march away from the line until a black separation or the tissue ends.

SIX CONSTANTS, each the MEDIAN over training birds of a number read directly off her outlines -- the
gap from `cmm_offset`, the depth, the two end positions at the line and the two end slopes. None is
fitted to IoU, which is the pattern that has held up here ([[minimum-line-length-beats-the-step]]).
Held out by bird throughout, so no section is ever predicted by a constant its own bird helped set.

Usage:
  python rule_trapezoid.py
  python rule_trapezoid.py --label 1        # the same shape asked of NCM, as a control
"""

import argparse

import numpy as np
from scipy.ndimage import binary_fill_holes, label

from polygon_solid import (
    LEASH_UM,
    MIN_LABEL_PX,
    black_barrier,
    solid_offsets,
    tally,
)
from review_polygon import FAN, rasterise, sections
from score_region_cells import CANVAS_UM

BIN_UM = 60.0
SEED_UM = 120.0
KEYS = ("gap", "depth", "low", "high", "slope_low", "slope_high")

# Whether CMM's line is placed as a fraction of the section's own tissue depth (True) or as the band
# edge plus a constant (False, the rule before 2026-08-19). A module switch rather than an argument
# because `fit` is called from four closures in `main` and threading a flag through all of them would
# be more code than the change itself. `--absolute-line` flips it back so the two are comparable.
RELATIVE_LINE = True
RELATIVE_KEYS = ("gap", "f_low", "f_high", "f_depth", "slope_low", "slope_high")

# Whether the line's POSITION is read as the MIDDLE of her drawn side (True) or as her single most
# extreme pixel (False, the rule before 2026-08-19). See `side_middle` for why the extreme pixel is
# the wrong number. `--corner` flips it back so the two are comparable on the same rows.
SIDE_MIDDLE = True
SIDE_MIN_PX = 20


def band_thickness(section):
    """The bright Field L patch's width in um. `across` = 0 is its NCM-facing edge."""
    band = section["band"]

    return (band["ncm_offset"] - band["cmm_offset"]) * CANVAS_UM


# Field L's width is RESTRICTED at the first `across` bin where this much of the band's own tissue is
# black. Alice, 2026-08-19: *"can you put a width restriction on the field l region"*, and separately, on
# what marks a border at all: *"its usually either a black gap or a visible line separation"*. Fitted by
# sweep against her own two Field L lines: at 15% the median error falls 49 -> 38 um and it fires on 7 of
# 15 sections; at 25% it fires on 2 and gains nothing; below 15% it starts cutting on speckle.
BAND_BLACK = 0.15
BAND_BIN_UM = 20.0
BAND_FLOOR_UM = 200.0


def band_width(section):
    """Field L's width in um: the constant, RESTRICTED where a black gap crosses the band first.

    WHY THIS IS SEPARATE FROM `band_thickness` AND NOT A FIX TO IT. `band_thickness` is 550 um on every
    section and every fitted CMM constant is referenced to it -- `measure_one` returns
    `gap = -band_thickness - line`, so changing the thickness would silently move CMM's line on every
    section without a single row of any table saying so. This is a second, narrower number for callers
    that want the patch itself.

    WHY IT ONLY EVER NARROWS. Graded against her own widths (the distance between her NCM line and her CMM
    line, the one ground truth for this):

        the constant, 550 um                median error 49 um   mean 103   worst 365
        a brightness run, best level        median 110           mean 117   worst 268
        the first black gap, as the width   median 162           mean 249   worst 859
        the first black gap, AS A CAP       median  38           mean 100   worst 365   fires 7/15

    So the image cannot say how wide Field L IS -- a brightness profile is twice as bad as one constant,
    which is the same lesson as [[field-l-is-findable-by-brightness]] (position and angle are findable,
    every extent is a constant). What it CAN say is where the patch definitely STOPS, and a cap is the
    only form that cannot make a section worse than the constant already was.

    The mean is a wash (100 vs 103), so this is a median-and-worst-case improvement, not a large one.
    """
    thickness = band_thickness(section)
    inside = section["tissue"] & (np.abs(section["along"]) <= band_half_um(section))

    if not inside.any():
        return thickness

    dark, across = black_barrier(section["green"])[inside], section["across"][inside]

    # FROM THE NEAR EDGE OUTWARD, and starting past a floor so the band's own near border cannot end it.
    for edge in np.arange(-BAND_FLOOR_UM, -thickness, -BAND_BIN_UM):
        take = (across <= edge) & (across > edge - BAND_BIN_UM)

        if take.sum() >= 30 and dark[take].mean() >= BAND_BLACK:
            return abs(edge)

    return thickness


def band_half_um(section):
    """Half the band's measured extent ALONG Field L, in um. Imported here to keep `band_width` local.

    `rule_span_ends.band_half` is the same number; it is recomputed rather than imported because
    `rule_span_ends` imports this module, and importing it back would be a cycle.
    """
    band = section["band"]

    return float(band["span_high"] - band["span_low"]) / 2.0 * CANVAS_UM


def tissue_extent(section):
    """How deep the section's own tissue runs across the band, in um. What CMM's line distance scales with."""
    across = section["across"][section["tissue"]]

    return float(np.ptp(across)) if across.size else 1.0


def near_border(section, constants, sign):
    """Where the straight line goes, in `across` um.

    TWO RULES, and which one applies is decided by whether a `share` was fitted:

      ABSOLUTE  the band's own edge plus one constant -- for NCM `across = gap`, for CMM
                `across = -band_thickness - gap`. Note that `band_thickness` is a CONSTANT 550 um on
                every section, because find_band supplies only the band's position and angle from the
                image and every extent is a constant ([[field-l-is-findable-by-brightness]]). So this
                already places CMM's line a fixed distance from NCM's, parallel to it -- exactly what
                Alice proposed: *"take ncm's field l line, draw a line parallel to it but at a set
                amount of distance away"*. It was already the rule.

      RELATIVE  the other half of the same sentence, *"(could be relative to the image)"*, which was
                NOT in the rule: CMM's line at `across = -tissue_extent * share`, one fitted fraction
                of the section's own tissue depth. Measured at 93 um against the absolute rule's 115
                (`diag_line_position.py`), and it replaces a PAIR of constants with one
                ([[knob-count-is-a-modelling-choice]]).

    NCM keeps the absolute rule, because there is nothing there to scale: her NCM line sits a median
    +35 um from the band's own edge, and tissue-scaling it scores 88 um either way.

    `constants` may be a plain float for callers that only have a gap; a dict is what enables `share`.
    """
    if not isinstance(constants, dict):
        constants = {"gap": constants}

    if sign < 0:
        if constants.get("share") is not None:
            return -tissue_extent(section) * constants["share"]

        return -band_thickness(section) - constants["gap"]

    return constants["gap"]


def measure_one(section, label_value, mask=None):
    """A region's six trapezoid numbers, read off it in the band frame.

    The ends are fitted on SLICES rather than on boundary pixels: a slice's min and max `along` is
    where the end actually is at that depth, and it does not care how many pixels were drawn there.

    `mask` measures something OTHER than her outline -- the radii polygon, so its own best-fitting
    trapezoid can be drawn and the two shapes compared at the same placement.
    """
    hers = (section["labels"] == label_value) if mask is None else mask

    if not hers.any():
        return None

    along, across = section["along"][hers], section["across"][hers]
    sign = -1.0 if label_value == 2 else +1.0

    near = across.max() if sign < 0 else across.min()
    span = float(np.ptp(across))

    # WHERE HER LINE IS, as distinct from where her furthest pixel is. See `side_middle`.
    line = near

    if SIDE_MIDDLE:
        middle = side_middle(along, across, sign)
        line = near if middle is None else middle

    # HOW FAR THE LINE MOVED, in the offset coordinate the slices use. EVERY returned number has to be
    # referenced to the line the region is DRAWN from, or the shape is displaced rather than corrected:
    # measuring the depth from the corner and drawing it from the middle pushes the far border out by
    # exactly this much, which is what made the ceiling row fall from 0.750 to 0.713 when only the
    # position moved. `span` keeps the slice loop starting at `near` so no drawn pixel is dropped.
    shift = sign * abs(near - line)
    depth = span - abs(near - line)

    offsets, lows, highs = [], [], []

    for start in np.arange(0.0, span, BIN_UM):
        # A slice `start` um away from the line, measured in the direction the region runs.
        far = near + sign * start
        take = ((across <= far) & (across > far + sign * BIN_UM) if sign < 0
                else (across >= far) & (across < far + sign * BIN_UM))

        if take.sum() < 20:
            continue

        offsets.append(sign * (start + BIN_UM / 2.0))
        lows.append(float(along[take].min()))
        highs.append(float(along[take].max()))

    if len(offsets) < 3:
        return None

    slope_low, low = np.polyfit(offsets, lows, 1)
    slope_high, high = np.polyfit(offsets, highs, 1)

    # THE INTERCEPTS ARE AT OFFSET 0, which is `near`. Slide each along its own fitted slope to where
    # the line actually is. `shift` is 0 whenever the position was not moved, so this is a no-op under
    # `--corner` and the two definitions stay comparable on the same rows.
    low, high = low + slope_low * shift, high + slope_high * shift

    return {"gap": (-band_thickness(section) - line) if sign < 0 else line,
            # Her line's distance as a SHARE of this section's own tissue depth, which is the number the
            # relative rule medians. `line` is her line's own `across`, so this inverts the relative
            # branch of `near_border` the same way `gap` inverts the absolute one.
            "share": (-line / tissue_extent(section)) if sign < 0 else None,
            "depth": depth, "low": float(low), "high": float(high),
            "slope_low": float(slope_low), "slope_high": float(slope_high)}


def side_middle(along, across, sign):
    """The MIDDLE of her drawn Field-L-facing side, in `across` um -- not its furthest pixel.

    WHY THIS EXISTS, and it is a real bug and not a refinement. `measure_one` used to read her line's
    position as `across.max()`, her mask's single most EXTREME pixel. But her sides are TILTED in the
    band frame -- a median 93 um of drift per 1000 um down a side that runs ~1600 um -- so the extreme
    pixel is the CORNER where the side meets an end, sitting about half the tilt beyond the side's own
    middle. Measured against `diag_cmm_ends.her_side`, which reads the middle: CMM +82 um mean and
    positive on 15 of 15 sections, NCM -122 um and negative on 15 of 15. Unanimous both ways, so it is
    systematic, and it pushed every fitted position constant outward by that much.

    THE FIX IS TO SAMPLE THE WHOLE SIDE. Walk down the line in bins of `along`; in each bin take her
    outermost pixel; take the MEDIAN of those. Under a pure tilt the median IS the middle, so the tilt
    cancels instead of accumulating, and a median rather than a mean keeps one drawn bump from moving
    the line. Bins with fewer than SIDE_MIN_PX pixels are dropped: at the extreme ends the slanted end
    cuts pinch the region to a few pixels, and a two-pixel bin is not evidence of where a side is.

    Returns None if fewer than three bins survive, in which case the caller keeps the old number --
    there is no side to average and the extreme pixel is as good as anything.

    Only the POSITION is taken from here. `near` still starts the slice loop, because the slices are
    what measure the two ends and moving their origin inward would throw away the pixels beyond it.
    """
    outer = []

    for start in np.arange(along.min(), along.max(), BIN_UM):
        take = (along >= start) & (along < start + BIN_UM)

        if take.sum() < SIDE_MIN_PX:
            continue

        outer.append(float(across[take].max() if sign < 0 else across[take].min()))

    return float(np.median(outer)) if len(outer) >= 3 else None


def tissue_span(section, near, sign, slab=200.0, reach=4000.0):
    """The TISSUE's own extent measured from the line: how far it runs down the band, and how deep.

    Alice's own trick, applied to the shape instead of to a threshold: scale each number by something
    the section itself reports rather than fixing it in microns ([[relative-floor-beats-absolute]]).
    Her CMM's depth spans 402-1234 um and its length 1065-2013 um across fifteen sections, so an
    absolute constant is wrong on most of them by construction -- but a FRACTION of the tissue behind
    the line can be right on all of them at once.

    The depth is measured down the MIDDLE HALF of that span only. At the two ends the tissue curves
    away and its depth collapses, which would drag the number down for a reason that has nothing to do
    with how deep CMM is.
    """
    past = sign * (section["across"] - near)
    slab_mask = section["tissue"] & (past >= 0.0) & (past <= slab)

    if not slab_mask.any():
        return None

    along = section["along"]
    low, high = float(along[slab_mask].min()), float(along[slab_mask].max())
    middle = section["tissue"] & (np.abs(along - (low + high) / 2.0) <= (high - low) / 4.0)
    middle &= (past >= 0.0) & (past <= reach)

    if not middle.any():
        return None

    return low, high, float(past[middle].max())


def measure_relative(section, label_value):
    """Her six numbers again, but the three SIZES expressed as fractions of the tissue at the line."""
    absolute = measure_one(section, label_value)

    if absolute is None:
        return None

    sign = -1.0 if label_value == 2 else +1.0
    span = tissue_span(section, near_border(section, absolute, sign), sign)

    if span is None:
        return None

    low, high, deep = span
    width = max(high - low, 1.0)

    return {"gap": absolute["gap"],
            # Where her ends sit BETWEEN the tissue's own ends, as a fraction of the span.
            "f_low": (absolute["low"] - low) / width,
            "f_high": (absolute["high"] - low) / width,
            "f_depth": absolute["depth"] / max(deep, 1.0),
            "slope_low": absolute["slope_low"], "slope_high": absolute["slope_high"]}


def relative_trapezoid(section, constants, label_value):
    """The trapezoid with its three sizes read off this section's own tissue."""
    sign = -1.0 if label_value == 2 else +1.0
    near = near_border(section, constants, sign)
    span = tissue_span(section, near, sign)

    if span is None:
        return np.zeros(section["tissue"].shape, bool)

    low, high, deep = span
    width = high - low

    return trapezoid(section, {"gap": constants["gap"],
                               "depth": constants["f_depth"] * deep,
                               "low": low + constants["f_low"] * width,
                               "high": low + constants["f_high"] * width,
                               "slope_low": constants["slope_low"],
                               "slope_high": constants["slope_high"]}, label_value)


def fit(training, label_value, relative=False):
    """The median of each number over the training birds. Six knobs, no search."""
    read = measure_relative if relative else measure_one
    seen = [m for m in (read(s, label_value) for s in training) if m is not None]
    keys = RELATIVE_KEYS if relative else KEYS
    out = {k: float(np.median([m[k] for m in seen])) for k in keys}

    # `share` is kept OUT of KEYS so the printed tables do not change shape, and because it exists only
    # for CMM -- for NCM every `share` is None and only the CMM branch of `near_border` ever reads it.
    # Fitted like everything else here: a median over the training BIRDS, never a search.
    shares = [m["share"] for m in seen if m.get("share") is not None]
    out["share"] = float(np.median(shares)) if RELATIVE_LINE and shares else None

    # THE MARCH'S BIAS, fitted like every other constant here: a median over the training birds, no search.
    # `high_offset` stays None unless there is a `high` to correct, which keeps the relative variant (whose
    # end is `f_high`, a fraction of the tissue span) out of this entirely.
    out["high_offset"] = None

    if MARCH_HIGH and label_value == 2 and "high" in out:
        signed = []

        for section, hers in zip(training, (read(s, label_value) for s in training)):
            stop = hers and march_high(section, near_border(section, out, -1.0))

            if stop is not None:
                signed.append(stop - hers["high"])

        out["high_offset"] = float(np.median(signed)) if signed else None

    return out


# Whether CMM's HIGH end stops on black instead of sitting at its fitted constant. Alice, 2026-08-19, on
# YW113 1-1-7: *"the line is clearly going over the black separation so why don't you stop it?"*. Measured
# in `diag_end_march.py`: the raw stop is WORSE than the constant (289 vs 257 um) because it overshoots her
# end on 11 of 15 sections, but the overshoot is a consistent +193 um, so pulling it back by that fitted
# median and averaging with the constant scores 190 um and wins 10/15
# ([[march-wins-only-at-cmms-high-end]]). The LOW end keeps its constant -- nothing marks it
# ([[each-line-has-one-marked-end]]), and marching it scores 408 against 206.
#
# OFF BY DEFAULT, AND THE REASON IS THE WHOLE LESSON. The line really is more accurate -- 257 -> 190 um --
# and the region is WORSE: `trapezoid, all six constants` goes 79 -> 70% of her cells kept and 70 -> 65%
# density for +0.003 IoU, and `extent FROM the polygon` loses IoU outright, 0.499 -> 0.491. The marched end
# is SHORTER than the constant on 9 of 15 sections, and a short region loses her cells while a long one
# only adds area, so a symmetric um error is an asymmetric region error ([[grade-regions-on-cells-not-area]],
# [[area-ratio-hides-a-displaced-region]]). `--no-march` was the flag while this was on; `MARCH_HIGH = True`
# turns it back on. Note the four `radii` rows are byte-identical either way, because the shipped
# configuration reads the line only to slide and cut and never reads its high end at all.
MARCH_HIGH = False

MARCH_STEP_UM = 20.0

# Half-thickness of the strip the march reads, so one stray dark pixel cannot end the line. The same
# 40 um `diag_cmm_ends` uses, kept well under her band's own 473 um width.
MARCH_SLAB_UM = 40.0

# A bin is black enough to stop the march at this share, and off tissue below a plain majority. Both are
# `diag_cmm_ends`'s numbers, and the black share made no difference between 0.15 and 0.50 there.
MARCH_BLACK = 0.15
MARCH_TISSUE = 0.5
MARCH_REACH_UM = 2600.0


def march_high(section, near):
    """Where a march out from the line's middle first meets black or leaves the tissue, going HIGH.

    Written here rather than imported from `diag_cmm_ends.stops` for one reason that is not style: that
    module imports `look_cmm_border`, which imports this one, so the import would be a cycle. The numbers
    are the same numbers -- 20 um bins, a +-40 um slab on the line, a 0.15 black share, a majority tissue
    share -- and `diag_end_march.py` is what graded them.

    Binned in the BAND FRAME, never sampled on the canvas: a canvas sample needs a rounded pixel index and
    a single pixel is exactly what the slab exists to avoid.
    """
    along, across = section["along"], section["across"]
    strip = np.abs(across - near) <= MARCH_SLAB_UM

    if strip.sum() < 200:
        return None

    dark = black_barrier(section["green"])
    grid = np.arange(-MARCH_REACH_UM, MARCH_REACH_UM + MARCH_STEP_UM, MARCH_STEP_UM)
    bins = np.append(grid, grid[-1] + MARCH_STEP_UM)
    total = np.histogram(along[strip], bins=bins)[0].astype(np.float64)
    tissue = np.histogram(along[strip & section["tissue"]], bins=bins)[0]
    black = np.histogram(along[strip & dark], bins=bins)[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        tissue = np.where(total > 0, tissue / total, 0.0)
        black = np.where(total > 0, black / total, 0.0)

    start = int(np.argmin(np.abs(grid)))

    if tissue[start] < MARCH_TISSUE:
        # The line's own middle is off tissue, so the frame is wrong here. Say so rather than marching
        # from wherever the nearest tissue happens to be.
        return None

    index = last = start

    while True:
        index += 1

        if index >= len(grid) or tissue[index] < MARCH_TISSUE or black[index] >= MARCH_BLACK:
            break

        last = index

    return float(grid[last])


def high_end(section, constants, label_value, near):
    """The HIGH end to use: the fitted constant, or that averaged with the pulled-back march.

    The average is not a hedge. The raw stop is a CEILING rather than her line -- on YW113 1-1-7 her own
    end sits 220 um short of the wall the march stops at -- so the stop carries her end's position with a
    bias, exactly like the border dip in [[border-cue-is-calibrated-not-specific]]. Removing the fitted
    bias and then averaging scores 190 um where the raw stop scores 289 and the constant 257.
    """
    if not MARCH_HIGH or label_value != 2 or constants.get("high_offset") is None:
        return constants["high"]

    stop = march_high(section, near)

    return constants["high"] if stop is None else (stop - constants["high_offset"] + constants["high"]) / 2.0


def trapezoid(section, constants, label_value, depth_um=None, shift=0.0, tissue=True):
    """The polygon: one straight line, two slanted ends, and a far border `depth_um` away.

    Everything is in the BAND FRAME, so nothing here refers to the canvas and a section photographed
    at another angle lands in the same place. `shift` slides both ends down the band together, which
    is how the region's ALONG placement can come from somewhere other than a constant -- and it has
    to, because her end positions scatter over 1300 um while the depth and the gap barely move.
    """
    sign = -1.0 if label_value == 2 else +1.0
    near = near_border(section, constants, sign)
    depth = constants["depth"] if depth_um is None else depth_um

    # Signed depth past the line, NEGATIVE inside the region for CMM and positive for NCM, which is
    # what makes one set of end slopes work for either. `sign * past` is then depth into the region
    # for both, so the two bounds are the same expression.
    past = section["across"] - near
    region = (sign * past >= 0.0) & (sign * past <= depth)
    region &= section["along"] >= shift + constants["low"] + constants["slope_low"] * past
    region &= section["along"] <= shift + high_end(section, constants, label_value, near) \
        + constants["slope_high"] * past

    return (region & section["tissue"]) if tissue else region


def along_shift(section, constants, label_value, depth_um, target):
    """How far to slide the trapezoid down the band so its centroid lands on `target`.

    Measured on the UNCLIPPED shape: the tissue mask cuts the ends unevenly, so a centroid measured
    after clipping would chase its own tail and the shift would not be the shift asked for.
    """
    region = trapezoid(section, constants, label_value, depth_um, tissue=False)

    if not region.any():
        return 0.0

    return float(target - section["along"][region].mean())


def grown_from_line(region, section, label_value, constants):
    """The part of the polygon that reaches the line without crossing a black separation.

    `polygon_solid.stop_at_black` keeps the piece nearest find_band's BOX CENTROID, which is the
    thing this rule exists to stop depending on. Here the line is the anchor, so the piece to keep is
    the one that touches the line. Holes are filled afterwards and never before: she draws straight
    over interior speckle ([[regions-must-be-continuous-patches]]).
    """
    allowed = region & ~black_barrier(section["green"])

    if not allowed.any():
        return region

    pieces, count = label(allowed)

    if count <= 1:
        return binary_fill_holes(allowed)

    sign = -1.0 if label_value == 2 else +1.0
    near = near_border(section, constants, sign)
    seed = allowed & (np.abs(section["across"] - near) <= SEED_UM)
    keep = np.unique(pieces[seed & (pieces > 0)])
    keep = keep[keep > 0]

    if not len(keep):
        # The line itself landed on black. Fall back to the biggest piece rather than to nothing.
        counts = np.bincount(pieces.ravel())
        keep = np.array([int(np.argmax(counts[1:]) + 1)])

    return binary_fill_holes(np.isin(pieces, keep))


def slide_onto_line(offsets, section, label_value, gap, leash=None):
    """Move the radii polygon along `across` until its near border sits on the line.

    THE POINT OF THE WHOLE FILE, in one function. The radii polygon already sizes itself correctly
    (x0.96 of her area) and is misplaced; the across half of that misplacement is exactly what a line
    can fix, because the line IS the near border. A translation by (0, delta) in the frame moves every
    projection by `dx * delta`, so this is one number added to sixteen offsets and not a new shape.

    The polygon is measured UNCLIPPED: the tissue mask can cut the near border away, and then the
    edge being pinned would be the tissue's edge rather than the polygon's own.
    """
    sign = -1.0 if label_value == 2 else +1.0
    bare = rasterise(section["projected"], offsets, section["tissue"], False)

    if not bare.any():
        return offsets

    here = section["across"][bare].max() if sign < 0 else section["across"][bare].min()
    delta = float(near_border(section, gap, sign) - here)

    if leash is not None:
        delta = float(np.clip(delta, -leash, leash))

    return offsets + np.array([dx * delta for _, dx in FAN])


def cut_at_line(region, section, label_value, gap):
    """Nothing on the Field L side of the line. *"field l stops at the line i draw!!!"*"""
    sign = -1.0 if label_value == 2 else +1.0

    return region & (sign * (section["across"] - near_border(section, gap, sign)) >= 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=2, help="2 = CMM, 1 = NCM")
    parser.add_argument("--reach", type=float, default=2400.0,
                        help="um the marching far border may travel before the constant depth caps it")
    parser.add_argument("--absolute-line", action="store_true",
                        help="place CMM's line off the band edge plus a constant -- the rule before "
                             "2026-08-19, kept so the relative line can be compared against it")
    parser.add_argument("--march", action="store_true",
                        help="stop CMM's high end on black instead of leaving it at its fitted constant. "
                             "More accurate on the LINE (257 -> 190 um) and worse on the REGION (5-13 "
                             "points of density), which is why it is off by default")
    parser.add_argument("--corner", action="store_true",
                        help="read her line's position as her single furthest pixel -- the rule before "
                             "2026-08-19, which a tilted side puts +82 um outside her CMM border on "
                             "15 of 15 sections. Kept so the middle-of-the-side fix can be compared")
    args = parser.parse_args()

    global RELATIVE_LINE, MARCH_HIGH, SIDE_MIDDLE
    RELATIVE_LINE = not args.absolute_line
    MARCH_HIGH = args.march
    SIDE_MIDDLE = not args.corner

    name = "CMM" if args.label == 2 else "NCM"
    pool = [s for s in sections() if (s["labels"] == args.label).sum() >= MIN_LABEL_PX]

    print(f"\n  {name}: the six numbers read off each of her outlines. Held out by bird below, so no")
    print(f"  section is ever predicted using a constant its own bird helped set.\n")
    print(f"  {'section':24s} {'gap':>7s} {'depth':>7s} {'low':>8s} {'high':>8s} "
          f"{'slope lo':>9s} {'slope hi':>9s}")

    seen = {}

    for section in pool:
        one = measure_one(section, args.label)

        if one is None:
            print(f"  {section['stem'][:22]:24s}   too few slices")
            continue

        seen[section["stem"]] = one
        print(f"  {section['stem'][:22]:24s} {one['gap']:+7.0f} {one['depth']:7.0f} "
              f"{one['low']:+8.0f} {one['high']:+8.0f} {one['slope_low']:+9.2f} "
              f"{one['slope_high']:+9.2f}")

    stack = {k: np.array([m[k] for m in seen.values()]) for k in KEYS}
    print(f"\n  {'median':24s} " + " ".join(f"{np.median(stack[k]):+7.0f}" if k not in
          ("slope_low", "slope_high") else f"{np.median(stack[k]):+9.2f}" for k in KEYS))
    print(f"  {'same sign':24s} " + " ".join(
        f"{max((stack[k] > 0).mean(), (stack[k] < 0).mean()):7.0%}" for k in KEYS))

    saved = dict(np.load("shape_masks.npz"))

    def measure(title, region_for):
        share, ratio, density, _, ratios, out, ious = tally(pool, args.label, name,
                                                            region_for=region_for)
        worst = min(ratios, key=ratios.get)
        leakiest = max(out, key=out.get)
        print(f"  {title:38s} {share:5.0%} x{ratio:5.2f} {density:6.0%} "
              f"{np.mean(list(out.values())):7.0%} {np.mean(list(ious.values())):6.3f} "
              f"x{ratios[worst]:6.2f}  {out[leakiest]:3.0%} {leakiest[:18]:20s}", flush=True)

        return ious

    print(f"\n  {'rule':38s} {'kept':>5s} {'area':>6s} {'dens':>6s} {'off it':>7s} {'IoU':>6s} "
          f"{'worst':>7s}  {'leakiest':22s}")

    # THE CURRENT BEST, re-measured here rather than quoted, so the comparison is same-code.
    from review_polygon import rasterise

    def radii(section, training):
        offsets = solid_offsets(section, training, args.label, name,
                                saved.get(f"{section['stem']}|{name}"), LEASH_UM)

        return rasterise(section["projected"], offsets, section["tissue"], True)

    measure("radii off the box centroid (current)", radii)

    def flat(section, training, depth_um=None, black=False, along_from=None):
        constants = fit(training, args.label)
        shift = 0.0 if along_from is None else along_shift(
            section, constants, args.label, depth_um, along_from(section))
        region = trapezoid(section, constants, args.label, depth_um, shift)

        return grown_from_line(region, section, args.label, constants) if black else region

    # THE SHAPE CEILING. Every one of the six numbers taken from the section itself, so nothing is
    # being predicted at all: this is what a straight line plus two slanted ends plus a flat far
    # border CAN represent. If it is low, no set of constants will rescue the shape.
    measure("trapezoid, all six from HER -- CEILING",
            lambda s, t: trapezoid(s, measure_one(s, args.label), args.label))
    measure("trapezoid, all six constants", lambda s, t: flat(s, t))
    measure("trapezoid, all six + black", lambda s, t: flat(s, t, black=True))
    measure("trapezoid, march to black/tissue", lambda s, t: flat(s, t, args.reach, True))

    # ALONG FROM THE BOX. Her line pins `across`; the box centroid is 177 um off on `along`, which
    # beats a constant that scatters over 1300 um. The box's own along centroid is the only part of
    # find_band's box this keeps.
    def box_along(section):
        box = section["box"][name]

        return float(section["along"][box].mean()) if box.any() else 0.0

    def her_along(section):
        return float(section["along"][section["labels"] == args.label].mean())

    print()
    measure("line + along from the box", lambda s, t: flat(s, t, along_from=box_along))
    measure("line + along from the box + black",
            lambda s, t: flat(s, t, black=True, along_from=box_along))
    measure("line + march + along from the box",
            lambda s, t: flat(s, t, args.reach, True, box_along))
    measure("line + along from HER -- the oracle",
            lambda s, t: flat(s, t, black=True, along_from=her_along))

    # THE HYBRID. Keep the polygon that already sizes itself right; use the line only to fix where it
    # sits across the band, which is the half of the placement error a line can see.
    def pinned(section, training, slide=True, leash=None, cut=True, black=True, oracle=False):
        constants = fit(training, args.label)
        offsets = solid_offsets(section, training, args.label, name,
                               saved.get(f"{section['stem']}|{name}"), LEASH_UM)
        # THE ORACLE LINE: her own near border instead of the predicted one. This separates "pinning
        # the near border is the wrong operation" from "the gap constant is not accurate enough",
        # which the mean cannot tell apart.
        gap = measure_one(section, args.label) if oracle else constants

        if slide:
            offsets = slide_onto_line(offsets, section, args.label, gap, leash)

        region = rasterise(section["projected"], offsets, section["tissue"], True)

        if cut:
            region = cut_at_line(region, section, args.label, gap)

        return grown_from_line(region, section, args.label, constants) if black else region

    print()
    measure("radii, cut at the line only",
            lambda s, t: pinned(s, t, slide=False, black=False))
    measure("radii, slid onto the line",
            lambda s, t: pinned(s, t, cut=False, black=False))
    measure("radii, slid + cut",
            lambda s, t: pinned(s, t, black=False))
    measure("radii, slid + cut + black",
            lambda s, t: pinned(s, t))

    for leash in (50.0, 100.0, 150.0, 250.0):
        measure(f"radii, slid {leash:.0f}um leash + cut + black",
                lambda s, t, leash=leash: pinned(s, t, leash=leash))

    print()
    measure("radii, slid onto HER line -- ORACLE",
            lambda s, t: pinned(s, t, cut=False, black=False, oracle=True))
    measure("radii, slid onto HER line + cut + black",
            lambda s, t: pinned(s, t, oracle=True))

    # WHICH OF THE SIX IS THE BOTTLENECK. The shape's ceiling is 0.750 and the medians reach 0.388, so
    # the whole loss is in predicting the numbers. One oracle per number, everything else the median:
    # a number that lifts IoU on its own is a number worth finding a cue for, and a number that does
    # nothing is a number that can stay a constant forever.
    # SCALED BY THE SECTION'S OWN TISSUE, which is the only version of this shape that can track a
    # 3x spread in depth without being told the answer.
    print()
    relative = {}

    for section in pool:
        one = measure_relative(section, args.label)

        if one is not None:
            relative[section["stem"]] = one

    show = {k: np.array([m[k] for m in relative.values()]) for k in RELATIVE_KEYS}
    print(f"  {'fractions of the tissue':24s} " + " ".join(f"{k:>10s}" for k in RELATIVE_KEYS))
    print(f"  {'median':24s} " + " ".join(f"{np.median(show[k]):10.2f}" for k in RELATIVE_KEYS))
    print(f"  {'mean abs deviation':24s} " + " ".join(
        f"{np.abs(show[k] - np.median(show[k])).mean():10.2f}" for k in RELATIVE_KEYS))
    print()

    def scaled(section, training, black=False):
        constants = fit(training, args.label, relative=True)
        region = relative_trapezoid(section, constants, args.label)

        return grown_from_line(region, section, args.label, constants) if black else region

    measure("trapezoid SCALED by the tissue", lambda s, t: scaled(s, t))
    measure("trapezoid SCALED + black", lambda s, t: scaled(s, t, True))
    measure("trapezoid SCALED, all six from HER",
            lambda s, t: relative_trapezoid(s, measure_relative(s, args.label), args.label))

    # THE SYNTHESIS. The radii polygon is the best-PLACED thing available (IoU 0.555) and the
    # trapezoid is the best SHAPE available (ceiling 0.750). So take the extent from the polygon and
    # the four sides from the trapezoid: nothing new is fitted, the two known-good halves are just
    # asked to do the part each is good at.
    def from_polygon(section, training, black=False, pad=0.0):
        constants = fit(training, args.label)
        offsets = solid_offsets(section, training, args.label, name,
                                saved.get(f"{section['stem']}|{name}"), LEASH_UM)
        poly = rasterise(section["projected"], offsets, section["tissue"], True)

        if not poly.any():
            return poly

        sign = -1.0 if args.label == 2 else +1.0
        near = near_border(section, constants, sign)
        past = sign * (section["across"] - near)
        # The polygon is WIDEST at the line and so is the trapezoid, so the polygon's overall along
        # extent is the right thing to hand over as the ends AT the line.
        constants = dict(constants,
                         low=float(section["along"][poly].min()) - pad,
                         high=float(section["along"][poly].max()) + pad,
                         depth=float(max(past[poly].max(), 1.0)))
        region = trapezoid(section, constants, args.label)

        return grown_from_line(region, section, args.label, constants) if black else region

    # The polygon TRAPEZOIDALISED: its own best-fitting four-sided shape, so the placement is the
    # polygon's and the shape is hers. If the shape is what matters, this is where it shows up.
    def fitted_to_polygon(section, training, black=False, keep_gap=False):
        constants = fit(training, args.label)
        offsets = solid_offsets(section, training, args.label, name,
                                saved.get(f"{section['stem']}|{name}"), LEASH_UM)
        poly = rasterise(section["projected"], offsets, section["tissue"], True)
        its_own = measure_one(section, args.label, poly)

        if its_own is None:
            return poly

        if keep_gap:
            # Its depth measured from ITS near border, but the border moved onto her line.
            its_own["gap"] = constants["gap"]

        region = trapezoid(section, its_own, args.label)

        return grown_from_line(region, section, args.label, its_own) if black else region

    print()
    measure("trapezoid, extent FROM the polygon", lambda s, t: from_polygon(s, t))
    measure("trapezoid, extent FROM the polygon + black", lambda s, t: from_polygon(s, t, True))
    measure("polygon TRAPEZOIDALISED", lambda s, t: fitted_to_polygon(s, t))
    measure("polygon TRAPEZOIDALISED + black", lambda s, t: fitted_to_polygon(s, t, True))
    measure("polygon TRAPEZOIDALISED, her line",
            lambda s, t: fitted_to_polygon(s, t, False, True))

    print(f"\n  ONE ORACLE AT A TIME -- everything else stays the median.\n")

    for key in KEYS:
        def one(section, training, key=key):
            constants = fit(training, args.label)
            hers = measure_one(section, args.label)

            if hers is not None:
                constants[key] = hers[key]

                # An ORACLE row must be an oracle. Handing it her own `high` and then averaging that with
                # a march would measure neither, so the march is switched off wherever the end is pinned.
                if key == "high":
                    constants["high_offset"] = None

            return trapezoid(section, constants, args.label)

        measure(f"trapezoid, HER {key}", one)

    def pair(section, training):
        constants = fit(training, args.label)
        hers = measure_one(section, args.label)

        if hers is not None:
            constants["low"], constants["high"] = hers["low"], hers["high"]
            constants["high_offset"] = None

        return trapezoid(section, constants, args.label)

    measure("trapezoid, HER low + high (the ends)", pair)

    print(f"\n  {BIN_UM:g} um slices, seed {SEED_UM:g} um, march limit {args.reach:g} um.")


if __name__ == "__main__":
    main()
