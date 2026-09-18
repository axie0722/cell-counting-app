"""Two borders and fill between them. No tissue mask as a region, no holes, no connectivity.

Alice, 2026-08-14: *"get rid of the tissue mask its never really been necessary my idea was to just
identify the borders, the field l line and the tissue edge, and then fill everything in between"*.

WHAT WAS WRONG BEFORE. Every region rule up to now built the region as A SET OF TISSUE PIXELS: take
the mask, close its interior holes so the patch is continuous ([[regions-must-be-continuous-patches]]),
intersect with the window, keep the largest connected blob. Every failure traced back to that choice:

  filling happened first, so a border she drew along real black space landed inside "solid" tissue
  and got filed as interior ([[ncm-border-is-three-kinds]])
  gaps are partial slits, so connectivity flowed around their ends and refilled everything beyond
  ([[rule_stops_at_black]] bought +0.008 IoU)
  using the gaps directionally then could not tell a separating slit from a hole she draws over, so
  Gre595 fell 0.813 -> 0.608 and the leave-one-bird-out mean got WORSE, 0.742 -> 0.728

WHAT SHE IS DESCRIBING INSTEAD. The region is an INTERVAL on each ray out from the Field L line, not a
set of surviving pixels. Fill from the line outward until the ray leaves the tissue. Consequences,
and they are the whole point:

  a hole BEFORE the stop is inside the interval, so it is filled automatically -- there is no
  fill_small_holes, no size cut, and no way for an interior hole to punch through
  each ray is one interval, so the patch is contiguous by construction -- no largest_blob
  past the edge of the section everything is black forever, i.e. an infinitely long gap, so
  "stop at the first wide gap" ALREADY means "stop at the tissue edge" when there is no gap inside.
  Her two borders are one rule, not two cases

TWO KNOBS, BOTH FITTED LEAVE-ONE-BIRD-OUT ([[blind-grid-is-the-only-honest-metric]]).

  --thickness  how wide black has to be to count as space rather than the dark pixel between two
               bright cells -- the distinction that killed the lamina detector
               ([[lamina-cue-failed-three-ways]]). Measured ALONG the ray, so it is a run length in
               the direction being crossed: simpler than a morphological opening and the thing that
               actually matters.
  --smooth     a running median of the stopping points across NEIGHBOURING rays. A border is a
               smooth curve down the band, not an independent guess per ray. This is what separates
               the two kinds of gap that defeated every earlier attempt: a tear inside NCM that she
               draws over stops one or two rays hundreds of um early, a SPIKE the median discards,
               while a gap that genuinely separates NCM from other tissue stops a long run of
               neighbouring rays together and so is the median's majority. No blob, no size cut, no
               connectivity -- the robustness comes from the border being a curve.

This is NOT a depth cap (*"definitley dont do the cap"*): there is no maximum distance anywhere, and
a ray with no gap runs to the far edge of the section.

SHIPPED SETTING: --thickness 80 --smooth 0 --black 0.005. Honest leave-one-bird-out, 2026-08-14:

  mean IoU  0.742 -> 0.753     mean precision  0.828 -> 0.846     cells kept  80.4% -> 79.7%

and every one of the seven folds picked the SAME setting, which is the part that makes it trustworthy.

THE ALONG WINDOW IS NO LONGER A CONSTANT. `--half 150@0.50`: walk the Field L line outward from the
+-950, by at most 150 um, stopping at the tissue edge OR at an 80 um stretch dimmer than half the band's
own brightness. Alice, 2026-08-14: *"for some of the field l lines expand them until it reaches tissue
edge because some of them don't, or prevent them from crossing the hippocampus"*, then, on the first
version: *"its extending to tissue edge now but jumping over separations that should tell it to stop"*.
Head-to-head against the constant, leave-one-bird-out, 6 of 7 folds pick it:

  IoU  0.759 -> 0.762    recall  0.890 -> 0.911    cells kept  83.4% (from 79.7%)
  NCM density error  -24.1% -> -20.5%              Gre595  0.811 -> 0.845

8 sections better, 5 worse. It does NOT fix YW113_RH_1-1-2, the one section where Alice says the line
crosses the hippocampus: the walk still runs 286 um past her end there and the relative cue never fires,
so that border is not a brightness change. That section and YW113_RH_1-1-10 (-0.063) are the whole loss.

HOW MANY KNOBS YOU SEARCH IS ITSELF A MODELLING DECISION. Sweeping all three at once (7 x 5 x 4 = 140
settings) scored 0.728 -- WORSE than today -- and the entire loss was Gre595, whose fold picked an 8 um
gap and fell 0.813 -> 0.647. Globally 8u/400u/0.005 (0.758) and 80u/0u/0.005 (0.759) are
indistinguishable, yet they differ by 0.16 on that one section: the selection was choosing between two
settings it had no power to separate, and the tie-break decided a huge swing. Hyperparameters overfit
a 7-bird dataset exactly like weights do. Fixing thickness at 80 um on physical grounds -- a gap that
separates two brain areas IS wide, while 8 um is one pixel of noise -- left one free knob, and the
selection became unanimous. Prefer the smaller search when the larger one cannot resolve its own
choices.

WHAT THIS RULE NO LONGER EXPLAINS. The YW113_RH_1-1-2 column moves only 0.767-0.784 across all 140
settings, mask or green. So the black cue is not what limits that section, and neither is it what
limits OR408_RH_1-2-1 (0.590). In the redrawn picture the red barrier now sits almost entirely on the
outer tissue edge -- the cue is working -- while the green region is offset and rotated relative to her
outline, and stops short along the band. The remaining error is WHERE THE BAND IS AND HOW FAR IT RUNS,
not what counts as black.

Usage:
  python regions/rule_two_borders.py
  python regions/rule_two_borders.py --thickness 80 --smooth 0 --black 0.005   # the shipped setting
  python regions/rule_two_borders.py --black mask                              # the old absolute mask
"""

import argparse

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    binary_propagation,
    distance_transform_edt,
    find_objects,
    label,
    maximum_filter1d,
    median_filter,
    minimum_filter1d,
    uniform_filter,
    uniform_filter1d,
)

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

from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import (
    EDGE_PERCENTILES,
    NCM_LENGTH_UM,
    bright_patch,
    dorsal_on_canvas,
    find_band,
    is_left_hemisphere,
)
from hippocampus_check import cells_kept, overlap
from review_rule import CANVAS_UM, solid_tissue
from rule_first_gap import gap_mask

# Ray spacing along the band, and step size out from the line. The along bin is two canvas pixels
# because the band frame is a ROTATED grid: at one-pixel spacing, rounding leaves scattered empty
# cells in the table, and an empty cell would have to be guessed either way. Two pixels fills them.
ALONG_BIN_UM = 16.0
ACROSS_BIN_UM = CANVAS_UM

# Candidate gap widths in um, measured along the ray. inf means "only ever stop at the outer edge of
# the section" -- her rule with no gap logic at all, which is the baseline the gap has to beat.
THICKNESSES = (8.0, 16.0, 24.0, 40.0, 80.0, 160.0, np.inf)

# Candidate widths for the running median of the stopping points, in um along the band. 0 means every
# ray keeps its own stop.
SMOOTHS = (0.0, 100.0, 200.0, 400.0, 800.0)

# How gap width is measured: "open" (a disc must fit inside the black) or "run" (length along the
# ray). See fill_between -- "run" is kept only because the comparison is the evidence.
BARRIER_MODE = "open"

# How far BACK from the band's NCM-facing edge the rays are allowed to start, in um. 0 is the shipped
# behaviour: the region begins exactly at across = 0 and everything she drew on the band side of that
# line is unreachable.
#
# WHY THIS KNOB EXISTS. check_band_extent.py decomposed the 12.27% of her NCM the rule misses:
# 5.43 points are 'behind' the line, 6.44 are outside the fixed along window, and only 0.40 are the
# border logic stopping early. Behind is the single largest bucket, so it is where to look first.
#
# A CONSTANT DOES NOT WORK, MEASURED. The 1st percentile of her NCM's `across` has median -51 um but
# scatters sd 111 um over -357..+105, and on 5 of 14 sections she starts AFTER the line. Swept, a
# constant does exactly what that predicts: recall climbs 0.890 -> 0.921 (the behind pixels really are
# recovered) while precision falls 0.843 -> 0.793, so IoU drops and the honest mean goes 0.753 -> 0.749.
# Six of seven folds pick 0 um. The constant is kept in the sweep anyway, as the control the
# proportional version has to beat -- on this project the plain constant has beaten every adaptive
# scheme so far (check_tissue_threshold.py).
#
# THE PROPORTIONAL VERSION, and why it should be better. A failing constant is not a failing mechanism:
# YW113_RH_1-1-2 gains 0.767 -> 0.874 at 100 um, so the line genuinely IS misplaced there. Her behind
# depth correlates +0.67 (n=14) with the patch's OWN SPREAD (p90-p10 of across over the bright patch),
# which has a physical reading: a diffuse band gives a diffuse p90, so the line lands too deep, by an
# amount proportional to how diffuse it is. Wh175_LH_1-2-1 has the widest patch (1165 um) and the
# deepest overshoot (357 um); Gre595 the narrowest (338 um) and almost none (24 um). A fraction of the
# spread is therefore the per-section correction, and it is the same thing as lowering
# find_band.EDGE_PERCENTILES[1] without having to recompute every band.
#
# (find_band.py:88 reports patch width correlating only +0.12 with her real GAP WIDTH. Different
# quantity -- that is the width of the band, this is the depth of the line. The note is not evidence
# against this.)
#
# AND IT LOSES TOO. Best proportional is 0.05xsp at IoU 0.758 against 0.759 at zero, and the honest
# leave-one-bird-out mean is 0.745 -- worse than both the constant (0.749) and doing nothing (0.753).
# SHIPPED VALUE IS ("um", 0.0). The whole knob is kept only so the negative result stays reproducible.
#
# WHY THE BIGGEST BUCKET IS THE LEAST PROFITABLE ONE, which is the transferable lesson here. Pulling
# the line back 100 um along a 1900 um border adds ~190,000 um2 to EVERY section, including the 5 of 14
# where she already starts past the line. Recall buys 3 points, precision pays 5. Bucket size in an
# error decomposition says where the error IS, not where the profit is: what decides that is the ratio
# of what a change recovers to what it adds wrongly, and immediately along a border those two are
# nearly the same quantity. Spend the effort where the recoverable region is far from the boundary --
# which is the near/far buckets, where OR408's 2411 um span is a genuine per-section fact rather than a
# noisy correction. Cells-kept does tick up (79.7% -> 80.9%) on the pooled table, but goes the other way
# under leave-one-bird-out (79.7% -> 79.0%), so it is not support ([[grade-regions-on-cells-not-area]]).
BACKSETS = (
    ("um", 0.0),
    ("um", 50.0),
    ("um", 100.0),
    ("um", 200.0),
    ("um", 350.0),
    ("frac", 0.05),
    ("frac", 0.10),
    ("frac", 0.15),
    ("frac", 0.20),
)


def backset_um(backset, spread_um):
    """How far behind the line to start this section's rays, in um."""
    kind, value = backset

    return value if kind == "um" else value * spread_um

# Candidate cutoffs for "this is black", on PERCENTILE-NORMALISED green. "mask" means the old
# ps6.compute_tissue mask, kept in the sweep because it is the thing to beat.
#
# WHY NOT THE TISSUE MASK. Alice, 2026-08-14: *"i thought i told you to remove the tissue mask?"* --
# and it was still here, not as a region but as the CUE: ray_table looked "is this black" up in the
# mask, so black meant "below ps6.TISSUE_THRESHOLD", an ABSOLUTE 0.10 on raw green. Measured across
# the 20 sections, that lands at the 13th percentile of a bright section and the 35th of Gre595, the
# dimmest -- and it fails in BOTH directions at once:
#
#   Gre595 (mean green 31.7, dimmest)   real tissue sits near 0.118, barely above the cutoff, so it
#                                       reads as background -> phantom gaps in the MIDDLE of her
#                                       outline -> rays stop early. IoU 0.813 -> 0.593, cells 28%.
#   YW113_RH_1-1-2 (mean 94.8, brightest)  a separation that is obviously black on screen still sits
#                                       near 0.157, ABOVE the cutoff, so the gap is invisible and the
#                                       ray runs straight through it. Alice: *"when there is a clear
#                                       black distinction the model didnt pick up on it"*.
#
# A lower constant only helps the first case and makes the second worse, which is why the one-sided
# "how much of her drawn NCM is called non-tissue" metric picked 0.03 and could not be trusted.
#
# Normalised green fixes both with ONE number because prepare_regions maps each section's own 1st
# percentile to 0 and its 99.5th to 1 -- so "0.15" means the same thing at any exposure. That is also
# exactly the stretch napari applies, which is why Alice's eye is right and the mask is wrong.
# The candidates are LOW because the normalised scale is not the intuitive one, measured:
# outside the tissue, normalised green is 0.000 at the 95th AND 99th percentile in almost every
# section, while her drawn NCM runs 0.09-0.46 at the median. True background really is zero here. A
# first guess of 0.15 called huge amounts of tissue black and scored IoU 0.428 against today's 0.743 --
# picking a threshold on a rescaled axis without looking at its distribution first.
#
# ONE CAVEAT ON THIS CHANNEL. prepare_regions takes its normalisation range from `green[tissue]`, i.e.
# from the very mask this is trying to escape. On Gre595 that mask had already dropped the dim tissue,
# so the low end was set too high and her own NCM has p1 = 0.000 there -- clipped flat into
# background. The cue is much better separated than the absolute mask but not yet independent of it;
# normalising on the whole image instead is the follow-up, and it needs the cache rebuilt.
#
# "mask" IS NO LONGER A CANDIDATE, and the sweep is why. With it in the set, six of the seven
# leave-one-bird-out folds picked a normalised-green level and either gained or held; the seventh
# (Gre595) picked "mask" on the strength of the other six birds' average and collapsed from IoU 0.813
# to 0.458, which alone dragged the honest mean below today's. A knob whose best value flips between
# "read the image" and "read a global binary mask" is not a knob worth having -- the two are different
# rules, not two settings of one rule. Removing it is also what Alice asked for outright:
# *"i thought i told you to remove the tissue mask?"*
# Keep 'mask' reachable by hand (--black mask) so the old behaviour stays reproducible for comparison.
BLACK_LEVELS = (0.005, 0.01, 0.02, 0.04)


def bright_from(green, black_level):
    """"Not black", from normalised green. `black_level` may be "mask" to reuse a supplied mask.

    Deliberately a plain threshold on a NORMALISED channel rather than a per-section adaptive rule:
    exposure rescaling, a fixed per-section percentile and Otsu were all measured against her
    outlines (check_tissue_threshold.py) and all lost to a plain constant. Normalising first and then
    using one constant is the same idea done in the right order.
    """
    return green >= black_level


# How large a neighbourhood "the local background" means, for relative_bright. 400 um is wider than any
# cell cluster and narrower than the brightness difference between Field L and NCM, so a border band
# reads as dark against its own surroundings without the band itself setting the reference.
RELATIVE_WINDOW_UM = 400.0


def relative_bright(green, ratio, window_um=RELATIVE_WINDOW_UM, black_level=BLACK_LEVELS[0]):
    """"Not dark", where dark means dark COMPARED WITH ITS OWN NEIGHBOURHOOD, not below a constant.

    Alice, 2026-08-15, on the band across the top of YW113 1-1-2: it is plainly a black separation to the
    eye, and yet raising the absolute cutoff from 0.02 to 0.08 barely marks it while turning Gre595 and
    OR408 into confetti (`grids/thin black 0.08.png`). The reason is that it reads as black BECAUSE it sits
    against bright tissue -- which is a relative statement, and no absolute number can express it.

    A relative cue was measured once before and rejected on its own: a dip lands +28 um from her border but
    fires on 27% of rays that were already correct ([[border-cue-is-calibrated-not-specific]]). What is new
    is that `long_strips` now has a filter for exactly that failure -- scattered false dips are SHORT.

    THE UNION WITH ABSOLUTE BLACK IS NOT OPTIONAL. Deep outside the section the local background is ~0, so
    a purely relative test calls the empty slide BRIGHT (0 >= ratio * 0), which would delete the tissue edge
    -- the one border this rule has always got right. Dark therefore means relatively dark OR absolutely
    dark, and the outside stays black.
    """
    size = max(int(round(window_um / ACROSS_BIN_UM)), 1)
    background = uniform_filter(np.asarray(green, np.float32), size=size)

    return (green >= ratio * background) & (green >= black_level)


# Candidate half-lengths for the along window, in um. NCM_LENGTH_UM / 2 = 950 is what ships.
#
# WHY SWEEP IT. check_band_extent.py puts 6.44 of the 12.27 points the rule misses OUTSIDE this window
# (2.88 near, 3.56 far) -- the single biggest recoverable share. Her NCM's own along-span runs
# 1461-2411 um against this one constant, and 5 of 14 sections are longer than it.
#
# AND WHY A CONSTANT IS ALL THERE IS. The obvious per-section replacement is the bright patch's own
# measured span, which find_band already computes as span_low/span_high and then discards. Measured, it
# does not work: correlation with her span is +0.57, but an OLS fit scores RMS 256 um out of sample
# against 261 um for simply using the mean -- 5 um on a 1871 um quantity, i.e. the correlation is
# entirely consumed by fitting 2 parameters to 14 points. The patch centre is worse than useless: it is
# the ORIGIN of the along axis by construction (band_atlas.band_frame subtracts it), so there is nothing
# to correlate, and her centre just scatters -470..+366 um around it. inf tests whether the window does
# any real work at all or is only clipping.
#
# THE TWO STRINGS ARE PER-SECTION WINDOWS, not constants. Alice, 2026-08-14: *"for some of the field l
# lines expand them until it reaches tissue edge because some of them don't"*.
#
#   tissue    walk the line outward from the centre in both directions until it leaves the tissue, and
#             use that. Symmetric, and it can SHORTEN a line as well as lengthen it.
#   expand    the same walk, but only ever outward from the 950: max(walk, 950) at each end. The literal
#             reading of "expand them until it reaches tissue edge" -- it cannot shorten anything.
#   expandN   expand, but by at most N um past the 950. A LEASH.
#
# WHY THE LEASH. Measured (check_window.py, 2026-08-14) the unleashed walk lands within ~110 um of her
# own ends at 19 of the 28 ends, and at the other 9 it overruns by 260-970 um. The picture says why, and
# it is not a bug in the walk: at those ends THE TISSUE DOES NOT END. NCM ends and the next structure
# begins, which is exactly what Alice means by *"prevent them from crossing the hippocampus"* -- except
# it is the hippocampus on one section and some other neighbour on the rest. The tissue edge cannot see
# an anatomical border, so until there is a cue for one, a leash bounds the damage: it is ONE knob whose
# two ends of the range are the two rules already measured (0 = the constant, inf = the free walk), so it
# cannot do worse than the better of them on the birds it is fitted on.
#
# AND WHY THE LEASH IS NOT THE ANSWER EITHER. Alice, on the picture: *"its not very good its extending to
# tissue edge now but jumping over separations that should tell it to stop"*. Exactly right, and the same
# diagnosis as the far border: `bright` is green >= 0.005, AS DARK AS THE EMPTY SLIDE, while the
# separations she means are dimmer TISSUE around 0.09. An absolute cutoff cannot see them. So the walk
# also carries a RELATIVE stop -- see line_extent's `fraction` -- and a window is a (leash, fraction)
# pair. fraction 0 is the absolute walk, so the pair (0, 0) reproduces the shipped constant exactly.
HALF_LENGTHS = (
    950.0,
    (150.0, 0.0), (np.inf, 0.0),
    (150.0, 0.50), (300.0, 0.50), (450.0, 0.50), (np.inf, 0.50),
    (300.0, 0.40), (np.inf, 0.40),
)


# How close to the Field L line a pixel must be to count as ON it for the ABSOLUTE tissue-edge test, in
# um. One canvas pixel: wider would average over depth and let a bright blob just off the line hold the
# run open past the tissue edge.
LINE_HALF_UM = 8.0

# For the RELATIVE test the strip is wider, because the two tests want different things from it. "Is
# this still tissue" is a binary and wants the narrowest honest sample; "how bright is this compared with
# the rest of the band" is an average, and a one-pixel line is mostly noise. A separation that crosses
# the line is dark on BOTH sides of it, so averaging a few pixels either way deepens the dip relative to
# the noise instead of blurring it.
STRIP_HALF_UM = 48.0

# How long a dim stretch has to be to count as a separation, in um. THIS IS THE WHOLE DISTINCTION: a dark
# space between two cell bodies is 10-20 um, a lamina is a band. Same 80 um the gap rule already uses, so
# it is not a new knob ([[lamina-cue-failed-three-ways]]).
DIM_WIDTH_UM = 80.0

# Where "how bright is this section's tissue" is measured from: |along| <= this, in um. Well inside the
# line on every section, so the reference can never be contaminated by the ends it is used to find.
REFERENCE_HALF_UM = 600.0


def line_profile(along, across, values):
    """Mean of `values` in a strip along the Field L line: (profile, seen, shift).

    `shift` is the bin index of along = 0, so a bin converts back to um as (bin - shift) * ALONG_BIN_UM.
    Bins no pixel landed in are 0 and not seen -- 0 is the right value for "outside the section", which
    is what an unseen bin means, and it is what lets one rule cover both stops.
    """
    on = np.isfinite(along) & np.isfinite(across) & (abs(across) <= STRIP_HALF_UM)

    if not on.any():
        return None

    rows, columns = np.nonzero(on)
    index = np.rint(along[rows, columns] / ALONG_BIN_UM).astype(np.int64)
    low = int(index.min())
    index = index - low

    total = np.bincount(index)
    total_value = np.bincount(index, weights=values[rows, columns].astype(float))

    return np.where(total > 0, total_value / np.maximum(total, 1), 0.0), total > 0, -low


def dim_stop(profile, shift, fraction, direction):
    """Walking out from along = 0, the um at which the first stretch dimmer than `fraction` starts.

    If nothing qualifies the walk reaches the end of the profile, i.e. no stop. `<=` and not `<` so that
    fraction 0 still stops outside the section, where the profile is exactly 0.
    """
    span = int(REFERENCE_HALF_UM / ALONG_BIN_UM)
    reference = float(np.median(profile[max(shift - span, 0):shift + span + 1]))
    needed = max(int(round(DIM_WIDTH_UM / ALONG_BIN_UM)), 1)

    steps = np.arange(shift, len(profile)) if direction > 0 else np.arange(shift, -1, -1)
    dim = profile[steps] <= fraction * reference
    run = 0

    for position, is_dim in enumerate(dim):
        run = run + 1 if is_dim else 0

        if run >= needed:
            return float((steps[position - needed + 1] - shift) * ALONG_BIN_UM)

    return float((steps[-1] - shift) * ALONG_BIN_UM)


def line_extent(along, across, bright, green=None, fraction=0.0):
    """How far the Field L line runs before it has to stop, either side of the centre, in um: (low, high).

    Alice, 2026-08-14: *"for some of the field l lines expand them until it reaches tissue edge because
    some of them don't, or prevent them from crossing the hippocampus"*. Two stops, and this is both:

      the TISSUE EDGE, from the absolute `bright` test on a one-pixel strip
      a SEPARATION, from the relative `fraction` test on a wider strip -- her *"separations that should
      tell it to stop"*, which the absolute test walks straight over because they are dimmer tissue and
      not empty slide

    THE EARLIER ONE WINS, and neither replaces the other. The two cues want different sampling (see
    LINE_HALF_UM vs STRIP_HALF_UM) and fail differently: the absolute test is never wrong about leaving
    the section and blind to anatomy, the relative test sees anatomy and can be fooled by a dim patch
    inside NCM. Combining them by "whichever fires first" is the same structure fill_between uses for
    the far border. fraction 0 disables the relative test, so this is a strict superset of the old walk.

    THE RUN MUST BE CONTIGUOUS. The min and max `along` over all bright pixels on the line would jump a
    fissure and report the far side of a gap as the tissue edge, which is the same mistake the 'run' gap
    mode made in fill_between -- a crack parallel to the direction being measured reads as open space.
    So the run containing the centre is grown outward one bin at a time and stops at the first bin that
    is not tissue, which is what "walk until you leave the tissue" actually means.
    """
    on = np.isfinite(along) & np.isfinite(across) & (abs(across) <= LINE_HALF_UM)

    if not on.any():
        return None

    rows, columns = np.nonzero(on)
    index = np.rint(along[rows, columns] / ALONG_BIN_UM).astype(np.int64)
    shift = int(index.min())
    index = index - shift

    total = np.bincount(index)
    solid = np.bincount(index, weights=bright[rows, columns].astype(float))
    # A bin no pixel landed in is NOT tissue: an unseen bin means the line has left the section's
    # sampled body, and calling it tissue would let the run continue straight through a hole.
    is_tissue = (total > 0) & (solid * 2 >= total)

    centre = int(np.clip(-shift, 0, len(is_tissue) - 1))

    if not is_tissue[centre]:
        near = np.flatnonzero(is_tissue)

        if not len(near):
            return None

        centre = int(near[np.argmin(abs(near - centre))])

    low = centre
    while low > 0 and is_tissue[low - 1]:
        low -= 1

    high = centre
    while high + 1 < len(is_tissue) and is_tissue[high + 1]:
        high += 1

    edge = float((low + shift) * ALONG_BIN_UM), float((high + shift) * ALONG_BIN_UM)

    if not fraction:
        return edge

    if green is None:
        raise ValueError("a relative line stop needs `green` to build a profile")

    built = line_profile(along, across, green)

    if built is None:
        return edge

    profile, _, centre = built

    # max/min, not the other way round: both stops are distances OUTWARD from 0, so on the low side the
    # nearer stop is the LARGER (less negative) number. Getting this backwards would silently keep the
    # later stop and make the relative cue look like it does nothing.
    return (max(edge[0], dim_stop(profile, centre, fraction, -1)),
            min(edge[1], dim_stop(profile, centre, fraction, +1)))


def window_label(kind):
    """A HALF_LENGTHS entry as at most 7 characters, so the sweep table's columns stay aligned."""
    def short(value):
        return "inf" if not np.isfinite(value) else f"{value:.0f}"

    if isinstance(kind, tuple):
        leash, fraction = kind

        if isinstance(leash, tuple):
            return f"{short(leash[0])}/{short(leash[1])}@{fraction:.1f}"

        return f"{short(leash)}@{fraction:.2f}"

    return "inf" if not np.isfinite(kind) else f"{kind:.0f}u"


def along_window(kind, along, across, bright, green=None, half_um=NCM_LENGTH_UM / 2.0):
    """Turn a HALF_LENGTHS entry into an explicit (low, high) along window, or None for "use half_um".

    A plain number means the old symmetric window and returns None, so the numeric path is bit-for-bit
    what it was and a sweep row for 950 still reproduces the shipped result exactly.

    A (leash, fraction) pair means: walk out from the band's centre and stop at the tissue edge or at a
    separation, then accept that stop if it is within `leash` um of +-half_um. `leash` inf means the
    walk decides outright and the constant is gone.

    IT MOVES BOTH WAYS, and the earlier one-sided version was a misreading. Alice asked to *"expand
    them until it reaches tissue edge because some of them don't"*, so this only ever lengthened -- but
    the ends of her own outlines land INSIDE +-950 on 8 of 14 sections, as far in as 455 um, so a
    lengthen-only walk could not reach them at any leash and no setting of it was ever going to work.
    She then said it directly: *"its not that hard to identify when to stop because there is black gap
    so what seems to be the problem"*. Stopping at the gap means stopping where the gap is, whether that
    is shorter or longer than the constant.

    If the walk fails -- no pixels on the line at all -- it falls back to the constant rather than
    raising. A section with an unusable band frame should score badly on its own merits, not vanish from
    the table and quietly raise the mean of everything that survived.
    """
    if not isinstance(kind, tuple):
        return None

    leash, fraction = kind
    # THE TWO DIRECTIONS ARE NOT THE SAME MEASUREMENT, so the leash may be a pair, (inward, outward). A
    # bare number still means the same allowance both ways, so every earlier sweep row reproduces exactly.
    #
    # Why they differ: the walk SHORTENS when the tissue or a separation really ends before half_um, which
    # is a stop she would draw to as well. It LENGTHENS when the tissue merely keeps going -- and it keeps
    # going through the hippocampus and straight past her end, so out there the walk has no reason to stop
    # at all. Trustworthy inward, untrustworthy outward, and one number could not say that.
    inward, outward = leash if isinstance(leash, tuple) else (leash, leash)
    run = line_extent(along, across, bright, green, fraction)

    if run is None:
        return -half_um, half_um

    # On the low side "outward" is the MORE negative bound, so the two ends are not symmetric in code.
    low = float(np.clip(run[0], -half_um - outward, -half_um + inward))
    high = float(np.clip(run[1], half_um - inward, half_um + outward))

    # A degenerate walk (both stops on the same side of the centre) would produce an empty or inverted
    # window and silently delete the section from the mean. Fall back to the constant instead.
    if not low < high:
        return -half_um, half_um

    return low, high


def ray_table(tissue, along, across, barrier=None, half_um=NCM_LENGTH_UM / 2.0, values=None,
              window=None):
    """Re-index the section as (ray, step) and report which cells are black.

    Not a downsample and not a resample: every pixel in the window is scattered into the cell its
    (along, across) coordinates put it in, and a cell is black when most of its pixels are. Cells no
    pixel landed in are called tissue, because guessing "black" there would invent barriers.

    If `barrier` is given it replaces the majority vote: a cell is black when ANY barrier pixel lands
    in it, because a barrier mask has already decided what counts and must not be diluted by a vote.

    `window` is an explicit (low, high) along range in um and overrides `half_um` when given. It exists
    because the two ends of the Field L line are NOT symmetric about the bright patch's midpoint: the
    line reaches the tissue edge on one side and stops at another structure on the other, so a single
    half-length cannot express both.
    """
    low_um, high_um = (-half_um, half_um) if window is None else (float(window[0]), float(window[1]))

    inside = (
        np.isfinite(along)
        & np.isfinite(across)
        & (across >= 0.0)
        & (along >= low_um)
        & (along <= high_um)
    )

    if not inside.any():
        return None

    rows, columns = np.nonzero(inside)

    # Rays are indexed from the window's own low edge, not from a fixed -950, so widening the window
    # adds rays at both ends instead of shifting every existing ray's index. Getting this wrong would
    # silently re-pair each ray with a different neighbour and change what the median smooths over.
    #
    # THE ORIGIN COMES FROM THE DATA WHEN THE WINDOW IS INFINITE. `-inf` as an origin makes every ray
    # index inf, which casts to a garbage int64 and then asks numpy for an array of ~10^18 rows. That
    # is what the first version did, and the failure was loud (ValueError: Maximum allowed dimension
    # exceeded) only by luck -- the cast itself is merely a RuntimeWarning, so a slightly different
    # shape would have produced silently wrong ray indices instead of a crash. Any sentinel that flows
    # into arithmetic needs an explicit branch, not a hope that it stays out of the way.
    origin = low_um if np.isfinite(low_um) else float(along[rows, columns].min())

    ray = np.rint((along[rows, columns] - origin) / ALONG_BIN_UM).astype(np.int64)
    step = np.rint(across[rows, columns] / ACROSS_BIN_UM).astype(np.int64)

    shape = (int(ray.max()) + 1, int(step.max()) + 1)
    total = np.zeros(shape, np.int32)
    solid = np.zeros(shape, np.int32)

    np.add.at(total, (ray, step), 1)
    np.add.at(solid, (ray, step), tissue[rows, columns].astype(np.int32))

    if barrier is None:
        black = (total > 0) & (solid * 2 < total)
    else:
        black = np.zeros(shape, bool)
        black[ray[barrier[rows, columns]], step[barrier[rows, columns]]] = True

    built = {"rows": rows, "columns": columns, "ray": ray, "step": step, "black": black,
             "seen": total > 0}

    # The MEAN of a real-valued channel per cell, for the relative dip cue. Accumulated here rather
    # than in a second binning pass so the profile is measured in exactly the frame the fill walks --
    # a parallel implementation of the same bins would drift from this one the moment either changed.
    if values is not None:
        value_sum = np.zeros(shape, np.float32)
        np.add.at(value_sum, (ray, step), values[rows, columns].astype(np.float32))
        built["mean"] = np.where(built["seen"], value_sum / np.maximum(total, 1), 0.0)

    return built


def first_wide_gap(black, thickness_um):
    """For each ray, the step at which the first gap of at least `thickness_um` starts.

    A run that reaches the end of the ray is treated as infinitely long, because that is the outside
    of the section -- which is how the tissue edge becomes the same rule as the gap and needs no
    special case. A ray with no qualifying run stops at its end, i.e. nothing is trimmed.
    """
    rays, width = black.shape
    needed = width + 1 if not np.isfinite(thickness_um) else max(int(np.ceil(thickness_um / ACROSS_BIN_UM)), 1)

    stop = np.full(rays, width, np.int64)

    for index in range(rays):
        row = black[index]

        # Run boundaries from the difference of a padded copy: +1 where a run starts, -1 where it ends.
        padded = np.concatenate([[False], row, [False]])
        change = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(change == 1)
        ends = np.flatnonzero(change == -1)

        if not len(starts):
            continue

        lengths = ends - starts

        # A run touching the last step is the outside of the section: unbounded, so always qualifies.
        lengths = np.where(ends == len(row), width + 1, lengths)

        qualifying = np.flatnonzero(lengths >= needed)

        if len(qualifying):
            stop[index] = starts[qualifying[0]]

    return stop


# --- the relative dip cue: MEASURED, REJECTED, KEPT OFF -------------------------------------------
#
# VERDICT FIRST, because the code below is inert by default and the temptation is to switch it on.
#
#   dip   IoU     recall  precision  Gre595      LOBO mean
#   off   0.759   0.890     0.843     0.811        0.753
#   0.15  0.750   0.878     0.845     0.687        0.738
#   0.30  0.737   0.857     0.850     0.592        0.731
#
# and NCM density error -- the number the lab reports -- is -24.1% at EVERY fraction, identical to off
# (check_dip_density.py). It buys at most 1 point of precision, costs 2-3 of recall, collapses Gre595 at
# every setting, and the LOBO folds DISAGREE about it (Gre595's fold picks 0.10, five pick off), which is
# the same overfitting signature that killed the 3-knob sweep. Do not ship it. It is kept because the
# diagnosis below is worth more than the code.
#
# WHY IT EXISTS. bright_from is an ABSOLUTE test: green >= 0.005, i.e. "as dark as the empty slide".
# check_border_profile.py measured what sits at Alice's far border on the 410 rays where the fill runs
# past it: brightness 0.0928, nineteen times the cutoff. Her separation is dimmer TISSUE, so no setting
# of an absolute cutoff can stop there -- and raising it globally is worse, because the dim spaces
# BETWEEN CELLS inside NCM start qualifying and every ray stops a few hundred um in. One number cannot
# separate "dim because it is a border" from "dim because it is between two cells".
#
# The relative version can, and unlike the four cues before it, it beats its own control: at her border
# the tissue is 57.9% darker than the tissue just inside it, against 20.2% at a control depth halfway
# into her NCM on the same ray, and her border is the deeper dip on 84.9% of rays.
#
# THE REFERENCE IS PER RAY. A per-section reference would let a dim section (OR99 is a low-contrast
# section) read as one long border, and a bright one as none at all -- the same absolute-cutoff failure
# one level up. Per ray, only the tissue on that ray sets the bar.
#
# WHY IT FAILS, which is the part worth keeping. check_dip_stop.py split the rays by whether the fill
# currently overruns her border. At fraction 0.30 the cue fires on 56% of the OVERRUNNING rays and lands
# a median +28 um from her hand-drawn line -- inside her own drawing tolerance. It is not miscalibrated;
# it finds her border. It fires on 27% of the rays that were ALREADY CORRECT too, cutting them ~300 um
# short, so it helps 236 rays and damages 314. The fault is SPECIFICITY, and no offset fixes that: an
# offset that helps one group by construction hurts the other.
#
# 600 and not the original 240 because the false firings are systematically shallower; raising the floor
# recovered 0.746 -> 0.750, still short of 0.759 for off. A pinned constant, not a knob.
DIP_BASE_UM = 600.0
DIP_SMOOTH_ACROSS_UM = 56.0
DIP_SMOOTH_ALONG_UM = 80.0

# THE DIP MUST BE WIDE, and this is what makes the cue work at all. The first version stopped at the
# first step below the threshold and collapsed: IoU 0.759 -> 0.652 at fraction 0.30, recall 0.890 ->
# 0.744, Gre595 0.811 -> 0.463, for 1 point of precision.
#
# The measurement that justified the cue compared MEDIANS -- her border dips 57.9%, a control depth
# inside NCM dips 20.2%. But a stopping rule is a FIRST CROSSING, so it is governed by the deepest
# interior dip anywhere along the ray, not the typical one. One dark space between two cells at 250 um
# stops the ray there and the medians never enter into it. A cue can beat its control on average and
# still be useless as a stop.
#
# Width separates the two: a space between cell bodies is 10-20 um, a lamina is a band. Pinned to the
# gap rule's own already-selected 80 um rather than swept, so this stays a ONE-knob change.
DIP_WIDTH_UM = 80.0

# ONE knob, and the smoothing scales above are deliberately NOT knobs: 3 swept knobs on 7 birds scored
# LOBO 0.728 where 1 swept knob scored 0.753. See "HOW MANY KNOBS YOU SEARCH IS ITSELF A MODELLING
# DECISION" in the module docstring. 0.0 means the cue is off, which is the row every other row must beat.
#
# THE RANGE MATTERS AND THE FIRST ONE WAS WRONG. It started at 0.30, which check_dip_stop.py showed is
# past the useful operating point: at 0.30 the cue fires on 27% of the rays that were ALREADY correct,
# at 0.15 on only 6%, while still firing on 41% of the rays that overrun. Sweeping a range that begins
# above the answer looks exactly like a cue that does not work.
DIPS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)


def dip_stops(mean, seen, fraction, gap_stop):
    """For each ray, the step where brightness first falls below `fraction` of that ray's own tissue.

    `gap_stop` bounds the search at both ends: the reference is taken from tissue the ray actually has,
    and nothing past the absolute stop is examined, because outside the section the brightness is 0 and
    every fraction would trigger there. That would make the cue look like it fires everywhere while
    really only re-finding the tissue edge the gap rule already handles.
    """
    rays, width = mean.shape

    if not fraction:
        return np.full(rays, width, np.int64)

    # Unseen cells are excluded by weighting rather than left as zeros: a zero inside the smoothing
    # window is indistinguishable from genuinely dark tissue, so it would invent borders at the window
    # edges where coverage thins out.
    size = (max(int(round(DIP_SMOOTH_ALONG_UM / ALONG_BIN_UM)), 1),
            max(int(round(DIP_SMOOTH_ACROSS_UM / ACROSS_BIN_UM)), 1))
    weight = seen.astype(np.float32)
    profile = (uniform_filter(np.where(seen, mean, 0.0).astype(np.float32), size=size)
               / np.maximum(uniform_filter(weight, size=size), 1e-6))

    base = max(int(round(DIP_BASE_UM / ACROSS_BIN_UM)), 2)
    needed = max(int(round(DIP_WIDTH_UM / ACROSS_BIN_UM)), 1)
    stop = np.full(rays, width, np.int64)

    for index in range(rays):
        limit = int(gap_stop[index])
        end = min(base, limit)

        # A ray with less tissue than the reference window is left to the gap rule. Measuring a
        # reference off two or three cells would make the fraction meaningless, and these are the
        # short rays at the ends of the window where the gap rule is already right.
        if end < 2 or limit <= base:
            continue

        reference = float(np.median(profile[index, :end]))

        if reference <= 0.0:
            continue

        below = profile[index, base:limit] < fraction * reference

        # Run boundaries from a padded difference, the same construction as first_wide_gap. A run that
        # reaches the end of the searched stretch is NOT treated as unbounded here: unlike the absolute
        # cue, the end of this stretch is the gap stop rather than the outside of the section, and the
        # gap rule already owns that point.
        padded = np.concatenate([[False], below, [False]])
        change = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(change == 1)
        lengths = np.flatnonzero(change == -1) - starts
        qualifying = np.flatnonzero(lengths >= needed)

        if len(qualifying):
            stop[index] = base + int(starts[qualifying[0]])

    return stop


# --- the dark-bright pair, for the rays the gap rule leaves blind ---

# How far either side of a step the crest is looked for, in um.
PAIR_REACH_UM = 48.0

# Smoothed ALONG the band only. dip_stops also smooths 56 um ACROSS, which would erase a 48 um pair outright:
# the two cues want opposite treatment of the same axis. Short, because the lamina's offset from her clicks
# drifts along the boundary and a long average is what smeared the first measurement of this cue to nothing.
PAIR_SMOOTH_ALONG_UM = 48.0

# No pair is looked for closer to the Field L line than this, in um. Her NCM is never this shallow.
PAIR_START_UM = 400.0


def pair_score(mean, seen, reach_um=PAIR_REACH_UM, smooth_along_um=PAIR_SMOOTH_ALONG_UM):
    """Per (ray, step): how much brighter the neighbourhood's crest is, where this step is its own trough.

    HER BORDER IS A DARK-BRIGHT PAIR, not a dip. A thin dark lamina runs along it with a bright ridge
    immediately beside it, and every statistic that pools both sides cancels one against the other -- the
    32 um band median called OR408's boundary BRIGHTER than the tissue either side (+0.041) while the line is
    plain to the eye ([[hp-border-is-a-dark-bright-pair]]). Asking for the pair instead of the darkness is what
    makes it visible: above chance on all 8 sections that carry a measurable HP/NCM boundary.

    THE FRAME IS FREE. In the band frame `across` is the across-the-border direction already, so `ray_table`'s
    `mean` IS a cross-border profile per ray. No resampling and no normals; a 1-D filter along each row does it.

    Zero where the step is not its neighbourhood's trough, so a step either carries the pair or scores nothing.
    """
    size = (max(int(round(smooth_along_um / ALONG_BIN_UM)), 1), 1)
    weight = seen.astype(np.float32)

    # Weighted, so an unseen cell does not read as genuinely dark tissue and invent a trough where the
    # coverage thins out. Same guard as dip_stops.
    profile = (uniform_filter(np.where(seen, mean, 0.0).astype(np.float32), size=size)
               / np.maximum(uniform_filter(weight, size=size), 1e-6))

    reach = max(int(round(reach_um / ACROSS_BIN_UM)), 1)
    crest = maximum_filter1d(profile, size=2 * reach + 1, axis=1, mode="nearest")
    trough = minimum_filter1d(profile, size=2 * reach + 1, axis=1, mode="nearest")

    return np.where(profile <= trough, crest - profile, 0.0)


def pair_stops(mean, seen, limit, reach_um=PAIR_REACH_UM, start_um=PAIR_START_UM):
    """Per ray: the step carrying the STRONGEST pair, searched from `start_um` out to that ray's own limit.

    STRONGEST, NOT FIRST, and this is the whole design. A test calibrated so that a tenth of interior steps
    pass it fires about ten steps past wherever the search begins, cutting every ray short -- the failure
    written up at DIP_WIDTH_UM (IoU 0.759 -> 0.652 for one point of precision). This is only ever asked about a
    ray with NO wall of its own, so there is no competing evidence to be overridden and the best candidate on
    the ray is the honest answer. Same shape as the path decode that beat thresholding on the Field L border
    ([[path-decode-lands-within-112um]]).
    """
    rays, width = mean.shape
    score = pair_score(mean, seen, reach_um)
    steps = np.arange(width)[None, :]
    first = max(int(round(start_um / ACROSS_BIN_UM)), 1)

    usable = seen & (steps >= first) & (steps < np.asarray(limit)[:, None])
    out = np.full(rays, width, np.int64)
    any_usable = usable.any(axis=1)
    out[any_usable] = np.argmax(np.where(usable, score, -1.0), axis=1)[any_usable]

    return out


# --- the bright ridge, the OTHER kind of separation she draws on ---

# Half the width of the strip itself, and where its flanks are measured, in um. Read off the profile she
# circled and not fitted to IoU: the ridge there is 0.23 over 0.16 across roughly 60 um, so a 48 um strip
# against flanks starting 48 um out and 96 um long covers it with nothing tuned.
RIDGE_HALF_UM, RIDGE_SKIP_UM, RIDGE_FLANK_UM = 24.0, 48.0, 96.0

# How far ALONG the band the score is averaged, in um. OFF, against expectation, and the expectation is worth
# recording because it was wrong. `grids/ridge OR408...png` shows the unsmoothed cue firing as scattered specks
# all over the interior of her region -- a PS6 cell IS a bright blob 20-30 um across, so this looked like the
# same failure as the dark cue firing on the gaps between cells ([[pair-cue-fires-on-cell-gaps]]), and averaging
# along the band is what tells a dot from a line. It does clean up the picture, and it makes the answer WORSE
# where the cue is actually used: on OR408's 20 blind rays 16 um becomes 32, and the whole-ray decode is
# unmoved at 216 um. So the specks are not what beats the argmax; some other long bright structure on the ray is.
RIDGE_SMOOTH_ALONG_UM = 0.0

# No ridge is looked for shallower than this, in um -- the same guard as the pair cue, for the same reason.
RIDGE_START_UM = 300.0


def filled_mean(mean, seen):
    """`mean` as float32, with the cells no pixel landed in held at their own ray's average.

    Every cue across the band needs this and needs it the same way: a thinning ray must not read as dark tissue,
    or the end of the section manufactures a step out of the background beyond it. A whole ray with no coverage
    at all falls back to zero.
    """
    # Summed by hand rather than with `nanmean`, which warns on a ray that has no covered cell at all -- and
    # confining a cue to the section's interior makes such a ray ordinary rather than exceptional.
    covered = seen.sum(axis=1, keepdims=True)
    per_ray = np.where(seen, mean, 0.0).sum(axis=1, keepdims=True) / np.maximum(covered, 1)

    return np.where(seen, mean, np.where(covered > 0, per_ray, 0.0)).astype(np.float32)


def ridge_score(mean, seen, half_um=RIDGE_HALF_UM, skip_um=RIDGE_SKIP_UM, flank_um=RIDGE_FLANK_UM,
                smooth_along_um=RIDGE_SMOOTH_ALONG_UM, deeper=np.maximum):
    """Per (ray, step): how much brighter a thin strip is than the BRIGHTER of its two flanks.

    ALICE DREW THE CIRCLE THAT FOUND THIS, 2026-08-17: *"do you want me to circle the separation so you can see
    it for yourself? it is very obvious"*. Inside the band she circled on OR408 there is no black on her border
    at all -- 0.0% of steps below 0.02 -- and instead a 60 um strip at 0.23 against 0.16 on BOTH sides. Her own
    rule from before: *"its usually either a black gap or a visible line separation"*. `long_strips` implements
    the first kind, and this is the second.

    A RIDGE, NOT A STEP, and the distinction is what keeps depth out of it. NCM is brighter than the tissue
    beyond it on most sections, but so is anything nearer the middle, so a step test rediscovers depth for the
    third time ([[ncm-is-brighter-than-the-tissue-beyond]], [[depth-alone-beats-the-border-detector]]).
    Subtracting the BRIGHTER flank means a monotone gradient scores at or below zero however steep it is: the
    same construction that made her end cuts findable as dark local minima ([[her-ends-are-dark-local-minima]]).

    It is also SPECIFIC to the sections that need it, which is the rarer property. Her border sits at the 90th
    percentile of this score on OR408 and the 76th on Gre595, and at the 8th-32nd on the twelve black-gap
    sections, where it scores negative -- her border there is a dim place, as a gap should be
    (`diag_bright_ridge.py`).

    Unseen cells are held at their ray's own mean rather than at zero, so the end of the tissue cannot
    manufacture a ridge out of the background beyond it.

    `deeper` picks which flank the strip is compared against, and `valley_score` passes `np.minimum` to get the
    dark-line mirror of this. It is a parameter rather than a second copy of the body because a duplicate DID
    drift today: `diag_bright_ridge.py` carried its own copy and silently ignored a change made here.
    """
    filled = filled_mean(mean, seen)

    # ALONG the band, before anything is compared across it: a cell is a dot and a separation is a line. Off by
    # default -- see RIDGE_SMOOTH_ALONG_UM, where it is measured and loses. Weighted by coverage for the same
    # reason the unseen cells were filled above: a thinning ray must not read as dark tissue.
    if smooth_along_um > 0.0:
        size = max(int(round(smooth_along_um / ALONG_BIN_UM)), 1)
        weight = uniform_filter1d(seen.astype(np.float32), size=size, axis=0, mode="nearest")
        filled = (uniform_filter1d(np.where(seen, filled, 0.0), size=size, axis=0, mode="nearest")
                  / np.maximum(weight, 1e-6))

    def smooth(width_um):
        size = max(int(round(width_um / ACROSS_BIN_UM)), 1)

        return uniform_filter1d(filled, size=size, axis=1, mode="nearest")

    middle = smooth(2.0 * half_um)
    flank = smooth(flank_um)
    shift = max(int(round((skip_um + flank_um / 2.0) / ACROSS_BIN_UM)), 1)

    # `nearest`-style padding by hand: rolling wraps, and a wrapped flank would compare the two ENDS of a ray.
    inner, outer = np.roll(flank, shift, axis=1), np.roll(flank, -shift, axis=1)
    inner[:, :shift] = flank[:, :1]
    outer[:, -shift:] = flank[:, -1:]

    return middle - deeper(inner, outer)


# Alice, 2026-08-17, correcting the section above: *"the thing im trying to get you to identify isn't a bright
# line its the dark line that my annotations outline in the area i circled"*, *"its not dotted though its
# continuous like its very clear"*. `look_circle.py` draws her circled area magnified and she is plainly right --
# a thin dark ribbon hugs her outline for its whole length, with the brighter rim band above it. The bright
# "ridge" the block above measures is that rim, found by averaging her rays and washing the dark line out.
#
# THICKNESS from that picture, not fitted: her ribbon is two to four canvas pixels across, so a 32 um strip.
# MEASURED AND DEAD, both of them, kept because the numbers are the argument (`diag_dip_path.py --cue valley`,
# `--cue oriented`). Her border ranks at the 9th-43rd percentile of either score along its own ray -- BELOW
# chance on all fifteen sections -- while RAW DARKNESS ranks it 72nd-100th on fourteen of them. The mechanism is
# the property two docstrings down: the middle of a wide gap has dark flanks too and scores ~0, whereas the
# 16 um gaps between bright PS6 cells are perfect valleys. Nothing calls either function in the shipped rule.
#
# AND THE HARDER FACT the same run turned up, which no cue of this shape can get around: on OR408 her border is
# at the 20th percentile of raw darkness on its own ray. Her line is dark, and other things on the same ray are
# darker. So "the darkest place" cannot be the decision rule there, however it is measured -- four cues have now
# failed that way ([[lamina-cue-failed-three-ways]], [[pair-cue-fires-on-cell-gaps]], and these two).
VALLEY_HALF_UM, VALLEY_SKIP_UM, VALLEY_FLANK_UM = 16.0, 24.0, 48.0
# ON by default here, unlike the ridge's: her line is thin ACROSS the band and continuous ALONG it, while the
# speckle that defeated every threshold is small and round (`look_circle.py`, `diag_circle_dip.py`: the pieces
# near her border are 16-48 um and hundreds of others are as long). Anisotropy is the one thing that separates
# them, and it is the reason this cue is worth trying at all.
VALLEY_SMOOTH_ALONG_UM = 240.0
# The ORIENTATION test, and the reason the cue above is not enough on its own: painted on her circled crop it
# hugs her line for a third of its length and loses the rest to DIAGONAL dark streaks deeper in the tissue
# (`grids/look valley OR408...png`, and [[lamina-cue-failed-three-ways]] is the same structure defeating a
# different cue). Her line runs ALONG the band; a streak crosses it. One width, used both ways, so there is
# no second knob to overfit ([[knob-count-is-a-modelling-choice]]).
VALLEY_LINE_UM = 240.0


def band_smooth(score, seen, width_um, axis):
    """Average a score along the band (`axis=0`) or across it (`axis=1`), weighted by coverage.

    Weighted because the cells no pixel landed in must not be averaged in as zeros -- a thinning ray would
    then read as bright tissue and manufacture a valley beside itself.
    """
    if width_um <= 0.0:
        return score

    size = max(int(round(width_um / (ALONG_BIN_UM if axis == 0 else ACROSS_BIN_UM))), 1)
    weight = uniform_filter1d(seen.astype(np.float32), size=size, axis=axis, mode="nearest")
    total = uniform_filter1d(np.where(seen, score, 0.0).astype(np.float32), size=size, axis=axis,
                             mode="nearest")

    return total / np.maximum(weight, 1e-6)


def valley_score(mean, seen, half_um=VALLEY_HALF_UM, skip_um=VALLEY_SKIP_UM, flank_um=VALLEY_FLANK_UM,
                 smooth_along_um=VALLEY_SMOOTH_ALONG_UM):
    """Per (ray, step): how much darker a thin strip is than the DARKER of its two flanks -- `ridge_score`'s mirror.

    `np.minimum`, not `np.maximum`, and that single choice is the whole cue: a cell only scores when the tissue
    is brighter on BOTH sides of it, so depth cannot fake it -- a border that merely gets darker outwards scores
    at or below zero however steep it is ([[depth-alone-beats-the-border-detector]] is the failure this avoids,
    for the fourth time). The same construction found her end cuts ([[her-ends-are-dark-local-minima]]).

    Why not the cues already shipped. The absolute one, green < 0.02, marks 0.0% of her border on OR408 -- her
    line is dark against BRIGHT tissue, 0.10 against 0.22. The relative one, green < 0.8 x its own 400 um box
    mean, marks dim speckle all over the section and not her line at all (`grids/look mask ...png`), and the 80
    um agreement I reported from it was the tolerance finding noise, not the cue finding her border. A 400 um
    box is the wrong scale for a 32 um line; its two flanks are the right one.
    """
    return -ridge_score(mean, seen, half_um=half_um, skip_um=skip_um, flank_um=flank_um,
                        smooth_along_um=smooth_along_um, deeper=np.minimum)


def oriented_valley(mean, seen, half_um=VALLEY_HALF_UM, skip_um=VALLEY_SKIP_UM, flank_um=VALLEY_FLANK_UM,
                    line_um=VALLEY_LINE_UM):
    """A valley that CONTINUES ALONG THE BAND, minus one that crosses it.

    The band's own coordinates do the work, so no orientation has to be estimated: in the ray table axis 0 is
    along the band and axis 1 is depth. Averaging the valley score along axis 0 rewards a dark structure that
    keeps going in her border's direction; averaging it along axis 1 rewards one that reaches inwards or
    outwards instead. Her line is the first kind and the tissue's diagonal streaks are the second, so the
    difference separates them where neither the score nor any threshold on it could.

    A cell in the middle of a wide dark AREA scores about zero here, which I wrote down as correct and which
    turned out to be fatal -- see the measurement below.
    """
    raw = valley_score(mean, seen, half_um=half_um, skip_um=skip_um, flank_um=flank_um, smooth_along_um=0.0)

    return band_smooth(raw, seen, line_um, axis=0) - band_smooth(raw, seen, line_um, axis=1)


# The RIM, not the line -- the one thing in `grids/look raw OR408...png` that no cue has used yet. Her dark
# ribbon is 32 um across and four cues have now failed to pick it out of the speckle and the diagonal streaks,
# the last two below chance ([[her-line-is-not-the-darkest-thing]]). Directly OUTSIDE it, though, is a wide
# clearly brighter band, and her outline runs along that band's inner edge for its whole length. A 100-200 um
# structure is a far easier thing to locate than a 32 um one, and it is the reason the averaged profile in
# `read_circle.py` showed a bump where the line is (0.228 against 0.16) -- that bump was real, it just was not
# the line ([[or408s-separation-is-a-bright-line]] is the write-up of getting this backwards).
#
# So the question changes from "is this cell dark?" to "does the tissue get brighter as I move OUTWARD from
# here?", which is a question about two neighbourhoods rather than one place.
#
# MEASURED AND DEAD, in BOTH directions, and nothing calls it (`diag_dip_path.py --cue rim`, `--cue drop`). Her
# border sits at the 52nd percentile of the outward step along its own ray and the 50th of its reverse -- chance,
# to within the noise, on fifteen sections. The rim is real and it is not where her line is; the fifth cue to
# fail here ([[her-line-is-not-the-darkest-thing]] has the other four).
#
# THE NEAR MISS IS THE PART TO REMEMBER. Before `inside` was passed the same run put her border at the 18th
# percentile, which flipped is 82nd and would have read as the best far-border cue ever measured here. It was
# the 160 um outer window hanging off the section's SURFACE and averaging in the background: the cue said "you
# are one window from the edge of the brain", which is depth for the fifth time and correlates with her border
# only because her border is a far border. A wide window needs a depth limit for the same reason a search does
# ([[depth-alone-beats-the-border-detector]]), and a suspiciously good number in the direction you did not
# predict is the symptom.
RIM_SPAN_UM = 160.0
# The gap left between the two neighbourhoods, in um: her ribbon is two to four canvas pixels across, and it
# belongs to NEITHER side. Without this the dark line itself drags the inner window down and the score stops
# measuring the rim and starts measuring the line again.
RIM_SKIP_UM = 32.0
# THE GUARD, and the only reason this is not depth for the fifth time. Brightness rises with depth on its own
# ([[ncm-is-brighter-than-the-tissue-beyond]], [[hp-is-brighter-at-equal-depth]], and
# [[depth-alone-beats-the-border-detector]] is what happens when that leaks in), so a bare outward step scores
# positive nearly everywhere and its argmax is arbitrary. Subtracting the same score averaged over 800 um
# ACROSS the band cancels any smooth ramp exactly, however steep, and leaves only a step that is sharp
# relative to its own surroundings. `ridge_score` gets the same protection from comparing against the BRIGHTER
# flank; this is that idea at the rim's scale instead of the line's.
RIM_TREND_UM = 800.0
# Off by default, as it is for the ridge, and to be argued with numbers rather than expectation.
RIM_SMOOTH_ALONG_UM = 0.0


def rim_edge(mean, seen, span_um=RIM_SPAN_UM, skip_um=RIM_SKIP_UM, trend_um=RIM_TREND_UM,
             smooth_along_um=RIM_SMOOTH_ALONG_UM, inside=None):
    """Per (ray, step): how much brighter the tissue is just OUTSIDE this cell than just inside, de-trended.

    Positive where brightness steps up going outward -- the inner edge of the bright rim band. Signed on
    purpose: the mirror image, a step DOWN going outward, is a different structure and must not score.

    `inside` is the mask of cells that are actually within the section, and passing it is not optional in
    practice. The 160 um outer window is wide enough to hang off the far SURFACE, where green is true
    background, and her border often sits a couple of hundred um inside that surface -- so without it the
    strongest step of all is "you are one window from the edge of the brain", which is depth leaking in for the
    fifth time ([[depth-alone-beats-the-border-detector]]) wearing a step's clothes. Cells outside it are held
    at their ray's interior average, exactly as uncovered cells are.
    """
    keep = seen if inside is None else (seen & inside)
    filled = band_smooth(filled_mean(mean, keep), keep, smooth_along_um, axis=0)
    size = max(int(round(span_um / ACROSS_BIN_UM)), 1)
    box = uniform_filter1d(filled, size=size, axis=1, mode="nearest")

    # `nearest` padding by hand, as in `ridge_score`: rolling would wrap a ray's outer end onto its inner end.
    steps = np.arange(box.shape[1])
    shift = max(int(round((skip_um + span_um / 2.0) / ACROSS_BIN_UM)), 1)
    step = box[:, np.clip(steps + shift, 0, box.shape[1] - 1)] - box[:, np.clip(steps - shift, 0, box.shape[1] - 1)]

    return step - band_smooth(step, keep, trend_um, axis=1)


def ridge_stops(mean, seen, limit, start_um=RIDGE_START_UM):
    """Per ray: the step carrying the STRONGEST bright ridge, from `start_um` out to that ray's own limit.

    Strongest and not first, for the reason spelled out at `pair_stops`: a threshold fires just past wherever
    the search starts. Only ever asked about a ray that found NO wall, where the alternative is copying a
    neighbour -- on OR408's 20 such rays this lands a median 16 um from her border against the rule's 64
    (`diag_bright_ridge.py --rays gapless`). On the 54 rays whose wall came from the LOWERED floor it lands 160
    um out against that wall's 64, so it must never overrule a wall, however weak.
    """
    rays, width = mean.shape
    score = ridge_score(mean, seen)
    steps = np.arange(width)[None, :]
    first = max(int(round(start_um / ACROSS_BIN_UM)), 1)

    usable = seen & (steps >= first) & (steps < np.asarray(limit)[:, None])
    out = np.full(rays, width, np.int64)
    any_usable = usable.any(axis=1)
    out[any_usable] = np.argmax(np.where(usable, score, -np.inf), axis=1)[any_usable]

    return out


def bridge_stops(stop, width):
    """A ray that never found a gap takes the border its neighbours found, instead of the image's edge.

    Alice, 2026-08-15: *"i think most of the borders are pretty straightforward because of the black gap?
    this goes back to what i was saying about taking field l endpoint and then tracing"* -- and the trace
    (trace_far_border.py) says she is right: 71% of rays end where the tissue ends, and on every ray that
    finds a gap the rule's border is already a median 8-48 um from hers, inside her own drawing precision.
    The whole far-border error is the OTHER rays: 0% of rays in six sections but 34% in OR408, 31% in
    LBlu59_RH, 24% in OR99_RH, and they overrun by a median 632-2308 um because `first_wide_gap` returns
    the end of the ray when it finds nothing, i.e. the edge of the image.

    Interpolating between the flanking rays that DID find one is the cheapest form of her "trace it":
    a ray with no evidence of its own is not allowed to invent an answer, it continues the curve. np.interp
    extends with the nearest value at the two ends, which is the right behaviour there too -- a run of
    unresolved rays at the end of the window continues its last resolved neighbour rather than escaping.

    NO CONSTANT AT ALL, which is why it is worth trying before anything fitted.
    """
    found = stop < width

    if not found.any() or found.all():
        return stop

    index = np.arange(len(stop))

    return np.rint(np.interp(index, index[found], stop[found].astype(float))).astype(np.int64)


def clamp_stops(stop, max_step_um):
    """Forbid the border from moving faster between neighbouring rays than she ever draws it.

    Her own outlines move a median 8 um per ray, p99 80 um, max 192 um (check_border_smooth.py), and the
    leave-one-bird-out p99 is 72-81 um -- flat across folds, so 80 um is a property of her drawing and not
    of a fold. The rule's own p99 is 608 um and its max 2032 um.

    Two sweeps, forward then backward, each taking a MINIMUM: the border may only ever be pulled IN. That
    asymmetry is deliberate. The failure is rays running too far out, and a clamp that could also push a
    ray outward would grow the region wherever a neighbour overran -- turning one bad ray into a bulge
    instead of removing it. One pass per direction is enough for a min-clamp to reach its fixed point.
    """
    if max_step_um <= 0.0:
        return stop

    limit = max(int(round(max_step_um / ACROSS_BIN_UM)), 1)
    out = stop.astype(np.int64).copy()

    for index in range(1, len(out)):
        out[index] = min(out[index], out[index - 1] + limit)

    for index in range(len(out) - 2, -1, -1):
        out[index] = min(out[index], out[index + 1] + limit)

    return out


def lift_stops(stop, max_step_um, width):
    """The MIRROR of clamp_stops: forbid the border from being pulled IN faster than she draws it.

    Alice, 2026-08-15: *"are there measures to ensure that the shape is regular because the ups and downs
    are not normal"*. `clamp_stops` only ever takes a minimum, so it deletes a ray that ran too FAR and
    spreads a ray that stopped too EARLY -- and an early stop is exactly what speckle produces. Gre595's
    teeth are notches, not spikes, and clamping made that section worse (0.667 -> 0.648) while capping its
    p99 step at 192 um, so the regularity it bought was the wrong regularity.

    Two sweeps taking a MAXIMUM, so a notch is lifted back out to meet its neighbours. Same constant, read
    off her own outlines (max step 192 um, p99 80) and not fitted to IoU.

    THE RISK, stated because it decides whether this can ship: a ray that stopped early on a REAL gap gets
    lifted over it, growing the region past a border she drew. That is the same trade `bridge` makes and
    the reason both are measured rather than argued about. Rays that never stopped at all (stop == width)
    are excluded from lifting their neighbours, because they carry no evidence.
    """
    if max_step_um <= 0.0:
        return stop

    limit = max(int(round(max_step_um / ACROSS_BIN_UM)), 1)
    out = stop.astype(np.int64).copy()
    ran_off = out >= width

    for index in range(1, len(out)):
        if not ran_off[index - 1]:
            out[index] = max(out[index], out[index - 1] - limit)

    for index in range(len(out) - 2, -1, -1):
        if not ran_off[index + 1]:
            out[index] = max(out[index], out[index + 1] - limit)

    return np.minimum(out, width)


def dome_stops(stop, width, use_dome, cap_um=0.0, balance=False):
    """Force the border to be ONE OUTWARD DOME: replace the stops with their least concave majorant.

    Alice, 2026-08-15: *"i dont know if you can ensure that its all a general concave shape?"*

    `clamp_stops`, `lift_stops` and `smooth_stops` are all LOCAL -- they limit what happens between two
    neighbouring rays, so a border can still drift 500 um inward over 30 rays and back out again without
    ever breaking the limit. Concavity is the GLOBAL version of the same request, and it has no knob at
    all: the least concave majorant is the smallest curve that is concave everywhere and never dips below
    the stops, i.e. the upper boundary of the convex hull of the points (ray, stop).

    WHAT IT COSTS ON HER OWN BORDERS, measured before writing anything that uses it (oracle_concave.py):
    doming her 14 outlines adds 1.2% of region, worst single ray 50 um, and leaves a ceiling of 0.989 IoU
    -- far above the 0.799 the rule actually reaches, so the constraint is not what limits the answer. A
    parabola through the same stops is far too stiff (0.951, off by 201 um at worst), which is why this is
    a hull and not a fit.

    `cap_um` LIMITS HOW FAR THE DOME MAY LIFT ONE RAY, and it exists because the unlimited version loses:
    0.778 against 0.797, precision 0.861 -> 0.827 with recall 0.921 -> 0.932. The reason is not the shape
    -- her own borders dome for 1.2% -- it is that the RULE has outward spikes she does not, and every
    spike becomes a hull vertex that drags its whole neighbourhood out with it. Clamping the spikes off
    first does not rescue it either (0.780 at clamp 80, 0.777 at 192). A cap keeps the part of the idea
    that is safe: 50 um is the worst single ray doming HER borders moves, so a dome allowed 50 um can
    remove a dent she would never draw and cannot bridge an inlet she did.

    `balance` PULLS THE WHOLE DOME BACK IN by the average depth it added, and it is the answer to the
    objection above rather than another knob: subtracting a constant from a concave curve leaves it
    concave, so the shape constraint is kept in full while the region stops growing. Every uncapped or
    capped dome loses ONLY through precision (0.861 -> 0.827 at no cap, -> 0.841 at 50 um) because it is
    an outward-only operation and the region is already too big.

    Rays that never stopped (stop >= width) are excluded from the hull. They ran off the end of the table
    and carry no evidence, and as hull vertices they would drag the whole dome out to the table's edge.
    """
    if not use_dome:
        return stop

    found = np.flatnonzero(stop < width)

    if len(found) < 3:
        return stop

    hull = []

    for index in found:
        value = float(stop[index])

        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]

            # A non-negative cross product means the middle point sits at or below the chord, i.e. a dent.
            if (x1 - x0) * (value - y0) - (index - x0) * (y1 - y0) >= 0:
                hull.pop()
            else:
                break

        hull.append((int(index), value))

    xs = np.array([point[0] for point in hull], float)
    ys = np.array([point[1] for point in hull], float)
    lifted = np.interp(np.arange(len(stop)), xs, ys)

    # Only ever outward, and never past the end of the table. Rays outside the hull's span keep their own
    # stop, because np.interp holds the end value flat there rather than extrapolating a dome.
    target = np.rint(lifted).astype(np.int64)

    if cap_um > 0.0:
        target = np.minimum(target, stop + max(int(round(cap_um / ACROSS_BIN_UM)), 1))

    target = np.maximum(stop, target)

    if balance:
        inside = (np.arange(len(stop)) >= xs[0]) & (np.arange(len(stop)) <= xs[-1])
        target = target - int(round(float((target - stop)[inside].mean())))

    return np.minimum(np.maximum(target, 0), width)


def long_strips(bright, thickness_um, min_length_um, require_edge=False, wide_too=False,
                min_aspect=0.0, rejoin=False):
    """Black too THIN to survive the width test, but belonging to a piece longer than min_length_um.

    Alice, 2026-08-15: *"do you understand what i mean when i talk about the black gap separating hp and
    ncm?"* and *"is there a way for you to recognize that that magenta stuff isnt the border but just
    spottiness in the tissue?"*. The answer to both is one mask.

    `gap_mask`'s width test exists because a vessel or a fold would otherwise stop rays at random, and it
    works -- but the ventricle between hippocampus and NCM is a median 40 um across, so it fails the test
    too and the ray walks straight through it into hippocampus (`check_black_gap.py`: 64-75% of her
    interior border has raw black on it, ~80% of it thinner than 80 um).

    Length is what tells the two apart, and it is nearly perfect at it (`check_strip_shape.py`): of 563
    thin-black pieces sitting INSIDE her region, median length 16 um and p90 40 um, exactly TWO reach
    200 um -- while the pieces on her far border reach 1840 um. Spottiness is short by nature; a
    ventricle or a cleft is millimetres long.

    Two details that are not free choices:

      the rind    a piece is only measured after black within half a disc of a WIDE gap is removed. An
                  opening shrinks the black and regrows it, so the outer 40 um of the huge black area
                  outside the brain comes back looking thin, which drew a false ribbon around every
                  section. Growing `wide` back by the same radius deletes exactly that artefact and keeps
                  a genuine thin arm reaching further off a wide gap -- a ventricle that opens onto the
                  surface is one, so dropping whole connected components instead would delete the answer.
      the length  the longer side of the piece's bounding box, not a skeleton. It overstates a diagonal
                  piece, but the two populations are three orders of magnitude apart and no refinement
                  moves a piece across the line.
    """
    if min_length_um <= 0.0 and not require_edge:
        return np.zeros(bright.shape, bool)

    radius = max(int(round(thickness_um / ACROSS_BIN_UM / 2.0)), 1)

    # `wide_too` LETS A CANDIDATE BE ANY WIDTH, and it is kept only because the measurement is the
    # evidence: it scores 0.702 against 0.803, halving Gre595 (0.846 -> 0.461) and Wh175_LH_1-2-1
    # (0.704 -> 0.476), because interior black that is genuinely wide is usually a tear she draws over.
    # It also leaves Purp30 EXACTLY unchanged at 0.785, which is what killed the diagnosis it was built
    # for -- her ribbon there is not lost for being too wide, it is lost for being too SHORT.
    if wide_too:
        grown = binary_dilation(~binary_fill_holes(bright), iterations=radius)
    else:
        grown = binary_dilation(gap_mask(bright, thickness_um), iterations=radius)

    thin = (~bright) & ~grown

    tagged, count = label(thin)

    if not count:
        return thin

    # `rejoin` MEASURES A STRIP ACROSS THE HOLE THE RIND REMOVAL PUNCHED IN IT. The order above is delete, then
    # label, then measure, so a separation that runs alongside a wide gap is cut into stubs and each stub is
    # judged alone -- `diag_rind_damage.py` caught it doing exactly that: Wh175_LH_1-1-7 loses a 1112 um strip
    # entirely (100% eaten, 40 um left), YW113_RH_1-1-3 81% of a 528 um one, Wh175_LH_1-2-1 76% of 152 um, and
    # those are three of the four worst-scoring sections.
    #
    # MEASURED AND DEAD, kept because the number is the argument: mean IoU 0.832 -> 0.020, all fifteen sections
    # (`scan_ridge.py --what rejoin`). Grouping stubs by proximity cannot work at any radius, because speckle is
    # DENSE -- a 48 um dilation links the whole speckle field of a section into one piece whose bounding box
    # spans the section and clears any length bar, so every ray stops on the first speck. The same wall that
    # lowering the length floor ran into ([[thin-black-strips-are-her-border]]).
    #
    # Labelling BEFORE the deletion fails for a related reason: the rind is a continuous ribbon around the whole
    # section, so every cleft that opens onto the surface merges into it and inherits a bounding box the size of
    # the brain. Both packagings say the same thing -- a strip cut in two can only be rejoined along its OWN
    # DIRECTION, which is what `extend_strips` already does for a strip that ends at a wide gap
    # ([[follow-the-strip-where-black-closes]]), and that is where the fix belongs.
    if rejoin:
        near, total = label(binary_dilation(thin, iterations=radius + 1))

        if total:
            tagged, count = np.where(thin, near, 0), total

    limit = min_length_um / ACROSS_BIN_UM
    keep = np.zeros(count + 1, bool)

    # `require_edge` is Alice's own rule, 2026-08-15: *"can you solve the 0.08 problem by just saying all
    # the speckled stuff within the tissue border is just noise"*. A ventricle or a cleft OPENS ONTO the
    # brain's surface; speckle floating in the middle of the tissue does not. Note this is the exact
    # inverse of the rind test above -- that one deletes black for being near the outside, this one keeps
    # it for the same reason -- so the two have to be measured together. Distance from `grown` rather than
    # adjacency to `wide`, because the rind subtraction has already cut `radius` steps off the arm's root:
    # a genuine arm therefore starts EXACTLY one step from `grown`, and an interior piece starts further.
    reach = distance_transform_edt(~grown) if require_edge else None

    # `min_aspect` is the SHAPE test. Alice, 2026-08-15, having seen that lowering the length floor to
    # 100 um rescues Purp30's ribbon but turns Gre595's speckle into a comb of teeth: *"like the testing
    # the shape"*. Length alone cannot separate the two there, because in a speckled section her outline
    # runs through the speckle and neighbouring specks merge into clouds with long bounding boxes. A cloud
    # is as wide as it is long; a cleft is not. Aspect is length / MEAN width, i.e. length^2 / area, so it
    # needs no second pass over the image and no new threshold on brightness.
    #
    # Cross-ray smoothing was tried first and does nothing here (mean 0.789 -> 0.785/0.786/0.788 at
    # 100/200/400 um): Gre595's speckle stops MOST rays early, not a few, so the median stop is early too.
    for index, box in enumerate(find_objects(tagged), start=1):
        piece = tagged[box] == index
        length = max(box[0].stop - box[0].start, box[1].stop - box[1].start)
        long_enough = length >= limit
        thin_enough = min_aspect <= 0.0 or length * length >= min_aspect * piece.sum()
        keep[index] = (long_enough and thin_enough
                       and (not require_edge or reach[box][piece].min() <= 1.5))

    return keep[tagged]


def smooth_stops(stop, smooth_um):
    """A running median of the stopping points across neighbouring rays.

    A median and not a mean: a mean is dragged by a single ray that stopped early, which is exactly
    the failure being fixed, while a median needs most of its window to agree before it moves.
    """
    if smooth_um <= 0.0:
        return stop

    size = max(int(round(smooth_um / ALONG_BIN_UM)), 1)

    if size <= 1:
        return stop

    # ODD, always. An even window has no centre ray, so median_filter takes stops[i-size/2 : i+size/2-1] and
    # the whole border slides half a ray along the band. Measured 2026-08-16: every even rung of the ladder
    # (10 and 20 rays) cost ~0.005 IoU on nearly every section at once while the odd rungs either side (5 and
    # 15) cost nothing, which is a shift and not a smoothing effect.
    size += 1 - size % 2

    # THE WINDOW SHRINKS AT THE ENDS instead of being padded. Alice, 2026-08-16: *"i thought you had
    # something in place to prevent spikes"* -- it was in place and it was a no-op exactly where her spikes
    # are. `median_filter(mode="nearest")` fills the missing half of ray 0's window by REPEATING ray 0, so
    # with size 35 that ray casts 18 of 35 votes and can never be outvoted; 'reflect'/'mirror' only change
    # which of its own neighbours get counted twice. A spike at the END of the band was therefore the one
    # spike the median could not touch, and both of the corners she has pointed at are end spikes.
    # Taking the median of the rays that actually exist needs no padding rule at all.
    half = size // 2
    smoothed = [np.median(stop[max(ray - half, 0):ray + half + 1]) for ray in range(len(stop))]

    return np.rint(smoothed).astype(stop.dtype)


def extend_strips(stop, on_strip, may_change=None, min_rays=3, order=1):
    """Continue a strip's own slope along the band until it meets the wall the rays already found.

    Alice, 2026-08-16, on YW113 1-1-7: *"in the magenta 0.02 you can see that it identifies the separation
    between the hippocampus and ncm but stops midway because the separation is smaller, so i think you just
    ignore it entirely. however, i was wondering if you could follow that line and for the stuff in between
    just fill it out following that concave shape and connect to the outer tissue rather than just draw a
    straight line"*.

    WHAT THE MEASUREMENT SAYS, and it is why no threshold can fix this. On that section the separation's black
    exists on rays 115-124 and NOWHERE ELSE: no black at all within 40 um of her border on rays 92-114, where
    the green under her line reads 0.19-0.44. The rays there sail through the closed separation and stop at the
    brain surface, 184-472 um past her. So the cue is not too strict -- the evidence has run out, and the only
    thing left to do is continue the border that the evidence DID establish.

    THE SHAPE IS MEASURED, NOT FITTED. A run of rays that stopped on a long thin strip is a piece of her border
    with a direction, and least-squares over the run gives it. The continuation walks ray by ray from the run's
    end along that fit. Nothing here is tuned to IoU.

    `order` is how much of the strip's shape is carried out into the closed stretch, and it is the difference
    between a corner and a curve. Alice, 2026-08-16: *"its good, can you make it a little rounder though"*. A
    LINE leaves the continuation at the strip's own end slope, which is the steepest part of her border -- on
    1-1-7 the strip falls 44 um per ray while her border in the middle of the closed stretch falls 16-24 -- so
    a straight continuation runs 150-200 um outside her and meets the tissue edge early, leaving a corner. A
    PARABOLA carries the run's curvature too, and because the strip steepens toward the end of the band, going
    backwards it eases off exactly as she does:

      ray            114   110   105   100    95        her border   408  560  672  776  880
      line           436   610   829     -     -
      parabola       427   557   692   797   872

    A parabola TURNS OVER, and past its vertex it would curve back in and eat the region. So the walk also stops
    where the fit stops moving outward. That is not another constant -- it is the same "stop when you reach the
    border you already had" test applied to the curve's own shape.

    AND THE CURVATURE IS ONLY USABLE WHEN IT BENDS THE RIGHT WAY, which the fit itself says. Walking away from
    a run, the parabola either eases off -- its turning point lies AHEAD, it flattens, it is the rounder border
    -- or it steepens, its turning point lies behind and it accelerates with nothing to stop it before the wall.
    Measured over the 8 runs of 8+ rays the rule finds (diag_extend_shape.py):

      YW113 1-1-7    back  +30 um/ray  eases off, turn  +11 rays ahead      0.882 -> 0.884
      Wh175 1-1-9    back  -21 um/ray  steepens,  turn   -5 rays behind     0.923 -> 0.886
      LBlu59 3-1-8   back  -49 um/ray  steepens,  turn  -35 rays behind     0.836 -> 0.759
      Purp30 1-1-8   back  -21 um/ray  steepens,  turn   -7 rays behind     0.893 -> 0.845

    An accelerating parabola walks INWARD through her region and there is no vertex to stop it, which is where
    the ungated version loses 2 points of recall. So each end of each run decides for itself: carry the curve
    where it eases off, walk the straight line where it does not. The sign of the bend is the whole test.

    STOPPING AT THE TURNING POINT IS RIGHT AND LEVELLING OUT IS NOT -- measured, do not retry. Past the vertex
    the fit still has a depth, and carrying that depth on flat looked like the shape of a border running along
    the tissue edge. But this walk is bounded only by the wall, so a plateau runs to the END OF THE BAND and
    cuts the far half of the region off: 1-1-7 0.884 -> 0.537, Purp30 0.894 -> 0.702, recall 0.925 -> 0.857.
    Any version of it needs a maximum extension length, which is a constant nothing measures.

    WHAT THE CURVE IS AND IS NOT WORTH. Over the 14 sections it is a wash against the straight line -- both
    mean 0.827, honest 0.827 -- winning 1-1-7 (+0.002), Purp30 (+0.001) and 1-1-2 (+0.001), losing OR99_RH
    (-0.008), identical on the other ten. It is kept because it is the shape she asked for and it costs
    nothing, NOT because it scores better. The middle of 1-1-7's closed stretch (rays 92-104) is still hers
    alone: the vertex arrives at ray 104 and there is no black anywhere out there, so no continuation of a
    black strip can reach it.

    TWO GUARDS, both structural:

      it takes the MINIMUM, so this can only ever shrink the region. A continuation that would run outside
      the wall the rays already found is not applied at all.
      it STOPS at the first ray where the continuation reaches that wall -- which is exactly her "connect to
      the outer tissue rather than just draw a straight line". Past that ray the tissue edge is the border
      again and the strip has nothing more to say.

    Placed AFTER the median. The median is 35 rays wide, so on 1-1-7 it would average a 12-ray continuation
    against 17 rays of brain surface and vote the continuation straight back out; and it has to stay before
    the mirror, because the whole point is to give the mirror something continuous to be consistent with
    ([[an-angle-floor-frees-the-mirror]] -- fix the input, do not weaken the safeguard).

    A run shorter than `min_rays` is skipped: two rays define a slope but not a direction worth trusting, and
    speckle that survives the length floor is short by nature.

    `may_change` is which rays the continuation is ALLOWED to overwrite, and it is the difference between a
    win and a loss. Unrestricted, this scored 0.816 against 0.823: it fixed 1-1-7 by 0.039 and Purp30 by
    0.005 but cut 0.058 out of YW113 1-1-10 and 0.086 out of OR99_RH, because a continuation was overwriting
    rays that had stopped on a strip of their OWN -- real evidence, replaced by an extrapolation from
    somewhere else. Alice's sentence is narrower than what I first built: *"connect to the OUTER TISSUE"*. So
    the continuation may only replace a ray that stopped on a wide gap or found no wall at all.
    """
    if not on_strip.any():
        return stop

    out = stop.astype(np.int64).copy()

    padded = np.concatenate([[False], on_strip, [False]]).astype(np.int8)
    starts = np.nonzero(np.diff(padded) == 1)[0]
    ends = np.nonzero(np.diff(padded) == -1)[0] - 1

    for start, end in zip(starts, ends):
        if end - start + 1 < min_rays:
            continue

        rays = np.arange(start, end + 1)
        depth = stop[start:end + 1].astype(float)
        straight = np.polyfit(rays, depth, 1)
        curved = np.polyfit(rays, depth, min(order, end - start)) if order > 1 else straight

        for direction, anchor_ray in ((-1, start), (1, end)):
            # Does the curve ease off going this way, or steepen? The travel slope is the fit's slope in the
            # direction of the walk; the bend is constant. Opposite signs mean the turning point is ahead.
            travel = direction * float(np.polyval(np.polyder(curved), anchor_ray))
            bend = float(np.polyval(np.polyder(curved, 2), anchor_ray)) if len(curved) > 2 else 0.0
            fit = curved if travel * bend < 0.0 else straight

            # The fit's OWN value at the anchor, not stop[anchor_ray]: starting from the ray's measured stop
            # and adding fitted increments mixes two curves and puts a step at the join.
            previous, outward = float(np.polyval(fit, anchor_ray)), 0.0

            for distance in range(1, len(out)):
                ray = anchor_ray + direction * distance

                if not 0 <= ray < len(out):
                    break

                predicted = float(np.polyval(fit, ray))

                # Which way the continuation set off. Once it stops going that way it has passed the fit's
                # turning point and is curving back into the region, so there is nothing left to follow.
                if not outward:
                    outward = np.sign(predicted - previous)

                if outward * (predicted - previous) <= 0.0:
                    break

                previous = predicted

                # Stop at the wall rather than crossing it, and never propose a border behind the line.
                if predicted < 0.0 or predicted >= out[ray]:
                    break

                # Walk PAST a ray it may not touch rather than stopping: a single speckle-stopped ray in the
                # middle of the closed stretch would otherwise end the continuation there.
                if may_change is None or may_change[ray]:
                    out[ray] = int(round(predicted))

    return out


def sharp_stops(stop, angle_floor, rays=2, rounds=10):
    """Cut back any tip whose corner is sharper than `angle_floor` degrees. Her floor, not a fitted one.

    Alice, 2026-08-16: *"can you look at angles like it shouldn't just be a random acute angle line sticking
    out"*. `check_border_angles.py` measures the corner at every protruding ray over a 32 um baseline, on her
    14 outlines and on the rule's borders:

        HERS  sharpest 83 deg   p1  83   median 152        RULE  sharpest 15 deg   p1  18   median 110

    She never draws a corner sharper than 83 degrees, in 1777 rays. So this is the same kind of floor as the
    80 um mirror: read off her own outlines rather than fitted to IoU.

    WHY AN ANGLE CATCHES WHAT NOTHING ELSE DOES. A clamp measures how far a ray went, a median how many rays
    agree, the floors how long the black is. A stub is narrow AND deep at once, which is precisely a tip
    angle: 2 rays wide and 300 um deep is 6 degrees, while 400 um wide and 200 um deep is 90.

    THE ENDS ARE PADDED WITH ZERO DEPTH, because that is the shape -- past the last ray the outline turns and
    runs back to the Field L line. So this sees the 2D corner between the far border and the end cut, which is
    where both corners she has objected to are and the one place a cross-ray median is blind
    ([[end-rays-were-the-medians-blind-spot]]). A border that genuinely ends 700 um deep still measures ~90
    degrees and is untouched; only a NARROW deep end is cut.

    The cut is to the chord between the ray's two baseline neighbours and only ever inward, so this cannot
    invent depth. Repeating it matters: cutting one tip back exposes the next one, and Purp30's stub is two
    rays wide.
    """
    if angle_floor <= 0.0:
        return stop

    stop = stop.astype(float)

    for _ in range(rounds):
        padded = np.concatenate([np.zeros(rays), stop, np.zeros(rays)])
        back, ahead = padded[:len(stop)], padded[2 * rays:]

        along = rays * ALONG_BIN_UM
        left = np.stack([np.full(len(stop), -along), (back - stop) * ACROSS_BIN_UM], axis=1)
        right = np.stack([np.full(len(stop), along), (ahead - stop) * ACROSS_BIN_UM], axis=1)
        cosine = (np.sum(left * right, axis=1)
                  / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        sharp = (stop > back) & (stop > ahead) & (angle < angle_floor)

        if not sharp.any():
            break

        stop = np.where(sharp, np.minimum(stop, (back + ahead) / 2.0), stop)

    return np.rint(stop).astype(int)


def round_bends(stop, bend_floor, wall=None, found=None, rays=2, rounds=20):
    """Round any BEND sharper than `bend_floor` degrees, by averaging the ray with its two neighbours.

    Alice, 2026-08-16: *"its not better i meant rounder more for the sharp 100 ish degree angle on the very
    bottom"*. A BEND is a change of direction with no requirement that the ray stick out past both of its
    neighbours, which is exactly what `sharp_stops` cannot see -- it tests tips only, and a seam between two
    straight stretches is not a tip. Measured over the 14 sections at a 32 um baseline (logs/check bend
    angles.txt):

        HERS  sharpest  83 deg   p1 135   median 170   below 120 deg  0.9% of rays
        RULE  sharpest  69 deg   p1 116   median 170   below 120 deg  1.5% of rays

    So the border is already about as smooth as hers on average and the problem is a handful of SEAMS. The one
    she is pointing at is YW113 1-1-7 ray 103, where the curved continuation meets the flat stretch left of it:
    115 deg over 32 um, 107 deg over 80 um, against 143 deg for her sharpest bend anywhere on that section.

    ROUNDING, NOT CUTTING, and that is the difference from every other safeguard here. Averaging a ray with its
    neighbours is what turns a corner into a curve; the cross-ray MEDIAN deliberately does the opposite and
    preserves a bend, which is why it has never touched this seam. Iterated, because rounding one ray moves the
    bend to its neighbour.

    A ray may be pulled IN freely but never pushed OUT past black it found for itself -- the same asymmetry as
    `smooth_inward` ([[median-deletes-spikes-clamp-cannot]]): its neighbours are evidence about the border's
    shape, not about that ray.
    """
    if bend_floor <= 0.0:
        return stop

    stop = stop.astype(float)

    for _ in range(rounds):
        back = np.concatenate([stop[:rays], stop[:-rays]])
        ahead = np.concatenate([stop[rays:], stop[-rays:]])

        along = rays * ALONG_BIN_UM
        left = np.stack([np.full(len(stop), -along), (back - stop) * ACROSS_BIN_UM], axis=1)
        right = np.stack([np.full(len(stop), along), (ahead - stop) * ACROSS_BIN_UM], axis=1)
        cosine = (np.sum(left * right, axis=1)
                  / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        # Only the sharpest bend each round, so a run of rays is rounded from its worst point outward rather
        # than averaged wholesale into a straight line.
        sharp = angle < bend_floor

        if not sharp.any():
            break

        rounded = np.where(sharp, (back + stop + ahead) / 3.0, stop)

        if found is not None:
            rounded = np.where(found, np.minimum(rounded, wall), rounded)

        if np.allclose(rounded, stop):
            break

        stop = rounded

    return np.rint(stop).astype(int)


def reachable(region, bright, table):
    """Keep only the part of the region joined to the Field L line without crossing an OUTSIDE gap.

    Alice, 2026-08-16: *"pur30 1-1-8 isn't quite right on the bottom, it includes a black spike looking
    thing"*. The rays cross that cleft at a shallow angle, so they slip past it and the region wraps around
    the black and takes the sliver of tissue beyond it. No cue and no length floor can fix that -- the cleft
    IS found, the rays just do not run across it.

    Two steps, neither with a constant:

      1. delete black that connects to the empty slide around the section. The region may contain black,
         because she draws over interior gaps ([[regions-must-be-continuous-patches]]), and an interior gap
         does not reach the outside -- so this cannot punch the holes she objected to.
      2. keep the piece still joined to the band line. That drops the black itself AND anything only
         reachable across it, which is the sliver.

    4-connectivity on purpose: a diagonal chain of black pixels should block, and an 8-connected walk would
    squeeze through it.
    """
    outside = np.zeros(bright.shape, bool)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside = binary_propagation(outside & ~bright, mask=~bright)

    keep = region & ~outside
    seed = np.zeros(bright.shape, bool)
    at_line = table["step"] == 0
    seed[table["rows"][at_line], table["columns"][at_line]] = True

    return binary_propagation(seed & keep, mask=keep) if (seed & keep).any() else keep


def strip_cues(strip_bright, strip_um):
    """Line the strip cues up with their floors as two equal-length lists.

    One cue with one floor is the normal case and stays a plain float and a plain mask. A list of masks with
    a list of floors is the union: a ray stops at a long strip found by any of them. A list of masks with a
    single float means "same floor for all", which is deliberately allowed and deliberately not the default
    -- the cues do not share a scale.
    """
    cues = list(strip_bright) if isinstance(strip_bright, (list, tuple)) else [strip_bright]
    floors = list(strip_um) if isinstance(strip_um, (list, tuple)) else [float(strip_um)] * len(cues)

    if len(cues) != len(floors):
        raise ValueError(f"{len(cues)} strip cues but {len(floors)} floors")

    return cues, floors


def fill_between(bright, along, across, thickness_um, smooth_um=0.0, mode=BARRIER_MODE,
                 offset_um=0.0, half_um=NCM_LENGTH_UM / 2.0, dip=0.0, values=None, window=None,
                 bridge=False, clamp_um=0.0, strip_um=0.0, strip_bright=None, strip_edge=False,
                 strip_wide=False, strip_aspect=0.0, strip_rejoin=False, lift_um=0.0, dome=False,
                 dome_cap_um=0.0, dome_balance=False,
                 grow=False, smooth_early=False, smooth_inward=False, reach=False, angle_floor=0.0,
                 extend=False, extend_rays=3, extend_order=1, bend_floor=0.0, pair=False, ridge=False,
                 blind_um=0.0, record=None, return_stop=False):
    """Her rule: on every ray, fill from the Field L line out to the first wide gap.

    `bright` is "this pixel is not black". It used to be ps6.compute_tissue's mask, which made the
    whole rule depend on an absolute cutoff that fails in both directions -- see BLACK_LEVELS. It is
    now normally `bright_from(normalised green, level)`, so the cue is the image rather than a global
    binary mask. Nothing else in the function changes: outside the section is still dark, so the
    tissue edge is still just "an infinitely long gap" and needs no special case.

    `mode` is how "wide" is measured, and the review picture says it is the whole ballgame:

      run   the black run along the ray must be `thickness_um` long. WRONG, and wrong in a way that
            is invisible in the numbers: a 8 um crack lying PARALLEL to the ray reads as a gap
            hundreds of um long, so vessels and fissures stop rays at random. This is what made
            Gre595 collapse and what no downstream filter could repair.
      open   the black must contain a disc of `thickness_um` across, i.e. survive a morphological
            opening. Direction-free, so a crack stays a crack whichever way it runs. Kept as the
            default; `run` stays available only because the comparison is the evidence.
    """
    # Start the rays `offset_um` BEHIND the line by sliding the whole across axis, rather than by
    # relaxing ray_table's `across >= 0` test. Same set of pixels either way, but this keeps `step`
    # non-negative so it stays a valid array index -- and, more importantly, it keeps the fill starting
    # at step 0, so a barrier found inside the offset strip still stops the ray. Relaxing the test
    # instead would have let the fill begin partway along its own ray and skip past a barrier.
    if offset_um:
        across = across + offset_um

    if mode == "open":
        # inf means "no internal barrier at all": only the space outside the section's own silhouette
        # counts, which is her tissue edge on its own.
        barrier = ~binary_fill_holes(bright) if not np.isfinite(thickness_um) else gap_mask(bright, thickness_um)

        # A long thin strip is added to the BARRIER rather than handled as a third stop after the fact,
        # so it stops a ray by exactly the machinery the tissue edge already uses -- `needed` is one bin
        # in this mode, so a single barrier cell ends the ray. Nothing downstream needs to know the
        # strip exists, which is the point: bridge, clamp and smooth all keep working unchanged.
        # `strip_bright` lets the STRIP be found at a different idea of black from everything else.
        # Alice, 2026-08-15, on the band across the top of YW113 1-1-2: *"yes the bottom image is correct
        # now can you redraw the outline to include those magenta parts"* -- the bottom image was
        # `black < 0.08`, four times the rule's own level, and at 0.02 that band is barely there. Raising
        # the single level instead would move the tissue edge and the outside of the brain with it, so the
        # cue has to carry its own threshold or the change cannot be attributed.
        # A gap that TAPERS is still a gap. Alice, 2026-08-15, on the spike at the bottom of Wh175 1-1-9:
        # *"its actually a pretty fat gap though"*, *"its colored in blue in your depiction"* -- and she is
        # right, the separation is 80 um thick and blue on rays 98-101. It narrows to 72 um on rays 104-111,
        # where TWO rules then break it: the width test needs a full 80 um disc, and `long_strips` cannot
        # rescue the thin part because the rind fix deletes thin black within 40 um of a wide gap -- which
        # the thin part of the same band is, being joined to its own wide half. The rays pour through the
        # break into the sliver beyond and overrun her border by up to 304 um.
        #
        # Growing the barrier through connected black fixes it with NO constant: a component that is wide
        # anywhere becomes a wall everywhere. Speckle is untouched, because it is connected to no wide gap.
        if grow:
            barrier = binary_propagation(barrier, mask=~bright)

        # `strip_bright` and `strip_um` may each be a SEQUENCE, which makes a ray stop at a strip found by
        # ANY of several cues -- the two measured cues own different sections (`compare_cues.py`: the
        # relative cue wins OR408 by 0.046, the absolute one wins Purp30 by 0.092) and neither is a
        # refinement of the other, so the union is the only way to have both. Each cue keeps its own floor
        # because the floors are on different scales: 0.02 calls 0.5-6% of the tissue dark and 0.8-relative
        # calls 12-37%, so one number cannot filter both.
        # The strips are kept SEPARATELY as well, so a ray's stop can be asked which kind of barrier it
        # is on. A wide gap is the brain's surface or a fold; a long thin strip is a separation, and only a
        # separation is a piece of her border with a direction worth continuing (see extend_strips).
        strips = np.zeros(bright.shape, bool)

        for cue, floor in zip(*strip_cues(strip_bright, strip_um)):
            if (floor > 0.0 or strip_edge) and np.isfinite(thickness_um):
                strips = strips | long_strips(bright if cue is None else cue, thickness_um, floor,
                                              strip_edge, strip_wide, strip_aspect,
                                              strip_rejoin)

        barrier = barrier | strips
        table = ray_table(bright, along, across, barrier, half_um, values, window)
        strip_table = (ray_table(bright, along, across, strips, half_um, values, window)
                       if extend and strips.any() else None)
        needed = ACROSS_BIN_UM
    else:
        table = ray_table(bright, along, across, None, half_um, values, window)
        needed = thickness_um

    if table is None:
        return np.zeros(bright.shape, bool)

    # A snapshot of the stops as they leave every stage, for diagnostics ONLY -- appended to a list handed
    # back through `record`, so no behaviour changes and no caller signature does either. Alice, 2026-08-17:
    # *"your predictions before were pretty accurate there was just one area where it was picking up the wrong
    # thing"*. Ten stages run between the raw wall and the final border, and a single IoU cannot say which of
    # them moved a ray off her line -- which is exactly the question a localised error asks.
    stages = []

    def note(name):
        """`stop` as it stands now, under `name`. Reads the enclosing `stop` at call time, after each rebind."""
        if record is not None:
            stages.append((name, stop.copy()))

    stop = first_wide_gap(table["black"], needed)

    # The two cues stop the ray INDEPENDENTLY and the earlier one wins, rather than the dip replacing
    # the gap: the gap rule is what makes the tissue edge work, and it needs no per-ray reference to do
    # it. Whichever fires first is the border. Applied BEFORE the median so smooth_stops smooths the
    # combined decision -- smoothing only the gap stop and then overriding it would throw the smoothing
    # away on exactly the rays the dip changed.
    if dip:
        if values is None:
            raise ValueError("dip needs `values` (the normalised green channel) to build a profile")

        stop = np.minimum(stop, dip_stops(table["mean"], table["seen"], dip, stop))

    # Where each ray found a wall FOR ITSELF, kept before bridging replaces the misses with guesses.
    wall, found = stop.copy(), stop < table["black"].shape[1]
    strong_found = found.copy()

    # A SHORTER LENGTH FLOOR, FOR THE BLIND RAYS ONLY. Alice, 2026-08-17, having redrawn OR408's HP outline
    # onto the black gap: *"your predictions before were pretty accurate there was just one area where it was
    # picking up the wrong thing"*.
    #
    # That area is 55 consecutive rays that find no wall at all, and `diag_stage_trace.py` shows the ten
    # stages after the bridge changing nothing on them -- their error is entirely the neighbours' answer being
    # copied in (156 um median, against 48 um on rays with a wall). Measured on her redrawn line, 45 of the 55
    # DO have a relatively dark piece under her border, median own length 248 um, longest 496 um. The relative
    # cue's floor is 608 um, so not one of them can clear it: the cue marks her strip and the floor deletes it.
    #
    # WHY NOT JUST LOWER THE FLOOR. `scan_relative_floor.py`: 608 wins all 7 folds unanimously, and dropping
    # it to 400 takes the mean IoU from 0.831 to 0.628 -- recall 0.925 to 0.688, because the 0.8 cue calls
    # 20-37% of the tissue dark and a short floor stops rays early EVERYWHERE. OR408 alone prefers 400
    # (0.750 against 0.733), which is precisely the trade that beat three earlier border decodes
    # ([[border-detector-hurts-the-region]]).
    #
    # Restricting it to `~found` is what makes it safe rather than merely better on average, and the argument
    # is arithmetic, not statistical: a ray that found a wall never consults this pass, and 14 of the 15
    # sections have NO blind rays since the barrier grew through connected black ([[tapered-gaps-leak-rays]]),
    # so they cannot move whatever this floor is set to. A rescued ray is written into `wall` and `found` as
    # well, so `smooth_inward` treats it as evidence -- it is a wall the ray found for itself, just under a
    # weaker filter -- and `strong_found` keeps the strict answer for diagnostics.
    if blind_um and mode == "open" and np.isfinite(thickness_um) and not found.all():
        weak = np.zeros(bright.shape, bool)

        for cue, floor in zip(*strip_cues(strip_bright, strip_um)):
            if floor > 0.0 or strip_edge:
                weak = weak | long_strips(bright if cue is None else cue, thickness_um,
                                          min(floor, blind_um) if floor > 0.0 else blind_um,
                                          strip_edge, strip_wide, strip_aspect,
                                          strip_rejoin)

        # A second table, because the strips live in the BARRIER and a ray's black profile is built from it.
        # Paid for only when a section has blind rays, i.e. once in fifteen.
        weak_table = ray_table(bright, along, across, barrier | weak, half_um, values, window)

        if weak_table is not None:
            weak_stop = first_wide_gap(weak_table["black"], needed)
            rescued = ~found & (weak_stop < table["black"].shape[1])

            stop = np.where(rescued, weak_stop, stop)
            wall = np.where(rescued, weak_stop, wall)
            found = found | rescued

            # The extension asks whether a ray's wall sits on a STRIP rather than on a wide gap, so it has to
            # see the weak strips too or a rescued ray reads as standing on the brain's surface and its
            # direction is never continued ([[follow-the-strip-where-black-closes]]).
            if extend and rescued.any():
                strips = strips | weak
                strip_table = ray_table(bright, along, across, strips, half_um, values, window)

    # An optional out-parameter for diagnostics ONLY. `found` is the single most useful thing a caller
    # cannot recompute -- the far border is already right to 8-48 um wherever a ray finds black
    # ([[far-border-is-solved-where-the-gap-is]]), so any new cue has to be judged on the rays where
    # `found` is False, and re-deriving that outside would duplicate the whole strip machinery. Handing
    # back a reference rather than adding a return value keeps every existing caller untouched.
    if record is not None:
        record.update({"wall": wall, "found": found, "table": table, "strip_table": strip_table,
                       "stages": stages, "strong_found": strong_found})

    note("wall")

    # THE BLIND RAYS ONLY, and that restriction is the whole reason this is allowed to exist. Every earlier
    # border cue was applied to all rays and lost to the constant rule ([[border-detector-hurts-the-region]]),
    # because a ray that found a wall is already within 8-48 um of her line ([[far-border-is-solved-where-the-gap-is]])
    # and a cue that fires on 27% of correct rays can only break those ([[border-cue-is-calibrated-not-specific]]).
    # A ray with `found` False has no answer at all: today it COPIES ITS NEIGHBOURS through bridge_stops, and
    # that copy is the bar. `diag_pair_stop.py`: on OR408's 54 blind rays the median distance from her border
    # falls from 180 um to 112 um, closer on 57% of them, overshooting by only +24 um. Every other section has
    # no blind rays left since the barrier grew through connected black, so this CANNOT move them.
    if pair:
        if values is None:
            raise ValueError("pair needs `values` (the normalised green channel) to build a profile")

        width = table["black"].shape[1]
        blind = ~found

        if blind.any():
            # The two numbers are read from the module HERE rather than taken as defaults inside
            # `pair_stops`, so a scan can set them on the module and have them take effect: a default
            # argument is bound once when the function is defined and would ignore the change.
            guess = pair_stops(table["mean"], table["seen"], np.full(len(stop), width),
                               reach_um=PAIR_REACH_UM, start_um=PAIR_START_UM)
            stop = np.where(blind & (guess < width), guess, stop)
            note("pair")

    # The SECOND kind of separation she draws on, for the same rays and under the same arithmetic guarantee.
    # Alice circled it herself on OR408 and there is no black on her border inside that circle at all: the wall
    # is a 60 um strip BRIGHTER than the tissue on both sides (`read_circle.py`). On the 20 rays that still find
    # nothing the brightest ridge lands a median 16 um from her border against the copy's 64
    # (`diag_bright_ridge.py --rays gapless`), and it is left out of the other 54 deliberately -- there it lands
    # 160 um out and the weakly-filtered wall is better, so a ridge never overrules a wall.
    if ridge:
        if values is None:
            raise ValueError("ridge needs `values` (the normalised green channel) to build a profile")

        width = table["black"].shape[1]
        blind = ~found

        if blind.any():
            guess = ridge_stops(table["mean"], table["seen"], np.full(len(stop), width),
                                start_um=RIDGE_START_UM)
            stop = np.where(blind & (guess < width), guess, stop)
            note("ridge")

    # Bridge BEFORE clamping and smoothing: a ray at the image's edge is not a rough border, it is a
    # missing answer, and letting the clamp deal with it would drag its neighbours out instead. Any ray the
    # pair just answered now reads as found here, so the copy no longer overwrites it.
    if bridge:
        stop = bridge_stops(stop, table["black"].shape[1])
        note("bridge")

    # A MISSING answer and a WRONG answer want different repairs, and `smooth_early` is which of the two a
    # spike is treated as. Late (the default) the median is the last word on the shape, but by then
    # `lift_stops` has already dragged a run of correct neighbours out to meet the outlier, so the median has
    # to be wide enough to outvote the smear as well as the spike. Early it runs on the bridged stops, where
    # an isolated overrun is still only as wide as itself, and the mirror never sees it -- so there is nothing
    # to smear. Alice, 2026-08-16: *"if you could smooth spikes and do something to prevent them"*.
    if smooth_early:
        stop = smooth_stops(stop, smooth_um)
        note("median")

    # The median may pull a ray IN but never push it OUT past a wall the ray found for itself, because a
    # found wall is evidence and its neighbours' votes are not evidence about this ray. Both of Alice's
    # cases need exactly this asymmetry: a spike's own wall is far outside the median so the median still
    # wins (YW113 1-1-3), while a short run that dipped into a real black notch keeps its own answer instead
    # of being voted out of it -- Alice, 2026-08-16: *"pur30 1-1-8 isn't quite right on the bottom, it
    # includes a black spike looking thing"*. A ray that never found a wall is left to the median and the
    # bridge, since there is nothing to be faithful to.
    if smooth_inward:
        stop = np.where(found, np.minimum(stop, wall), stop)
        note("faithful")

    # AFTER the median and BEFORE the mirror. Alice, 2026-08-16: *"follow that line and for the stuff in
    # between just fill it out following that concave shape and connect to the outer tissue"*.
    if extend and strip_table is not None:
        on_strip = np.zeros(len(stop), bool)
        inside = wall < strip_table["black"].shape[1]
        on_strip[inside] = strip_table["black"][np.nonzero(inside)[0], wall[inside]]
        stop = extend_strips(stop, on_strip, ~on_strip, extend_rays, extend_order)
        note("extend")

    # BEFORE the mirror on purpose. Purp30's corner is the case: rays 2-3 stop raw at 224/256 um against
    # her 176/248 -- the magenta cut, found correctly -- and lift_stops then pushed them to 488/408 to keep up
    # with the 32 um stub on rays 0-1. Alice, 2026-08-16: *"i also dont understand cuz if you just follow the
    # magenta cuts it should be right"*. She is right, and cutting the stub first is what lets the mirror
    # agree with her instead of overriding her.
    stop = sharp_stops(stop, angle_floor)
    note("sharp")

    stop = clamp_stops(stop, clamp_um)
    note("clamp")
    stop = lift_stops(stop, lift_um, table["black"].shape[1])
    note("mirror")
    stop = dome_stops(stop, table["black"].shape[1], dome, dome_cap_um, dome_balance)
    note("dome")

    # LAST, after the mirror. A bend is rounded by averaging, and the mirror would flatten the rounding back
    # out again if it ran afterwards -- while rounding cannot create a new inward spike for it to catch.
    stop = round_bends(stop, bend_floor, wall, found)
    note("bends")

    if not smooth_early:
        stop = smooth_stops(stop, smooth_um)
        note("median late")

    region = np.zeros(bright.shape, bool)
    region[table["rows"], table["columns"]] = table["step"] < stop[table["ray"]]

    if reach:
        region = reachable(region, bright, table)

    # The stops are the border AS A CURVE, one number per ray, and they are what a regularity test has to
    # look at -- the region mask cannot say which of two neighbouring pixels belonged to the same ray.
    # Alice, 2026-08-15, on the teeth in Gre595: *"are there measures to ensure that the shape is regular
    # because the ups and downs are not normal"*. `check_border_smooth.py` measured the same quantity on
    # HER outlines (median 8 um per ray, p99 80, max 192) but only for the plain gap rule, so it cannot
    # see the strips. Handing the array back costs nothing and keeps one implementation of the fill.
    if return_stop:
        return region, stop, table["black"].shape[1]

    return region


def today_rule(tissue, along, across):
    """The shipping rule, for comparison: mask closed first, then windowed, then largest blob."""
    inside = (
        solid_tissue(tissue)
        & (across >= 0.0)
        & (along >= -NCM_LENGTH_UM / 2.0)
        & (along <= NCM_LENGTH_UM / 2.0)
    )

    return largest_blob(binary_fill_holes(inside)) if inside.any() else inside


def load():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    sections = []

    for index, (stem, name) in enumerate(zip(stems, images)):
        truth = data["labels"][index] == 1

        if not truth.any():
            continue

        green = data["inputs"][index, 0].astype(np.float32)
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])

        band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size, scale_um))

        if band is None:
            print(f"  {stem[:36]:38s} no band found, skipped")
            continue

        along, across = band_frame(band, tissue, is_left_hemisphere(name))

        sections.append({
            "stem": stem,
            "name": name,
            "bird": stem.split("_")[0],
            "tissue": tissue,
            "along": along,
            "across": across,
            "truth": truth,
            "pixel_size": pixel_size,
            # Carried so the rule can read the IMAGE for its black cue instead of the tissue mask.
            # It was already loaded here for find_band and simply never reached the rule.
            "green": green,
            # The bright patch's OWN SPREAD along the band normal, in um. Measured off `across`
            # directly, which already IS the signed distance along that normal in um -- so no need to
            # recover the normal vector, and no risk of measuring it along a slightly different axis
            # than the one the rule uses. See BACKSETS: this is what the correction scales with.
            "spread": patch_spread(green, tissue, across),
        })

    return sections


def patch_spread(green, tissue, across):
    """p90 - p10 of `across` over the bright patch, in um. 0.0 if there is no patch."""
    found = bright_patch(green, tissue)

    if found is None:
        return 0.0

    low, high = np.percentile(across[found[0][0]], EDGE_PERCENTILES)

    return float(high - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thickness", type=float, default=None,
                        help="fixed gap width in um; omit for the leave-one-bird-out sweep")
    parser.add_argument("--smooth", type=float, default=None,
                        help="fixed median window in um along the band; omit to sweep")
    parser.add_argument("--mode", choices=("open", "run"), default=BARRIER_MODE,
                        help="how gap width is measured; see fill_between")
    parser.add_argument("--black", default=None,
                        help="cutoff on normalised green for 'black', or 'mask' for the old tissue "
                             "mask; omit to sweep")
    parser.add_argument("--half", default=None,
                        help="along-window half-length in um, or 'LEASH@FRACTION' for the per-section "
                             "walk -- 'inf@0.50' walks freely and stops at anything dimmer than half the "
                             "band's own brightness, '150@0' walks at most 150 um to the tissue edge; "
                             "omit to sweep")
    parser.add_argument("--back", default=None,
                        help="how far behind the Field L line the rays start: '100u' for a constant "
                             "100 um, '0.15f' for 15%% of the patch spread; omit to sweep")
    parser.add_argument("--dip", type=float, default=None,
                        help="stop where brightness falls below this fraction of the ray's own tissue; "
                             "0 disables the relative cue. Omit to sweep")
    args = parser.parse_args()

    # A number if it parses as one, so --black 0.15 sweeps nothing but --black mask still works.
    if args.black is not None and args.black != "mask":
        args.black = float(args.black)

    # 'inf@0.50' -> (inf, 0.50). One string rather than two flags, for the same reason as --back: a
    # setting that can be half-specified is a setting that silently measures something else.
    if args.half is not None:
        if "@" in args.half:
            leash, fraction = args.half.split("@")
            args.half = (float(leash), float(fraction))
        else:
            args.half = float(args.half)

    # '100u' -> ("um", 100.0), '0.15f' -> ("frac", 0.15). One string rather than two flags so a
    # setting can never be half-specified, which is how a run silently measures something else.
    if args.back is not None:
        args.back = (("frac", float(args.back[:-1])) if args.back.endswith("f")
                     else ("um", float(args.back.rstrip("u"))))

    sections = load()

    if not sections:
        raise SystemExit("no scoreable sections")

    for section in sections:
        rule = today_rule(section["tissue"], section["along"], section["across"])
        section["today"] = overlap(rule, section["truth"])
        section["cells_today"] = cells_kept(rule, section["truth"], section["name"], section["pixel_size"])

    thicknesses = (args.thickness,) if args.thickness is not None else THICKNESSES
    smooths = (args.smooth,) if args.smooth is not None else SMOOTHS
    levels = (args.black,) if args.black is not None else BLACK_LEVELS
    backsets = (args.back,) if args.back is not None else BACKSETS
    halves = (args.half,) if args.half is not None else HALF_LENGTHS
    dips = (args.dip,) if args.dip is not None else DIPS
    settings = [(t, s, b, o, h, d) for t in thicknesses for s in smooths for b in levels
                for o in backsets for h in halves for d in dips]

    print(f"\n  FILL BETWEEN THE TWO BORDERS, width measured by '{args.mode}'. 'inf' gap = never stop"
          f" at a gap, only at the section's outer edge.")
    print(f"  'black' is the cue for non-tissue: 'mask' = the old absolute ps6.compute_tissue mask,"
          f" a number = that cutoff on normalised green.\n")
    print(f"  {'gap':>6s} {'median':>8s} {'black':>6s} {'back':>7s} {'half':>6s} {'dip':>5s} {'IoU':>7s} {'recall':>8s} {'precision':>10s} "
          f"{'cells kept':>11s} {'Gre595':>8s} {'YW113 1-1-2':>12s}")

    torn = [s for s in sections if "Gre595_RH_3-1-4" in s["stem"]]
    # The opposite failure: the brightest section, where a real gap sits ABOVE the absolute cutoff.
    bright_section = [s for s in sections if "YW113_RH_1-1-2" in s["stem"]]

    for setting in settings:
        thickness, smooth, level, backset, half, dip = setting

        for section in sections:
            bright = section["tissue"] if level == "mask" else bright_from(section["green"], level)
            window = along_window(half, section["along"], section["across"], bright,
                                  section["green"])
            region = fill_between(
                bright, section["along"], section["across"], thickness, smooth, args.mode,
                backset_um(backset, section["spread"]), NCM_LENGTH_UM / 2.0 if window else half,
                dip, section["green"], window,
            )
            section[setting] = {
                "score": overlap(region, section["truth"]),
                "cells": cells_kept(region, section["truth"], section["name"], section["pixel_size"]),
            }

        scores = [s[setting]["score"] for s in sections]
        cells = [s[setting]["cells"] for s in sections if s[setting]["cells"] is not None]
        kept = f"{100 * np.mean(cells):10.1f}%" if cells else f"{'-':>11s}"
        gap_label = "inf" if not np.isfinite(thickness) else f"{thickness:.0f}u"
        torn_iou = f"{np.mean([s[setting]['score']['iou'] for s in torn]):8.3f}" if torn else f"{'-':>8s}"
        bright_iou = (f"{np.mean([s[setting]['score']['iou'] for s in bright_section]):12.3f}"
                      if bright_section else f"{'-':>12s}")
        # THREE decimals, not two. At .2f both 0.005 and 0.01 print as "0.01", so the first sweep
        # showed two identical-looking rows with different numbers behind them and there was no way
        # to tell which level won. A printed number is a lossy summary of a value; if two settings
        # render the same, the table is lying while every computation behind it is correct.
        level_label = level if isinstance(level, str) else f"{level:.3f}"

        back_label = f"{backset[1]:.0f}u" if backset[0] == "um" else f"{backset[1]:.2f}xsp"
        half_label = window_label(half)
        dip_label = "off" if not dip else f"{dip:.2f}"
        print(f"  {gap_label:>6s} {smooth:7.0f}u {level_label:>6s} {back_label:>7s} {half_label:>6s} "
              f"{dip_label:>5s} "
              f"{np.mean([s['iou'] for s in scores]):7.3f} "
              f"{np.mean([s['recall'] for s in scores]):8.3f} "
              f"{np.mean([s['precision'] for s in scores]):10.3f} {kept} {torn_iou} {bright_iou}")

    today_cells = [s["cells_today"] for s in sections if s["cells_today"] is not None]
    print(f"\n  {'today':>15s} {np.mean([s['today']['iou'] for s in sections]):7.3f} "
          f"{np.mean([s['today']['recall'] for s in sections]):8.3f} "
          f"{np.mean([s['today']['precision'] for s in sections]):10.3f} "
          f"{100 * np.mean(today_cells):10.1f}% "
          f"{np.mean([s['today']['iou'] for s in torn]) if torn else float('nan'):8.3f}")

    if (args.thickness is not None and args.smooth is not None and args.black is not None
            and args.back is not None and args.half is not None and args.dip is not None):
        return

    print(f"\n  LEAVE-ONE-BIRD-OUT: all three knobs picked on the other birds, scored on the held-out one\n")
    print(f"  {'held-out':10s} {'picked':>28s} {'IoU now':>9s} {'IoU new':>9s} "
          f"{'prec now':>9s} {'prec new':>9s} {'cells now':>10s} {'cells new':>10s}")

    rows = []

    for held in sorted({s["bird"] for s in sections}):
        others = [s for s in sections if s["bird"] != held]
        mine = [s for s in sections if s["bird"] == held]

        best, choice = -1.0, settings[0]

        for setting in settings:
            mean_iou = float(np.mean([s[setting]["score"]["iou"] for s in others]))

            if mean_iou > best:
                best, choice = mean_iou, setting

        row = {
            "iou_now": float(np.mean([s["today"]["iou"] for s in mine])),
            "iou_new": float(np.mean([s[choice]["score"]["iou"] for s in mine])),
            "prec_now": float(np.mean([s["today"]["precision"] for s in mine])),
            "prec_new": float(np.mean([s[choice]["score"]["precision"] for s in mine])),
            "cells_now": [s["cells_today"] for s in mine if s["cells_today"] is not None],
            "cells_new": [s[choice]["cells"] for s in mine if s[choice]["cells"] is not None],
        }
        rows.append(row)

        cells = (
            f"{100 * np.mean(row['cells_now']):9.1f}% {100 * np.mean(row['cells_new']):9.1f}%"
            if row["cells_now"] and row["cells_new"] else f"{'-':>10s} {'-':>10s}"
        )
        gap_label = "inf" if not np.isfinite(choice[0]) else f"{choice[0]:.0f}"
        # .3f for the same reason as the sweep table above: 0.005 and 0.01 are indistinguishable at
        # two decimals, and this column's whole job is to say WHICH setting the fold picked.
        black_label = choice[2] if isinstance(choice[2], str) else f"{choice[2]:.3f}"
        back_label = f"{choice[3][1]:.0f}u" if choice[3][0] == "um" else f"{choice[3][1]:.2f}xsp"
        half_label = window_label(choice[4])
        dip_label = "off" if not choice[5] else f"{choice[5]:.2f}"
        label = f"{gap_label}u/{choice[1]:.0f}u/{black_label}/{back_label}/{half_label}/{dip_label}"

        print(f"  {held:10s} {label:>16s} {row['iou_now']:9.3f} {row['iou_new']:9.3f} "
              f"{row['prec_now']:9.3f} {row['prec_new']:9.3f} {cells}")

    cells_now = [v for r in rows for v in r["cells_now"]]
    cells_new = [v for r in rows for v in r["cells_new"]]

    print(f"\n  {'mean':10s} {'':12s} {np.mean([r['iou_now'] for r in rows]):9.3f} "
          f"{np.mean([r['iou_new'] for r in rows]):9.3f} "
          f"{np.mean([r['prec_now'] for r in rows]):9.3f} "
          f"{np.mean([r['prec_new'] for r in rows]):9.3f} "
          f"{100 * np.mean(cells_now):9.1f}% {100 * np.mean(cells_new):9.1f}%")


if __name__ == "__main__":
    main()
