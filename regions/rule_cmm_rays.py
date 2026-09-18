"""NCM's rule, pointed at CMM: rays out from the Field L line, each stopping at the first black gap.

Alice, 2026-08-19: *"i meant whatever you did in ncm to prevent the region from crossing the black
separations, do on the cmm region"*.

WHAT SHE IS POINTING AT. CMM and NCM have been built by two completely different mechanisms:

  NCM   the region is an INTERVAL on every ray out from the Field L line -- fill from the line until
        the ray meets a wide black gap or leaves the section ([[regions-are-two-lines-plus-tissue]]).
        Interior holes are inside the interval so they fill themselves, the patch is contiguous by
        construction, and the tissue edge needs no special case because outside the section is an
        infinitely long gap. On top of that sit every refinement she and I measured: the long thin
        strip cue with a floor that scales with the section's own darkness
        ([[relative-cue-and-floor-union]]), `grow` for gaps that taper below the width test
        ([[tapered-gaps-leak-rays]]), `reach` to follow a strip to where its black closes
        ([[follow-the-strip-where-black-closes]]), a running median across neighbouring rays that
        deletes spikes ([[median-deletes-spikes-clamp-cannot]]) with the inward pass and the 83 degree
        angle floor ([[an-angle-floor-frees-the-mirror]]), and `extend` for the end rays
        ([[end-rays-were-the-medians-blind-spot]]).
  CMM   a trapezoid or 16 fitted radii, clipped to the TISSUE MASK, with `black_barrier` used only to
        cap a marching far border. None of the above. The tissue mask cannot see a black separation at
        all ([[dark-laminae-are-real-borders]]), which is exactly why CMM runs into hippocampus.

THE PORT IS A COORDINATE FLIP, NOT A REWRITE, and that is deliberate: every flag below is copied
byte-for-byte from `review_where_wrong.rule_border`, the NCM rule as it ships. If this wins, it wins
because the mechanism is better and not because something was retuned for CMM.

  `fill_between` fills rays where `across >= 0`, outward from `across = 0`. CMM sits at NEGATIVE
  `across`, on the far side of the band, so the axis is flipped about CMM's own line:
  `across -> a0 - section["across"]`, with `a0` from `rule_trapezoid.near_border`. Zero is then her
  CMM line and positive is into CMM, which is the frame the rule already expects.

  The ALONG WINDOW comes from CMM's own fitted ends, not from NCM's step decode. Her two Field L lines
  are different lines with different lengths ([[field-l-has-two-different-lines]]), so reusing NCM's
  window here would be measuring the wrong line.

THE ENDS. The trapezoid's ends are TILTED -- `end high` tilts one way on 15 of 15 sections and `end
low` the other on 13 of 15 -- and a ray window is straight, so a plain fan squares them off. `TAPER`
now gives 9 of 15 sections one end back: where the rays have overrun the end of the black separation,
the border closes on the Field L line along a straight tangent instead of squaring off (`taper_ends`,
0.506 -> 0.526 with cells kept unchanged). The other six ends are still square.

Usage:
  python regions/rule_cmm_rays.py
  python regions/rule_cmm_rays.py --label 1        # the same rule on NCM, as a control it should roughly match
"""

import argparse

import numpy as np
from scipy.ndimage import label, uniform_filter

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

from polygon_solid import LEASH_UM, MIN_LABEL_PX, solid_offsets, tally
from review_pair import ANGLE, EXTEND_ORDER, EXTEND_RAYS, EXTRA_UM, K, LIFT, RATIO, SMOOTH
from review_polygon import sections
from rule_step_stop import THICKNESS_UM
from rule_trapezoid import fit, measure_one, near_border, trapezoid
from rule_two_borders import (
    ACROSS_BIN_UM,
    ALONG_BIN_UM,
    BLACK_LEVELS,
    bright_from,
    fill_between,
    reachable,
    relative_bright,
)
from scan_cue_and_floor import dark_percent

# How many straight segments the far border is allowed. Alice, 2026-08-19, on the ray sheet: *"it looks
# good, can you straighten the lines out"* -- and she has always described this region as a shape rather
# than a curve: *"the cmm region is a convex polygon"*. 0 leaves the border as the raw per-ray curve.
SEGMENTS = 1

# Whether a straightened ray is still forbidden to pass a wall IT FOUND ITSELF. Alice, 2026-08-19, on the
# straightened sheet: *"it still sticks in yw1-1-7 at right bottom"* -- and she is right, and it is the
# straightening that did it. Fitting one line to the found rays and imposing it on all of them necessarily
# pushes some found rays PAST their own black. Clipping them back keeps the border straight exactly where
# straightness is doing the work (the blind rays) and lets the image win where the image has something to
# say, which is also how she draws: straight lines that stop at the separation.
# BACK ON. Alice, 2026-08-19, having seen both sheets: *"the other one was better, but why can't you do
# both"*. TEN ways of having both were measured (`diag_slide.py`) and the one that works is hers -- *"wait
# why can't you straighten then clip"* -- with one more straightening after it, which is `REFIT` below.
#
# WHY THE CLIP ALONE LEAVES A STAIRCASE is visible in `look_or99_walls.py` rather than in any aggregate: on
# OR99_RH_1-1-4 the black is a broad ragged patch, her border is a straight line through its FAR side, and
# the clip traces its frayed NEAR edge. Crossing the patch outright (`through=1.0`) also fixes that section
# (0.541) but costs 9 of the other 14, which all want the near edge; re-fitting after the clip fixes it
# better (0.550) and costs only YW113 1-1-7.
CLIP_TO_WALL = True

# HOW LONG A STRETCH OF AGREEING WALLS SLIDES THE WHOLE LINE INWARD. Alice, 2026-08-19, having seen both
# versions: *"the other one was better, but why can't you do both"*.
#
# BOTH IS POSSIBLE BECAUSE THE WALL CAN MOVE THE LINE INSTEAD OF THE RAYS. `CLIP_TO_WALL` pulls each found
# ray back to its own wall independently of its neighbours, and independent pull-backs on neighbouring rays
# ARE a staircase -- which is why the identical clip gains 0.05 on YW113 1-1-7, where a long black band
# makes every neighbouring wall agree, and loses 0.05 on OR99_RH_1-1-4, where a bright vessel scatters
# walls at unrelated depths. Sliding the RIGID line instead can only ever produce one straight border, and
# a run test decides how far: a long stretch of agreeing walls is a border and moves it, a few scattered
# walls are noise and move nothing. This differs from the `clip_runs` idea (0.500, rejected) in that that
# one still moved rays one at a time and only gated WHICH ones, so it kept the staircase.
# OFF: measured at 150/300/500/800 um. It half-fixes OR99_RH_1-1-4 (0.505 -> 0.514) and LOSES YW113 1-1-7
# (0.627 -> 0.584), because that section's black band CURVES and no rigid line can sit behind a curve, and
# it drops cells kept from 88% to 72-81%.
SLIDE_RUN_UM = 0.0

# OFF now that the clip is back on: the two are alternatives, and stacking them ("clip + 25th") scored the
# same 0.508 while costing OR99_RH_1-1-4 another 0.03.
#
# WHICH PERCENTILE OF THE FOUND WALLS THE STRAIGHT BORDER SITS AT. A least-squares line runs through the
# middle of them, so half the found rays cross black; sliding it to the 25th percentile of its own
# residuals leaves a quarter outside and keeps ONE straight border instead of a straight border plus a
# per-ray correction fighting it. Swept 20/25/30/35: 25 is best on mean IoU (0.503), best on density
# (78%), and nearest to her area (x1.12). Higher helps OR99_RH and hurts YW113 1-1-7; lower loses cells
# (kept 87% -> 82% at the 20th).
INWARD = 0.0

# HOW MUCH OF A RAY'S TIP HAS TO BE BLACK before that end ray is chopped off, and how deep a tip is.
# Alice, 2026-08-19: *"just chop the end off then buddy"*. A straightened border can still leave the last
# few rays running to a point that pokes past the separation, because those rays found no wall of their
# own and so have nothing to be clipped against -- `extend` gave them an interior ray's stop, which is too
# long for them. The chop walks inward from each end and drops rays until one lands clean.
# HOW LONG A BLACK RUN A RAY MUST CROSS before that end ray is chopped off. Alice, 2026-08-19: *"just
# chop the end off then buddy"*. The FIRST version of this tested whether the ray's TIP was black and fired
# on zero rays out of 15 sections, which was the useful failure: once a ray crosses a separation it carries
# on into the tissue on the far side, so its tip is BRIGHT. The evidence that a ray went too far is a black
# run somewhere BEHIND its tip, not black at it.
# HOW FAR THROUGH ITS OWN BLACK RUN a wall is placed: 0 is first contact, 0.5 the run's middle. See
# `wall_middle`. Swept below.
THROUGH = 0.0

# WHETHER ONE STRAIGHT LINE IS FITTED AGAIN THROUGH THE CLIPPED STOPS. OFF. Alice, 2026-08-19, having seen
# it: *"you need to straigh then clip since yw 113 1-1-7 goes back to whatever and none of the clipping
# relaly does anything"*.
#
# SHE IS DESCRIBING WHAT THE REFIT DOES TO THE CLIP, and she is right about the mechanism. Re-fitting a line
# through the clipped stops reduces every local pull-back to ONE global shift of the border, so the clip
# stops being evidence about where the border runs and becomes evidence about how deep it sits -- which is
# also why it scores the same as the fixed 25th-percentile slide it is meant to improve on. On YW113 1-1-7
# the wall's information IS local (a curving black band, 0.627 clipped against 0.590 refitted), so averaging
# it away throws out the only section where the black really tracks her border.
#
# The mean IoU prefers the refit by 0.002 (0.508 vs 0.506) and OR99_RH_1-1-4 by 0.045. That is a smaller
# number than a section where the rule is demonstrably following the right thing, and she has been consistent
# that the aggregate is not the thing to optimise: *"the cell count is not that important just get the region
# right"*.
REFIT = False

# WHETHER THE BORDER CLOSES ON THE FIELD L LINE past the end of the identified black separation, instead of
# running on at full depth and squaring off against the window's end. Alice, 2026-08-19, on YW113_RH_1-1-2:
# *"can you just connect the end of the identified black separation to the field l line rather than have that
# jut out"*. See `taper_ends`.
TAPER = True

# WHERE THE SEPARATION ENDS: how far beyond a ray's wall to look for tissue, how much of that stretch has to
# be tissue for the wall to count as INSIDE the section, and how long a run of edge-walled rays it takes.
#
# TWO ANCHORS THAT DO NOT WORK, measured, do not rebuild them. (1) The last ray that FOUND black: on
# YW113_RH_1-1-2 all 91 rays found black -- the flap's rays find the section's OUTSIDE -- so it fired nowhere
# there and instead tapered OR99_RH_1-1-4's 23 blind end rays, 0.505 -> 0.353. (2) Where the wall depth steps
# outward by more than a threshold (240 um on YW113_RH_1-1-2, 192 um on OR99_RH_1-1-4's high end): it cannot
# tell that from her region legitimately narrowing, and cut half of OR99_RH_1-1-4 away, 0.505 -> 0.224.
REACH_BEYOND_UM = 160.0
TISSUE_SHARE = 0.5
RUN_UM = 96.0

# HOW LONG a run of edge-walled rays can be and still count as an END FLAP rather than the region's own side
# running along the tissue edge, and how far past the flap the closing edge may look for its tangent. The cap
# is a wide plateau (700 and 1000 are identical; 400 blocks YW113_RH_1-1-10's 672 um flap). The margin is a
# HILL: 0.515 / 0.519 / 0.520 / 0.521 / 0.523 / 0.526 / 0.525 / 0.508 at 0 / 100 / 200 / 300 / 400 / 500 /
# 650 / 800 um. A top-of-hill constant is the kind that does not transfer, so it was held out by bird in
# `diag_taper_margin.py`: all seven birds prefer 500 on the other six, and the held-out mean is 0.526 -- the
# same as in-sample. A multiple of the flap's own length, which would be scale-free, is worse at every
# multiple tried (x1.5 0.519, x2 0.520, x3 0.508).
FLAP_CAP_UM = 700.0
MARGIN_UM = 500.0

# OR AS A MULTIPLE OF THE FLAP'S OWN LENGTH, which is scale-free where `MARGIN_UM` is a knife edge:
# the absolute version rises to 0.526 at 500 um and falls off a cliff by 800. 0 uses `MARGIN_UM`.
TIMES = 0.0

# HOW WIDE A DIP IN THE BORDER may be and still be bridged with a straight chord, so the region stays convex.
# Alice, 2026-08-19, on OR99_RH_1-1-4: *"can you do something about the concave dips"*. 0 leaves every dip;
# 9999 bridges every one, which is what SHIPS -- width turned out not to separate the good bridges from the
# bad ones (her dip there is 944 um wide, so every limit tried left the section she asked about untouched).
# What separates them is whether the chord stays inside the black: `CROSS_SHARE` is the fraction of the whole
# dip's walk, from each ray's old stop out to its new one, that has to be black before the dip may be bridged.
# 0 disables the test. Unlike the taper's margin this is a real PLATEAU -- 0.45 / 0.5 / 0.55 / 0.6 all score
# 0.530, against 0.527 at 0.4 and 0.525 at 0.7 -- and it transfers: held out by bird in `diag_dip_share.py`,
# six of seven birds prefer 0.55, LBlu59 prefers 0.45, and the held-out mean is 0.530, the same as in-sample.
# 0.5 is the middle of the plateau and says something sayable: the bridge has to be more black than not.
DIP_UM = 9999.0
CROSS_SHARE = 0.5

# A THIRD, SHALLOWER DARK CUE, for the hippocampus border where no black separation marks it. Alice,
# 2026-08-19, on Purp30_LH_1-1-8: *"i feel like you should be able to identify the hippocampus even though
# there isn't a black separation there is a line that is darker relative to the surroundings"*, then *"even if
# the cue is only present half the time can't you just connect it or just prevent the cmm region from crossing
# that area"*. `diag_hp_dip.py` measures her line there at 19% below its own surround -- and the shipped
# relative cue is RATIO = 0.8, a 20% dip, so her line falls a hair short of it. The floor is what keeps a
# shallower cue specific: her line is continuous over millimetres where speckle and cell gaps are tens of
# microns ([[thin-black-strips-are-her-border]], [[minimum-line-length-beats-the-step]]). 0 = off.
# CONNECTING IT is already in this path: `extend` continues a strip along its own direction and `reach`
# follows it to where its black closes. Grouping nearby stubs instead is measured and catastrophic
# (`long_strips(rejoin=True)`, mean IoU 0.832 -> 0.020, every section) because speckle is dense.
SHALLOW_RATIO = 0.0
SHALLOW_UM = 1200.0

# MEASURED: A SHALLOWER THRESHOLD IS NOT THE ANSWER. 0.85 / 0.88 at floors of 1200 and 2000 um all leave
# Purp30_LH_1-1-8 at exactly 0.495, because the threshold was never what failed: 46% of her HP line's pixels
# there are ALREADY below the shipped RATIO of 0.8, against 16% of the tissue inside her CMM. Her line is
# DOTTED, so its dark pixels form pieces shorter than the 608 um floor and `long_strips` discards them as
# speckle. (The one thing it did buy: YW113_RH_1-1-3 0.325 -> 0.352 at 0.88 / 1200 um.)
#
# SO COUNT DENSITY INSTEAD OF DEMANDING ONE UNBROKEN PIECE. Alice: *"even if the cue is only present half the
# time can't you just connect it or just prevent the cmm region from crossing that area"*. `DENSE_SHARE` is
# how much of an 80 um window has to be relatively dark for that window to count as a separation. A dotted
# line is dense; speckle is not. Nothing new is thresholded on brightness -- the cue is the shipped RATIO,
# aggregated differently.
#
# MEASURED: FEEDING IT IN AS FLAT BLACK IS FAR TOO GREEDY. `bright & ~dense` makes every dense-dark
# neighbourhood a wall, so the rays stop at the first CLUSTER OF CELLS: mean IoU 0.530 -> 0.302 at share 0.5,
# cells kept 88% -> 41%, and the sections where the tissue inside her CMM is itself 43-47% dark
# (Wh175_LH_1-1-7, OR99_RH_1-1-4) collapse to 0.118 and 0.000. It did lift the target section, Purp30_LH_1-1-8
# 0.495 -> 0.598, which is what says the cue is real and only its SELECTIVITY is missing.
#
# SO THE SMEARED MASK GOES THROUGH THE STRIP MACHINERY INSTEAD, as a third `strip_bright` cue with its own
# length floor `DENSE_LENGTH_UM`. Smearing is what lets a length test work at all here -- it joins the dashes
# into one piece, which is the part that was impossible before ([[thin-black-strips-are-her-border]]) -- and
# `long_strips` then keeps a thin LONG dense line while discarding a dense blob of cells as a wide gap plus
# its rind. 0 = off.
DENSE_SHARE = 0.0
DENSE_UM = 80.0
DENSE_LENGTH_UM = 608.0

# HOW WIDE A BLACK RUN HAS TO BE before `THROUGH` crosses it instead of stopping at it. 0 crosses every one.
WIDE_UM = 0.0

CHOP_GAP_UM = 40.0

# Whether to keep only the largest connected piece. Alice, 2026-08-19: *"ok can you remove that little
# extra piece"* -- on YW113 1-1-7 a detached speck of region sat past the border as its own closed loop.
# She has asked for one patch outright: *"can you make sure its a continuous patch"*.
ONE_PIECE = True


def run_members(inside, run_um):
    """Which rays belong to a stretch of at least `run_um` consecutive True values in `inside`."""
    needed = max(int(round(run_um / ALONG_BIN_UM)), 1)
    keep = np.zeros_like(inside)
    run = 0

    for ray in range(len(inside) + 1):
        if ray < len(inside) and inside[ray]:
            run += 1
            continue

        if run >= needed:
            keep[ray - run:ray] = True

        run = 0

    return keep


def run_shift(line, stop, usable, run_um):
    """How far inward to slide a fitted line, in ray steps, so it sits behind a RUN of agreeing walls.

    Negative, or zero when no run exists. The median over the run's members rather than the deepest wall in
    it, for the same reason a running median beats a clamp on the rays themselves
    ([[median-deletes-spikes-clamp-cannot]]): one unusually deep wall inside an otherwise agreeing stretch
    should not set the border for the whole section.
    """
    members = run_members(usable & (stop < line), run_um)

    if not members.any():
        return 0.0

    return float(np.median(stop[members] - line[members]))


def straighten(stop, found, width, segments, inward=None, slide_um=None):
    """Replace the wiggly per-ray far border with `segments` straight lines. No new fitted constant.

    THE FIT USES ONLY THE RAYS THAT FOUND BLACK, and that choice is the whole idea rather than a detail.
    A ray with a wall is already within 8-48 um of her border ([[far-border-is-solved-where-the-gap-is]]);
    a ray with no wall has no answer at all and today copies its neighbours, which is what lets it run to
    the tissue edge and makes the region 19% too big ([[ncm-rays-transfer-to-cmm-halfway]]). Fitting the
    line to the found rays and then imposing it on ALL of them straightens the border and pulls the blind
    rays in at the same time, from evidence that is already known to be good.

    A least-squares line over EVERY ray would do the opposite: the overshooting rays are the ones furthest
    out, so they would drag the line out with them.

    `segments` splits the rays into equal blocks and fits one line per block, so 1 is a flat far border --
    the trapezoid's shape, but positioned by where the black actually is instead of by a fitted depth.
    Blocks with too few found rays keep their own stops rather than inventing a line from two points.
    """
    if segments <= 0:
        return stop

    rays_index = np.arange(len(stop), dtype=float)
    usable = found & (stop < width)
    out = stop.astype(float).copy()

    for block in range(segments):
        edges = np.linspace(0.0, float(len(stop)), segments + 1)
        piece = (rays_index >= edges[block]) & (rays_index < edges[block + 1])
        take = piece & usable

        # THREE, not two: two points define a line exactly and so cannot disagree with it, which would let
        # a pair of unlucky rays set the border for a whole block with no way to notice.
        if take.sum() < 3:
            continue

        slope, intercept = np.polyfit(rays_index[take], stop[take], 1)
        line = slope * rays_index[piece] + intercept

        # SLIDING THE WHOLE LINE INWARD, as an alternative to clipping rays one at a time. Alice,
        # 2026-08-19: *"or99 1-1-4 needs some straightening because i don't know what's going on with
        # that"* -- and she is right: `CLIP_TO_WALL` fixed that section's overshoot by pulling each found
        # ray back to its own wall, which on a section whose walls sit at wildly different depths rebuilds
        # the staircase the straightening had just removed (OR99_RH_1-1-4, 0.58 -> 0.51).
        #
        # A least-squares line runs through the MIDDLE of the walls, so half of them are inside it and
        # those rays cross black. Shifting the line to the `inward` percentile of its own residuals leaves
        # only that fraction of found rays outside it -- one straight border that also mostly obeys the
        # walls, instead of a straight border plus a per-ray correction that fights it.
        # 0 MEANS OFF, not "behind every wall". The 0th percentile is the single innermost wall in the block,
        # which is a spike by definition, so no caller would ever want it and it is free to be the off
        # switch -- the same convention `slide_um` below uses.
        if inward:
            residual = stop[take] - (slope * rays_index[take] + intercept)
            line = line + float(np.percentile(residual, inward))

        # AND THE SAME MOVE DRIVEN BY THE IMAGE RATHER THAN BY A PERCENTILE. `inward` slides the line by a
        # fixed fraction of its own residuals whatever the walls look like; this slides it only as far as a
        # stretch of walls that AGREE with each other asks for, and not at all where they do not.
        if slide_um:
            line = line + run_shift(line, stop[piece], usable[piece], slide_um)

        out[piece] = line

    return np.clip(out, 0.0, float(width))


def clip_runs(straight, stop, found, run_um):
    """Clip the straightened border back to its own walls, but only along RUNS of agreeing rays.

    THE SAME CLIP HELPS ONE SECTION AND WRECKS ANOTHER, which is what makes the run test necessary. On
    YW113 1-1-7 a long black band curves inward and clipping every found ray to it gains 0.05 IoU; on
    OR99_RH_1-1-4 a bright vessel scatters walls at unrelated depths and the identical clip LOSES 0.05,
    turning a straight border into a staircase -- Alice: *"or99 1-1-4 needs some straightening because i
    don't know what's going on with that"*.

    A lone ray whose wall sits inside the line is noise. `run_um` consecutive rays that all agree is a
    border, and the same reasoning as [[border-cue-is-intermittent-not-absent]]: look for runs, not for
    single firings. Rays outside every run keep the straight line, so the border stays straight everywhere
    the evidence is not consistent.
    """
    inside = found & (stop < straight)
    needed = max(int(round(run_um / ALONG_BIN_UM)), 1)
    keep = np.zeros_like(inside)
    run = 0

    for ray in range(len(inside) + 1):
        if ray < len(inside) and inside[ray]:
            run += 1
            continue

        if run >= needed:
            keep[ray - run:ray] = True

        run = 0

    return np.where(keep, np.minimum(straight, stop), straight)


def smooth_walls(stop, found, span_um):
    """The wall depth at every ray, as a median over the found walls within `span_um` of it.

    THIS IS THE "BOTH" Alice asked for: *"the other one was better, but why can't you do both"*. Clipping
    each ray to its OWN wall is what makes a staircase, because neighbouring rays are pulled back by
    unrelated amounts. Clipping instead to a median over a stretch of neighbours keeps whatever the walls
    agree about and discards whatever they do not -- so a black band that genuinely curves along the border
    still pulls the border in and follows it, while a bright vessel that throws walls to random depths
    averages back out to roughly where the fitted line already was. A median over a window cannot make a
    rectangular notch; the narrowest feature it can draw is `span_um` wide.

    `inf` where fewer than 3 found rays are in range, which is the "no evidence, do not clip" answer rather
    than a clip against a guess -- the same three-point rule `straighten` uses.
    """
    half = max(int(round(span_um / ALONG_BIN_UM / 2.0)), 1)
    out = np.full(len(stop), np.inf)

    for ray in range(len(stop)):
        low, high = max(ray - half, 0), min(ray + half + 1, len(stop))
        take = found[low:high]

        if take.sum() >= 3:
            out[ray] = float(np.median(stop[low:high][take]))

    return out


def wall_middle(stop, found, black, share, wide_um=0.0):
    """Move each found wall `share` of the way THROUGH its own black run. 0 is first contact, 0.5 the middle.

    WHY THE NEAR EDGE IS THE WRONG PLACE TO STOP ON A WIDE BLACK PATCH. `look_or99_walls.py` on
    OR99_RH_1-1-4: the black there is not a thin line but a broad ragged patch, and her border is a straight
    line THROUGH it while the clip traces its frayed near edge -- and a frayed edge followed ray by ray is
    exactly the staircase Alice objected to. Entering the run by a fixed share of its own length turns a
    ragged edge into a much smoother one, because the ragged part is the edge and not the run's middle, and
    it costs nothing where the black really is a thin strip: on a 40 um strip the middle is 20 um away.

    `share` is one constant for every section, not a per-section fit, so this cannot buy a section by
    tuning to it ([[knob-count-is-a-modelling-choice]]).

    `wide_um` restricts the move to walls whose black run is at least that long, which is the only version
    of this that could ever be right on every section: a THIN strip is her border and the ray should stop at
    it ([[thin-black-strips-are-her-border]]), while a WIDE ragged patch is a structure she draws across.
    """
    limit = max(int(round(wide_um / ACROSS_BIN_UM)), 0)
    out = stop.astype(float).copy()

    for ray in np.nonzero(found)[0]:
        step, run = int(min(stop[ray], black.shape[1] - 1)), 0

        while step + run < black.shape[1] and black[ray, step + run]:
            run += 1

        if run >= limit:
            out[ray] = stop[ray] + share * run

    return out


def wall_is_interior(walls, found, black, reach_um, share):
    """Whether each ray's wall has TISSUE beyond it, rather than being the outside of the section.

    This is what tells the end of a separation from the end of the tissue. A ray that crosses a black strip
    INSIDE the region comes out into more tissue on the far side; a ray that has run past the separation's end
    stops at the section's own edge and everything beyond it is background, which reads as black too. `found`
    cannot tell them apart -- on YW113_RH_1-1-2 all 91 rays "found black" and 25 of them found the edge.
    """
    reach = max(int(round(reach_um / ACROSS_BIN_UM)), 1)
    out = np.zeros(len(walls), bool)

    for ray in range(len(walls)):
        if not found[ray]:
            continue

        start = int(min(max(walls[ray], 0.0), black.shape[1]))
        beyond = black[ray, start:start + reach]
        out[ray] = beyond.size > 0 and float((~beyond).mean()) >= share

    return out


# HER BORDER IS A DOME AND MINE IS A STAIRCASE. Alice, 2026-08-19: *"can you at least make it a little more
# dome like or no"*. Everything upstream produces one depth per ray from local evidence -- black walls, the
# tissue clip, the closing edge, the path -- and nothing makes those depths a single smooth curve. The dips are
# filled ([[bridge-dips-only-inside-the-black]]), so the border has no concave notches, but it still steps.
#
# THREE VERSIONS, because they are different claims: `smooth` is a running mean, which kills the staircase and
# keeps every real bulge; `arc` replaces the depths with one fitted quadratic, maximally smooth but unable to
# follow a real wall; `out` takes the arc only where it lies OUTSIDE the staircase, so evidence always wins.
# A fitted quadratic that opens the wrong way is a BOWL -- deeper at the ends than in the middle -- which is
# exactly the shape she ruled out (*"can you just prevent concave angles like that"*), so that case falls back
# to the straight line, and `out` is capped at where the tissue ends so an arc cannot invent region off-slide.
# OFF -- ALICE REJECTED IT BY EYE, 2026-08-19: *"i think you made it worse go back"*. It measured +0.011 held
# out by bird and she looked at the borders and said no, which settles it: her outlines are the target, and mean
# IoU is a proxy for them ([[grade-regions-on-cells-not-area]]). What the pictures show and the mean does not:
# a 960 um mean is a CHORD, so it cuts the corners off the polygon she actually draws, and on OR99_RH_1-1-1 it
# rounds away the border the path traced along her own hippocampus outline (0.501 -> 0.489). The measurements
# below stay because they are true and someone will reach for this again.
#
# Was: a 960 um running mean, LAST. Held out by bird in `diag_dome.py`: 6 of 7 birds choose 800-1120 um,
# mean +0.011 over the border before it, worst -0.008, and cells kept never moves off 88% while density rises
# 81% -> 83% and the area ratio FALLS 1.09 -> 1.07, so it tightens onto her region instead of inflating.
# 960 is the interior of a flat 800-1120 plateau rather than an edge of it. 1920 um collapses (cells 73%): a
# mean that wide is a chord, not a dome. `DOME_FIRST` measured 0.541 against 0.544 and cost OR99_RH_1-1-4
# 0.578 -> 0.535, so the path bulge is smoothed rather than protected -- OR99_RH_1-1-1 pays 0.501 -> 0.489 for
# the other fourteen.
DOME = False
DOME_FIRST = False
DOME_SPAN_UM = 960.0
# THE SAG: how far the middle of the border should sit outside the straight line through its ends. Her own far
# border bulges on 10 of 10 measurable sections, median 77 um, range 11-146 (`diag_dome_sag.py`) -- while the
# 960 um mean leaves mine at 37 um, less than half. So the dome is a missing CONSTANT, not a missing cue: the
# mean makes the border smooth and a wide mean makes it a CHORD, which is why smoothing gained by straightening.
#
# MEASURED AND OFF, and the reason is worth keeping. Imposing her own median sag makes it monotonically worse --
# 0.544 flat, 0.541 at 40 um, 0.537 at 80, 0.527 at 120, 0.489 at 240 -- with cells kept pinned at 88% the
# whole way while the area ratio climbs 1.07 -> 1.24 and density falls 83% -> 71%. The bulge adds AREA and no
# cells, because my border is already 7% too big: pushing its middle further out pushes it into background, not
# into her region. Two sections do want it (OR99_RH_1-1-4 0.574 -> 0.653 at 120 um, OR408_RH_1-2-1 rising to
# 0.529 at 240) but nothing about them transfers, and every other section pays. Her dome is also SMALL -- 77 um
# on a border ~700 um deep -- so it was never going to be the visible change it looks like in her outlines.
DOME_SAG_UM = 0.0


def dome_border(border, table, mode=None, span_um=None, found=None, sag_um=None):
    """CMM's far border as one smooth curve across the rays instead of a per-ray staircase.

    `smooth` SHIPS and is the default. `arc` (0.535) and `out` (0.526) are kept because they are the two
    versions people reach for first, and both lose to a running mean.
    """
    from diag_dip_path import tissue_limit

    want = "smooth" if mode is None else mode
    position = np.arange(len(border), dtype=float)

    if want == "smooth":
        size = max(int(round((DOME_SPAN_UM if span_um is None else span_um) / ALONG_BIN_UM)), 1)
        padded = np.pad(border.astype(float), size, mode="edge")
        smooth = np.convolve(padded, np.ones(size) / size, mode="same")[size:-size]
        want_sag = DOME_SAG_UM if sag_um is None else sag_um

        if not want_sag:
            return smooth

        # THE DEFICIT ONLY, so a border that already bulges enough is left alone: the hump adds the difference
        # between the sag it has and the sag she draws, and it is a parabola over the whole span of rays
        # because CMM's rays ARE its along-extent -- 1 in the middle, 0 at both ends, so the ends do not move.
        middle = (len(smooth) - 1) / 2.0
        hump = 1.0 - ((position - middle) / max(middle, 1.0)) ** 2
        curve = np.polyval(np.polyfit(position, smooth, 2), position)
        line = np.polyval(np.polyfit(position, smooth, 1), position)
        add = max(want_sag / ACROSS_BIN_UM - float(np.max(curve - line)), 0.0)

        return np.minimum(smooth + add * hump, tissue_limit(table["mean"], table["seen"], run_um=200.0))

    # FITTED ON THE RAYS WITH EVIDENCE where there are any: a blind ray sits wherever the tissue edge left it,
    # and letting those drag the arc outward is how a smooth border turns back into the overshoot.
    keep = np.ones(len(border), bool) if found is None or not found.any() else found
    curve = np.polyfit(position[keep], border[keep].astype(float), 2)
    arc = np.polyval(curve, position)

    if curve[0] > 0:
        arc = np.polyval(np.polyfit(position[keep], border[keep].astype(float), 1), position)

    if want != "out":
        return arc

    return np.maximum(border, np.minimum(arc, tissue_limit(table["mean"], table["seen"], run_um=200.0)))


# THE FAR BORDER WHERE NO BLACK MARKS IT, which is what CMM's rays have always been missing (task 14): the
# OUTERMOST relatively-dark line, chosen as one continuous path across the rays.
#
# Alice, 2026-08-19: *"even if the cue is only present half the time can't you just connect it or just prevent
# the cmm region from crossing that area"*, then *"its not density though its relative darkness?"*. Both
# corrections are in this rule. Density is not it -- thresholding the dark test and counting pixels in a window
# gives the dim HALF of the tissue, not a line ([[connecting-the-hp-cue-cuts-inside-her-line]]). Relative
# darkness taken CONTINUOUSLY is the cue, and connecting it is a path and not a mask
# ([[path-decode-lands-within-112um]], [[continuous-path-finds-cell-gaps]]).
#
# WHY `PATH_OUTER` EXISTS, and it is the whole difference between a border and a lucky cut. Her hippocampus
# line is relatively dark but NOT the darkest thing on its ray -- the 82nd percentile on Purp30_LH_1-1-8, with
# the darkest place near her line on only 30% of rays (`diag_hp_score.py`) -- so a plain path walks off onto
# whatever is darkest and lands 200 um INSIDE her own outline. That version scores better on Purp30 (0.605
# against 0.497) and is wrong, which `look_cmm_path.py` shows and no aggregate could. `PATH_OUTER` is a
# fraction of the ray's OWN best score, so nothing is compared between birds
# ([[border-detector-threshold-does-not-transfer]]), and it is a plateau: 0.5 to 0.9 all give the same answer
# on 14 of the 15 sections.
PATH_SLOPE = 2            # steps the border may move from one ray to the next
PATH_START_UM = 300.0     # how deep the path may first sit, so it cannot land on her own Field L line
PATH_SMOOTH_UM = 96.0     # averaging ALONG the band, the one direction her line does not vary in
PATH_OUTER = 0.9          # of the ray's own best score
PATH_SLACK_UM = 200.0     # how far inward of the outermost eligible place the path may still sit
PATH = True


def path_border(section, table, border, found, slope=None, start_um=None, smooth_um=None,
                outer=None, slack_um=None):
    """The outermost continuous relatively-dark line, allowed only to pull `border` IN, and only where the
    ray has no interior wall of its own.

    The helpers are imported HERE rather than at module level: `diag_dip_path` pulls in the whole NCM rule to
    grade itself against, and a rule should not pay for a diagnostic's imports on every run.
    """
    from diag_dip_path import along_band, best_path, darkness, tissue_limit

    shape = table["black"].shape
    smooth = PATH_SMOOTH_UM if smooth_um is None else smooth_um
    score = along_band(darkness(section["green"], table, shape), table["seen"], smooth)

    steps = np.arange(shape[1])[None, :]
    limit = tissue_limit(table["mean"], table["seen"], run_um=200.0)[:, None]
    start = PATH_START_UM if start_um is None else start_um
    usable = table["seen"] & (steps >= start / ACROSS_BIN_UM) & (steps < limit)
    usable[~usable.any(axis=1)] = table["seen"][~usable.any(axis=1)]

    # OUTERMOST, NOT DARKEST. A ray with nothing eligible keeps its whole range, which is what lets the path
    # bridge the rays where the cue goes missing -- *"even if the cue is only present half the time"*.
    want_outer = PATH_OUTER if outer is None else outer

    if want_outer:
        best = np.max(np.where(usable, score, -np.inf), axis=1, keepdims=True)
        eligible = usable & (score >= want_outer * best)
        last = np.where(eligible.any(axis=1), shape[1] - 1 - eligible[:, ::-1].argmax(axis=1), -1)
        floor = (last - (PATH_SLACK_UM if slack_um is None else slack_um) / ACROSS_BIN_UM)[:, None]
        limited = usable & ((steps >= floor) | (last[:, None] < 0))
        usable = np.where(limited.any(axis=1, keepdims=True), limited, usable)

    path = best_path(score, usable, PATH_SLOPE if slope is None else slope)

    # BLACK WINS. A ray that stopped at an interior wall already has a separation, and that is the evidence
    # Alice trusts: *"its usually either a black gap or a visible line separation"*. Same predicate as the
    # closing edge ([[closing-edge-where-the-separation-ends]]), so this adds no constant of its own -- and it
    # is what rescued OR408_RH_1-2-1 from 0.457 back to 0.534 when the path was free to speak everywhere.
    interior = wall_is_interior(border, found, table["black"], REACH_BEYOND_UM, TISSUE_SHARE)

    return np.where(interior, border, np.minimum(border, path))


def dense_dark(green, ratio, window_um, share, tissue=None):
    """Where relatively-dark pixels are DENSE, whether or not they join up into one piece.

    The measurement this exists for: 46% of the pixels on her HP border on Purp30_LH_1-1-8 are already below
    `ratio`, against 16% of the tissue inside her CMM -- so the cue is there, in dots, and every length filter
    throws dots away as speckle ([[hp-line-is-real-on-half-the-sections]]). A box average is the whole
    "connect it": each pixel is scored by its NEIGHBOURHOOD, so a run of dashes with gaps between them reads
    as one dark stretch while an isolated dark cell does not.

    RESTRICTED TO THE TISSUE, and that is not a detail: the background outside the section is uniformly dark
    and therefore uniformly dense, so it merges with the tissue's own dark rim, and `long_strips`' rind
    removal then chops her border into stubs of 280 um where the tissue-masked version leaves one piece of
    1920 um ([[rind-removal-eats-real-strips]]). Density is a claim about TISSUE texture anyway -- outside
    the section there is no texture to measure -- and the background already stops rays as a wide gap.
    """
    dark = ~relative_bright(green, ratio)
    size = max(int(round(window_um / ACROSS_BIN_UM)), 1)
    dense = uniform_filter(dark.astype(np.float32), size=size) >= share

    return dense if tissue is None else (dense & tissue)


def inside_black(border, chord, span, black, found):
    """Of every step this bridge would walk over, what fraction is BLACK? 1.0 if it walks nothing.

    That fraction is the difference between cutting through a wide ragged patch (fine) and jumping a thin strip
    to land in the tissue beyond it (forbidden). It is pooled over the whole dip on purpose: a veto on any
    single bad ray blocked OR99_RH_1-1-4's own 59-ray dip, because a ragged patch has holes in it and one ray
    through a hole is not evidence that the bridge left the separation.
    """
    walked = []

    # `chord` runs alongside `span`, one value per ray in it -- it is NOT indexed by ray number.
    for ray, want in zip(span, chord):
        if not found[ray]:
            continue

        low = int(min(max(round(border[ray]), 0), black.shape[1] - 1))
        high = int(min(max(round(want), 0), black.shape[1] - 1))

        if high > low:
            walked.append(black[ray, low:high + 1])

    return float(np.concatenate(walked).mean()) if walked else 1.0


def fill_dips(border, limit_rays, black=None, found=None, share=None):
    """Bridge every DIP in the border profile narrower than `limit_rays` with the straight chord over it.

    Alice, 2026-08-19, on OR99_RH_1-1-4: *"can you do something about the concave dips"*.

    WHY A DIP IS A CONCAVE CORNER. The region is everything from the Field L line out to `border[ray]`, so it
    is convex exactly when the profile has no dips -- and she describes this region as *"a convex polygon"*.
    The clip is what puts them there: it pulls each found ray back to ITS OWN wall independently of its
    neighbours, and on OR99_RH_1-1-4 the wall is the frayed near edge of a broad ragged patch, so the border
    traces the fraying ([[straightening-is-free-not-a-gain]]).

    THE UPPER CONVEX HULL IS THE OPERATION. Its vertices are the rays the border genuinely turns on; every ray
    skipped between two hull vertices sits below the chord and is a dip. `tangent_from_end` is one facet of the
    same hull taken from the fan's end, so the closing edge and this are one idea and not two hacks.

    STAYING INSIDE THE BLACK IS THE JUDGEMENT, AND A WIDTH LIMIT IS NOT. Bridging always moves the border
    OUTWARD and can therefore cross a black separation the clip pulled back from, which she has ruled out
    repeatedly -- *"i meant whatever you did in ncm to prevent the region from crossing the black
    separations, do on the cmm region"*. The first version limited the dip's WIDTH on the theory that a narrow
    dip is fraying and a wide one is real border. **Measured, that is wrong**: OR99_RH_1-1-4's dip is 944 um
    wide and 260 um deep, so every limit tried (100 / 250 / 500 um) left the section she asked about
    completely untouched, and Wh175_RH_1-1-9's dip -- which must NOT be bridged -- is deeper still at 268 um.

    THE PICTURES SAY WHAT SEPARATES THEM (`look_or99_walls.py --dip 9999`). On OR99_RH_1-1-4 the chord runs
    down the middle of a broad ragged patch and lands on her line: it stays INSIDE the black. On
    Wh175_RH_1-1-9 and at YW113_RH_1-1-7's bottom corner the chord jumps a thin strip and lands in the bright
    tissue on its far side. So the test is per ray: walk from the old stop to the new one and require that
    walk to be mostly black. A ray with no wall is safe -- there is no separation there to cross.
    """
    if limit_rays <= 0:
        return border

    out = border.astype(float).copy()
    hull = []

    for ray in range(len(out)):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            # `second` sits at or below the chord from `first` to `ray`, so it is not a corner of the hull.
            side = (second - first) * (out[ray] - out[first]) - (out[second] - out[first]) * (ray - first)

            if side < 0:
                break

            hull.pop()

        hull.append(ray)

    for first, second in zip(hull, hull[1:]):
        if not 0 < second - first - 1 <= limit_rays:
            continue

        span = np.arange(first + 1, second)
        chord = out[first] + (out[second] - out[first]) * (span - first) / (second - first)

        # ALL OR NOTHING PER DIP: clipping the chord back on the offending rays only would put the dip
        # straight back. Either the bridge stays inside the separation, or the dip stays.
        want = CROSS_SHARE if share is None else share

        if black is not None and want and inside_black(out, chord, span, black, found) < want:
            continue

        out[span] = chord

    return out


def tangent_from_end(out, limit):
    """The straight closing edge from a zero at ray 0, run TANGENT to the border profile within `limit`.

    Returns (slope, anchor): the border is `slope * ray` out to `anchor`, and untouched beyond it. The slope is
    the smallest `out[a] / a` in range, so the edge lies at or inside the profile on every ray it covers and
    touches it at `anchor`.

    Alice, 2026-08-19, seeing the first version notch: *"can you just prevent concave angles like that"*. A
    ramp aimed at the first non-edge ray starts wherever that ray happens to sit -- on YW113_RH_1-1-2 that is
    62 steps while the separation itself is at 46, so the border jutted out and then cut back in, which is a
    concave corner in a region she draws as *"a convex polygon"*. A tangent cannot do that: no point of the
    profile is outside it, so there is nothing to cut back from.

    THE RANGE HAS TO BE SHORT. `out[a] / a` keeps falling as `a` grows -- a shallow enough line from the origin
    sits under the whole fan -- so the tangent is only meaningful over the flap and a little past it. Searching
    the whole half-fan swallowed rays that are 51-88% inside her outline.
    """
    rays_index = np.arange(1, max(min(limit, len(out) - 1), 1) + 1)
    usable = rays_index[out[rays_index] > 0.0]

    if usable.size == 0:
        return 0.0, 0

    anchor = int(usable[np.argmin(out[usable] / usable)])

    return float(out[anchor] / anchor), anchor


def taper_ends(straight, walls, found, black, share=None, reach_um=None,
               run_um=None, cap_um=None, margin_um=None, tangent=True, times=None):
    """Past the end of the identified black separation, ramp the stop down to zero at the end of the fan.

    Alice, 2026-08-19, on YW113_RH_1-1-2: *"on the right you see how it identifies the black separation and
    traces it and suddenly jumps out, in that case can you just connect the end of the identified black
    separation to the field l line rather than have that jut out"*.

    WHAT THE JUT IS AND WHICH RAYS IT IS. Grading each ray by how much of its own slice falls inside her
    outline: on that section rays 0-24 are 0-8% inside her (the flap) and rays 32-88 are 51-88% (the border
    doing its job). The flap's rays all FOUND black -- they run past the end of the little separation strip
    and stop at the TISSUE EDGE, 60-78 steps out against the strip's 46-59. So the separation does not end
    where black runs out, it ends where the wall stops having tissue behind it (`wall_is_interior`).

    WHY A RAMP TO ZERO IS THE EDGE SHE DRAWS. A stop of zero is a ray of no length, so the border passes
    through the Field L line itself at the end of the fan; ramping linearly to there draws one straight
    segment between the two. Ray index is the `along` coordinate and the stop is the `across` coordinate of an
    affine frame, so a straight line in those two is a straight line in the image. The ramp is aimed by
    `tangent_from_end` rather than at the flap's own last ray, so it cannot make a concave corner.

    IT ALSO GIVES THE ENDS THEIR TILT BACK, which this module's docstring lists as a known cost: her `end
    high` tilts one way on 15 of 15 sections and `end low` the other on 13 of 15, and a straight ray window
    squares both off.

    IT FIRES ON 9 OF 15 SECTIONS for a mean IoU of 0.526 against the clip's 0.506, with cells kept unchanged
    at 88% and area x hers 1.14 -> 1.10 -- so what it removes is empty area. No section loses.
    """
    out = straight.astype(float).copy()

    # PAST THE SEPARATION A RAY FOUND A WALL AND THAT WALL IS THE SECTION'S OUTSIDE. Both halves matter.
    # `~interior` on its own is also true of every BLIND ray, whose stop is the fan width with nothing beyond
    # it, and those are the rays whose corner she fills -- reading them as the separation's end cost
    # OR99_RH_1-1-4 0.505 -> 0.047. A blind end keeps `extend`'s answer and is left alone.
    past = found & ~wall_is_interior(walls, found, black,
                                    REACH_BEYOND_UM if reach_um is None else reach_um,
                                    TISSUE_SHARE if share is None else share)
    least = max(int(round((RUN_UM if run_um is None else run_um) / ALONG_BIN_UM)), 1)
    most = max(int(round((FLAP_CAP_UM if cap_um is None else cap_um) / ALONG_BIN_UM)), 1)
    margin = max(int(round((MARGIN_UM if margin_um is None else margin_um) / ALONG_BIN_UM)), 0)
    times = TIMES if times is None else times

    for flip in (False, True):
        order = slice(None, None, -1) if flip else slice(None)

        # THE RUN HAS TO START AT THE END OF THE FAN, and is measured over the WHOLE fan. Measuring it on an
        # array already truncated to half the fan reports every longer run as exactly half, which slips under
        # the cap below: Wh175_LH_1-2-1's 69 rays came back as 40 and got closed off anyway.
        flags = past[order]
        run = int(np.argmin(flags)) if not flags.all() else len(flags)

        # A FLAP IS SHORT. A LONG run of edge-walled rays is the region's own SIDE lying along the tissue
        # edge, not an end that overran its separation: Wh175_LH_1-2-1 runs 1104 um of a 1296 um fan and
        # closing it off cut a wedge out of her own outline, 0.491 -> 0.318. YW113_RH_1-1-2's flap, the one
        # she asked about, is 464 um.
        end = run - 1 if least <= run <= min(most, len(out) // 2) else -1

        if end < 0:
            continue

        ramp = out[order].copy()

        if tangent:
            # HOW FAR PAST THE FLAP the tangent may look, as a MULTIPLE of the flap's own length when `times`
            # is set, so the search scales with the thing it is closing rather than with a fixed distance.
            reach_rays = int(round((end + 1) * times)) if times else end + 1 + margin
            slope, anchor = tangent_from_end(ramp, reach_rays)
        else:
            anchor = end + 1
            slope = ramp[anchor] / float(anchor)

        ramp[:anchor] = slope * np.arange(anchor)
        out = ramp[order]

    return np.clip(out, 0.0, None)


def crosses_gap(row, limit):
    """Whether this ray passes THROUGH a black run of at least `limit` steps. The chop's evidence."""
    run = 0

    for black in row:
        run = run + 1 if black else 0

        if run >= limit:
            return True

    return False


def chop_ends(straight, black, gap_um=CHOP_GAP_UM):
    """Drop end rays that CROSS a black run, walking inward from each end until one lands clean.

    WHY A CROSSING AND NOT A DARK TIP. The first version of this asked whether the ray's outermost pixels
    were black and fired on zero rays in 15 sections: a ray that oversteps a separation ends up in the
    tissue BEYOND it, so its tip is bright. What distinguishes it is the run of black it went through.

    WHY INWARD FROM THE ENDS AND NOT EVERYWHERE. A dark lamina inside the region makes interior rays cross
    black too, and cutting there would punch holes -- measured to delete real cells
    ([[punching-black-out-deletes-cells]]). Chopping only inward from each end, stopping at the first clean
    ray, keeps the region one contiguous patch ([[regions-must-be-continuous-patches]]).

    Setting a stop to 0 removes that ray, so the region simply gets shorter along Field L.
    """
    limit = max(int(round(gap_um / ACROSS_BIN_UM)), 1)
    out = straight.astype(float).copy()

    for order in (range(len(out)), range(len(out) - 1, -1, -1)):
        for ray in order:
            end = int(min(max(out[ray], 1), black.shape[1]))

            if crosses_gap(black[ray, :end], limit):
                out[ray] = 0.0
                continue

            break

    return out


def one_piece(region):
    """The largest connected piece of `region`, dropping every detached speck.

    Alice, 2026-08-19: *"ok can you remove that little extra piece"*. `reachable` already walks out from
    the Field L line, but it walks along BRIGHT pixels, so a fragment that is bright-connected round the
    outside of a border comes back as its own island once the stops are straightened. This is the plain
    geometric version of the same requirement and cannot be argued with.
    """
    pieces, count = label(region)

    if count <= 1:
        return region

    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0

    return pieces == int(np.argmax(sizes))


def rays(section, constants, label_value, segments=None, clip=None, chop=None, single=None,
         inward=None, run_um=None, slide_um=None, wall_span=None, strong=False,
         record=None, through=None, wide_um=None, refit=None, taper=None, dip_um=None, dip_share=None,
         shallow=None, shallow_um=None, dense=None, dense_um=None, dense_length=None, path=None,
         dome=None, dome_first=None):
    """CMM as an interval on every ray out from its Field L line. Flags identical to NCM's shipped call.

    `constants` supplies the line (`gap`/`share`) and the two end positions, so an ORACLE row can hand
    in her own measured numbers and separate "the mechanism is wrong" from "the line is mispredicted".
    """
    green = section["green"]
    sign = -1.0 if label_value == 2 else +1.0
    bright = bright_from(green, BLACK_LEVELS[0])
    floor = K * 100.0 * dark_percent(green, 0.02, None)



    # ZERO AT HER LINE, POSITIVE INTO THE REGION. For NCM (sign +1) this is the identity, so the
    # control row really does run the same code path rather than a mirrored copy of it.
    a0 = near_border(section, constants, sign)
    across = (a0 - section["across"]) if sign < 0 else (section["across"] - a0)

    low, high = float(constants["low"]), float(constants["high"])

    # THE CUES THE RAYS STOP AT, as two lists `fill_between` walks in step: a mask of what is NOT dark, and
    # the length a piece of that darkness has to reach. A third pair is the hippocampus cue.
    strip_um = [floor, EXTRA_UM]
    strip_bright = [bright_from(green, 0.02), relative_bright(green, RATIO)]
    # THE DENSITY CUE as a third strip, so her DOTTED hippocampus border can be found by length like every
    # other separation in this pipeline.
    want_dense = DENSE_SHARE if dense is None else dense

    if want_dense:
        strip_um.append(DENSE_LENGTH_UM if dense_length is None else dense_length)
        strip_bright.append(~dense_dark(green, RATIO, DENSE_UM if dense_um is None else dense_um,
                                        want_dense, section["tissue"]))

    want_shallow = SHALLOW_RATIO if shallow is None else shallow

    if want_shallow:
        strip_um.append(SHALLOW_UM if shallow_um is None else shallow_um)
        strip_bright.append(relative_bright(green, want_shallow))

    record = {} if record is None else record
    region, stop, width = fill_between(
        bright, section["along"], across, THICKNESS_UM, window=(low, high),
        # `half_um` clips the window internally, so it has to be wide enough to contain her own ends or
        # the region is cut by a default that has nothing to do with this region.
        half_um=max(950.0, abs(low), abs(high)),
        bridge=True, strip_um=strip_um, strip_bright=strip_bright,
        lift_um=LIFT, grow=True, smooth_um=SMOOTH, smooth_early=True, smooth_inward=True, reach=True,
        angle_floor=ANGLE, extend=True, extend_rays=EXTEND_RAYS, extend_order=EXTEND_ORDER,
        values=green, return_stop=True, record=record)

    wanted = SEGMENTS if segments is None else segments

    if wanted <= 0:
        return region

    # REPAINTED from the straightened stops through the SAME table the stops were decoded in, so a step
    # index cannot land in a different frame than the one it was measured in.
    table = record["table"]

    # `strong_found` IS THE STRICT ANSWER: rays whose wall cleared the full length floor, before the shorter
    # floor rescues the blind ones ([[relative-cue-and-floor-union]], [[minimum-line-length-beats-the-step]]).
    # Worth trying as the clip's evidence on its own, since Alice's borders are the long continuous strips --
    # *"its not dotted though its continuous like its very clear"* -- so a wall that only a weaker floor
    # accepted is the more likely false one.
    found = record["strong_found"] if strong else record["found"]
    # THROUGH THE BLACK, NOT UP TO IT -- applied before the fit as well as before the clip, so the straight
    # line and the clip are both referred to the same place in the run rather than to two different edges.
    share = THROUGH if through is None else through
    walls = stop if not share else wall_middle(stop, found, table["black"], share,
                                              WIDE_UM if wide_um is None else wide_um)

    straight = straighten(walls, found, width, wanted, inward=INWARD if inward is None else inward,
                          slide_um=SLIDE_RUN_UM if slide_um is None else slide_um)

    # NEVER PAST ITS OWN WALL. Only on rays that found one -- a blind ray has no evidence to be clipped
    # against, and clipping it against `stop` would just restore the tissue-edge overshoot.
    if wall_span:
        # ON FOUND RAYS ONLY, exactly like the plain clip. A smoothed profile has a value on every ray,
        # including rays that found nothing, and letting it clip those as well is a SECOND change -- it
        # pulls the blind rays in off their neighbours' walls, which measured 81% of cells kept against 88%
        # and cost OR99_RH_1-1-4 0.505 -> 0.468. Kept separate so the smoothing is graded on its own.
        straight = np.where(found, np.minimum(straight, smooth_walls(walls, found, wall_span)), straight)
    elif run_um:
        straight = clip_runs(straight, walls, found, run_um)
    elif CLIP_TO_WALL if clip is None else clip:
        straight = np.where(found, np.minimum(straight, walls), straight)

    # STRAIGHT AGAIN AFTER THE CLIP, which is the third and last form of Alice's *"why can't you do both"*.
    # A clip notches the border; fitting one line back through the CLIPPED stops turns those notches into a
    # position -- the wall moves the whole border inward by however much it pulled, and the result is
    # guaranteed straight because it is a single least-squares line. Every ray counts here, including blind
    # ones: an unclipped ray sits exactly on the previous line, so it is evidence that the line was right
    # there, not an overshoot to be discounted.
    if REFIT if refit is None else refit:
        straight = straighten(straight, np.ones(len(straight), bool), width, wanted)

    # THE CLOSING EDGE, past the end of the separation. Against `walls`, the per-ray evidence, not against the
    # straightened line -- one straight line has no step in it by construction, so the step has to be read off
    # the thing the line was fitted to.
    # `is not False` and not a plain truth test: `taper=dict()` means "on, with the defaults", and an empty
    # dict is falsy, so a truth test reads it as OFF and quietly grades the untapered rule under its name.
    want_taper = TAPER if taper is None else taper

    if want_taper is not False:
        straight = taper_ends(straight, walls, found, table["black"],
                              **(want_taper if isinstance(want_taper, dict) else {}))

    # SMOOTHED BEFORE THE PATH, optionally. A wide running mean rounds off convex bulges, and the path's one
    # real win is a bulge -- the border it traces along her hippocampus outline on OR99_RH_1-1-1
    # ([[outermost-dark-line-not-the-darkest]]). Smoothing the staircase FIRST lets the path bulge back out of
    # it afterwards, so the two changes stop fighting.
    want_dome = DOME if dome is None else dome
    first = DOME_FIRST if dome_first is None else dome_first

    if first and want_dome is not False:
        straight = dome_border(straight, table, found=found,
                               **(want_dome if isinstance(want_dome, dict) else {}))

    # THE OUTERMOST DARK LINE, before the dips so that they see the border this produced -- and after the
    # closing edge, so a tapered end is not re-extended by a path that knows nothing about the taper.
    want_path = PATH if path is None else path

    if want_path is not False:
        straight = path_border(section, table, straight, found,
                               **(want_path if isinstance(want_path, dict) else {}))

    # NO CONCAVE DIPS. Last, so it sees the border the clip and the closing edge actually produced.
    want_dips = DIP_UM if dip_um is None else dip_um

    if want_dips:
        straight = fill_dips(straight, int(round(want_dips / ALONG_BIN_UM)), table["black"], found,
                             share=dip_share)

    # ONE SMOOTH CURVE, last of all, so it smooths the border every other stage actually produced.
    if want_dome is not False and not first:
        straight = dome_border(straight, table, found=found,
                               **(want_dome if isinstance(want_dome, dict) else {}))

    if chop:
        straight = chop_ends(straight, table["black"], gap_um=chop)
    record["final"] = straight

    region = np.zeros(bright.shape, bool)
    region[table["rows"], table["columns"]] = table["step"] < straight[table["ray"]]

    # `reach` ran inside `fill_between` on the unstraightened region, so it has to run again here or a
    # straightened border can reintroduce the sliver beyond a cleft that it was there to remove.
    region = reachable(region, bright, table)

    return one_piece(region) if (ONE_PIECE if single is None else single) else region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=2, help="2 = CMM, 1 = NCM")
    args = parser.parse_args()

    name = "CMM" if args.label == 2 else "NCM"
    saved = dict(np.load("shape_masks.npz"))
    pool = [s for s in sections() if (s["labels"] == args.label).sum() >= MIN_LABEL_PX]

    def with_hers(section, training, keys):
        """The fitted constants with `keys` replaced by her own measured ones -- one oracle at a time."""
        out = dict(fit(training, args.label))
        mine = measure_one(section, args.label)

        for key in keys:
            if mine is not None and key in mine:
                out[key] = mine[key]

        return out

    rows = [
        ("radii off the box centroid (ships)",
         dict(offsets_for=lambda s, t: solid_offsets(s, t, args.label, name,
                                                     saved.get(f"{s['stem']}|{name}"), LEASH_UM))),
        ("trapezoid off the band line",
         dict(region_for=lambda s, t: trapezoid(s, fit(t, args.label), args.label))),
        ("RAYS, raw per-ray border",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=0))),
        ("RAYS, ONE straight border, no clip",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=1, clip=False))),
        ("RAYS, straight, CLIPPED to own wall",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=1, clip=True,
                                           taper=False))),
        # WITH THE CLOSING EDGE, which is what ships. Alice, 2026-08-19: *"can you just connect the end of the
        # identified black separation to the field l line rather than have that jut out"*.
        ("RAYS, clipped + closing edge (ships)",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=1, clip=True,
                                           taper=True))),
        ("RAYS, straight at the 25th wall",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=1, clip=False,
                                           inward=25.0))),
        # AND ONE MORE STRAIGHTENING AFTER THE CLIP, the best row on mean IoU and rejected anyway -- Alice:
        # *"you need to straigh then clip since yw 113 1-1-7 goes back to whatever and none of the clipping
        # relaly does anything"*. Kept as a row because it is the one alternative that is close.
        ("RAYS, straight, clip, straight again",
         dict(region_for=lambda s, t: rays(s, fit(t, args.label), args.label, segments=1, clip=True,
                                           inward=0.0, refit=True))),
        ("RAYS, her line -- ORACLE",
         dict(region_for=lambda s, t: rays(s, with_hers(s, t, ("gap", "share")), args.label))),
        ("RAYS, her ends -- ORACLE",
         dict(region_for=lambda s, t: rays(s, with_hers(s, t, ("low", "high")), args.label))),
        ("RAYS, her line and ends -- ORACLE",
         dict(region_for=lambda s, t: rays(s, with_hers(s, t, ("gap", "share", "low", "high")),
                                           args.label))),
    ]

    print(f"\n  {name}, {len(pool)} sections, held out by bird. The rays rule is NCM's shipped call\n"
          f"  with its across axis flipped about CMM's line and its window from CMM's own ends.\n")
    print(f"  {'rule':40s} {'kept':>5s} {'area':>6s} {'dens':>6s} {'out':>5s} {'IoU':>6s}  worst section")

    for title, how in rows:
        share, ratio, density, _, _, outside, ious = tally(pool, args.label, name, **how)
        worst = min(ious, key=ious.get)
        print(f"  {title:40s} {share:4.0%} x{ratio:5.2f} {density:5.0%} "
              f"{np.mean(list(outside.values())):4.0%} {np.mean(list(ious.values())):6.3f}  "
              f"{worst[:24]:24s} {ious[worst]:.3f}")


if __name__ == "__main__":
    main()
