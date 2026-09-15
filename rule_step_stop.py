"""Stop the Field L line at a STEP in tissue colour, in either direction. One knob.

Alice, 2026-08-14: *"for the hippocampus and other tissue border that its getting stuck on is that its
not always a clear black separation, for example in yw113 rh 1-1-2, one side there is a black
separation(though it is kind of gray) that you missed and the other side its actually a lighter change
in tissue color that indicates a stop"*.

WHAT WAS STRUCTURALLY WRONG. Every stop rule so far only ever looked for tissue getting DARKER --
`rule_two_borders.dim_stop` tests `profile <= fraction * reference`, and `bright_from` tests an absolute
cutoff of 0.005 on normalised green, i.e. "as dark as the empty slide". So:

  a border where the next tissue is LIGHTER cannot stop the walk at any parameter setting
  a GREY separation is not dark enough to be black, so it reads as solid NCM

Measured on her 14 outlines, tissue colour 240 um outside each of her 28 line ends over 240 um inside:

  48% DARKER (ratio < 0.90)    40% LIGHTER (ratio > 1.10)    12% flat
  YW113_RH_1-1-2   low 0.80x (her grey separation)   high 1.54x (her lighter tissue)

She named that section and both of its ends correctly. Being blind to the lighter half of the cue is
most of why the walk lands no better than a fixed +-950 um constant.

THE FOUR CHOICES HERE, and why each is the way it is.

  TISSUE COLOUR, NOT THE LINE. `rule_two_borders.line_profile` averages a 48 um strip centred ON the
  Field L line, so it straddles the border and is half Field L's own brightness -- a bright band whose
  own end is the thing being looked for. This averages across the WHOLE region depth at each step
  instead, which is "what colour is the tissue here".

  A RATIO, NOT A DIFFERENCE. Sections differ hugely in overall brightness ([[or99-is-a-low-contrast-
  section]]), and an absolute cutoff on green is exactly what fails in both directions
  ([[tissue-threshold-is-absolute]]). Dividing by the band's own middle makes the test self-calibrating,
  which is the only reason the dim cue worked at all.

  A RUN, NOT A BIN. 80 um of consistent change. One bin is a blood vessel, and single-bin sensitivity is
  how [[lamina-cue-failed-three-ways]] ended -- a detector that fired on the gaps between bright cells.

  ONE KNOB. `--fraction` and nothing else. Every time more than one knob has been fitted on 7 birds the
  folds disagreed and the fit lost ground to the unanimous single-knob version
  ([[knob-count-is-a-modelling-choice]]).

Usage:
  python rule_step_stop.py                    # sweep the fraction, fit leave-one-bird-out
  python rule_step_stop.py --fraction 0.25    # one setting, per-section detail
  python rule_step_stop.py --hp               # also subtract her HP outline where one is drawn
"""

import argparse

import numpy as np

from find_band import NCM_LENGTH_UM
from hippocampus_check import cells_kept, hp_on_canvas, overlap
from rule_two_borders import (
    ACROSS_BIN_UM,
    ALONG_BIN_UM,
    BLACK_LEVELS,
    DIM_WIDTH_UM,
    REFERENCE_HALF_UM,
    bright_from,
    fill_between,
    load,
    relative_bright,
)

THICKNESS_UM = 80.0

# How far out the profile is built. Wider than any of her ends (max 1649 um) so the walk is never
# stopped by the edge of its own measuring window, which would masquerade as a found border.
REACH_UM = 1800.0

# How deep into NCM tissue colour is measured, in um. None means the region's full (varying) depth, and
# None IS the shipped setting: capping the depth was tested (200-800 um) and the best two-knob cell
# scored 0.778 against this one knob's 0.774, inside the noise that has flipped folds before
# ([[knob-count-is-a-modelling-choice]]). Leaving a cap in as the default silently drew a review picture
# whose ticks disagreed with the rule being reviewed, which is exactly the failure a default should not
# have.
DEPTH_UM = None

# 0.60 is here so a fold that picks the top rung is picking a peak with both sides visible rather than
# the end of the ladder: the local-jump decode peaks at 0.50 (0.40 -> 0.773, 0.50 -> 0.781, 0.60 -> 0.764).
FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
DEPTHS = (200.0, 300.0, 400.0, 600.0, 800.0, None)

# How a stepped bin becomes THE border. The measurement is identical for all four; only this sentence
# differs, and the picture says this is where the error lives, not in the cue.
#
# Alice, 2026-08-14, on Wh175_RH_1-1-9's high end: *"there is no color change"*. Measured, in ratio to
# the band's own middle: 1.30 / 1.45 / 1.62 / 1.66 walking out from 400 um INSIDE her end, then 0.79 AT
# her end. She is right about what the rule fired on -- that brightening is a cell cluster inside NCM,
# not a border -- and her border is the 0.79 dark dip 370 um further out. Wh175_LH_1-2-1 is the same
# shape: 1.03 / 1.85 / 3.91 / 5.58, so the rule quit on the leading shoulder of an enormous real cue.
#
#   "first"    the first sustained step walking OUT from the band's centre. Greedy, and every interior
#              cell cluster gets to pre-empt the real border. This is what scored 0.774.
#   "inward"   walk IN from +-REACH_UM, which is off the tissue and so certainly not NCM, and stop at
#              the outermost 80 um that MATCHES the band middle. A blob cannot pre-empt anything,
#              because the walk has already passed it. Fails where the tissue outside genuinely looks
#              the same (the 12% flat ends), which shows up as overshoot.
#   "strong"   walk out, but collect EVERY sustained run and take the one with the most total change,
#              magnitude x duration. 80 um at 1.8x loses to 600 um at 1.6x.
#   "plateau"  no fitted knob at all: run the whole ladder per section and keep the setting whose two
#              stops move least when the threshold is nudged. A stop on a real border does not care
#              about the threshold; a stop on a blob's shoulder slides. Note this is NOT the "maximise
#              total step size over both ends" idea, which is degenerate -- a lower threshold marks
#              more bins, so it would always pick the smallest fraction.
#   "jump"     a LOCAL step instead: each bin against the 240 um just inside it, and the border is the
#              BIGGEST such jump on each side. No threshold anywhere, which matters because thresholds
#              on this kind of cue do not transfer between birds
#              ([[border-detector-threshold-does-not-transfer]]).
#   "jumpfirst" the same local feature, but the first crossing of `fraction`, so it can be compared with
#              "first" on equal footing. Reaches 0.781 at 0.50 with 7/7 folds agreeing, and it is NOT an
#              improvement: see LOCAL_UM, the second knob makes the honest number 0.764.
#
# WHY A LOCAL FEATURE. Alice, 2026-08-14, on YW113_RH_1-1-7's high end: *"there arent even color
# changes"*. She is right and the rule still fires, because dividing by the band's middle measures
# distance from a reference up to 600 um away, and a slow RAMP gets there without ever having an edge in
# it. Its high side, in ratio to the band middle: 0.81 0.94 1.08 1.16 1.25 1.37 1.47 -- crossing 0.40 at
# +512 by pure accumulation, with no local change over 0.16 anywhere in it. The biggest LOCAL jump on
# that whole side is 0.34, at +880, which is her border to the bin.
#
# And the global feature cannot be repaired by tuning: measured inside her own outlines, 13 of 14
# sections vary by MORE than the 0.40 threshold internally (Wh175_LH_1-2-1 by 4.58). NCM's interior is
# more variable than the test for leaving NCM, so "first" scores 0.774 by the first crossing landing late
# often enough, not by discriminating.
#   "dip"      her own stated cue as a rule: stop at the DEEPEST local minimum on each side -- darker than
#              the brightest tissue within 160 um on BOTH sides. Alice, 2026-08-14: *"its usually either a
#              black gap or a visible line separation"*, and her 28 ends sit above the 97th percentile of
#              her own interior on exactly this measure ([[her-ends-are-dark-local-minima]]). Deepest, not
#              first, so there is no threshold to transfer; the 160 um came from that measurement rather
#              than from fitting IoU, but it is still a number chosen by hand.
#   "edge"     the OTHER half of her sentence, unconditionally: the last bin with tissue in it. "dip" alone
#              scores 0.713 -- the most accurate decode measured (median end error 52 um, 12 of 14 sections
#              average 0.789) -- but Gre595 and Wh175_LH_1-2-1 collapse to 0.235 / 0.288, and BOTH are
#              sections whose ends have no tissue outside them. A conditional version scored 0.740 and was
#              a bug: it asked whether the tissue ran out before the search window ended, which is exactly
#              False when the tissue ends AT the window edge, i.e. on both sections it existed for.
#   "dipfloor" her minimum-Field-L-length idea: the deepest dip no closer to the centre than `fraction` um.
#              Both failures stop at -16 um and +80 um, essentially at the band's centre. Inert on 10 of 14
#              sections, rescues Gre595 0.235 -> 0.814, and 0.775 at its best rung against 0.759 fitted --
#              because 2 of her 28 ends are themselves inside a 750 um floor (455 and 589 um).
#   "dipshare" the same floor PER SIDE, as a share of that side's distance to the tissue edge, because
#              along = 0 is not the middle of her span (median 139 um off, 444 um on Wh175_LH_1-2-1).
#   "dipelse"  Alice, 2026-08-15: *"can you make a rule where its like if you cant find the dip just go to
#              the tissue edge"*. The deepest dip at least `fraction` deep; if there is none that deep, the
#              tissue edge. "How deep counts" is the knob: her 28 ends run a median 0.227 against a 90th
#              percentile of 0.141 inside her outlines. Only this decode makes "no dip" a real event -- with
#              no depth requirement some bin always wins, which is why plain "dip" cannot use her fallback.
#   "dipshare-edge"  both of her ideas on ONE knob: the per-side share floor, and her fallback for when
#              nothing qualifies inside it. The fallback is free here because it replaces a fallback that
#              already existed and was worse -- the +-950 um constant, i.e. the arbitrary centred number
#              this whole decode is trying to get away from.
#   "diplength"  her minimum length, applied to the WHOLE line instead of to each half, so it never refers
#              to the band's centre. Both ends are chosen together: the deepest pair of dips at least
#              `fraction` um apart, falling back to the two tissue edges when no pair can reach.
DECODES = ("first", "inward", "strong", "plateau", "jump", "jumpfirst", "dip", "edge", "dipfloor",
           "dipshare", "dipelse", "dipshare-edge", "diplength", "anchor", "lengthelse", "lengthclean")

# The ladder each decode's one knob is chosen from, leave-one-bird-out. These two measure their knob in um
# and as a share rather than as a colour fraction, so they need their own rungs; the um range brackets her
# own half-lengths, which run from 455 um (Wh175_LH_1-2-1's high end) to 1649 um (OR408's).
LADDERS = {"dipfloor": (300.0, 450.0, 600.0, 750.0, 900.0),
           "dipshare": (0.20, 0.30, 0.40, 0.50, 0.60),
           "dipshare-edge": (0.20, 0.30, 0.40, 0.50, 0.60),
           # Her own line lengths run 1543-2602 um, so this brackets them from below: 1500 forbids nothing
           # she has drawn, 2100 forbids her three shortest.
           "diplength": (1200.0, 1500.0, 1800.0, 2100.0),
           # "anchor" places the WHOLE line, so its knob is a full length and the rungs bracket her range
           # (1543-2602 um) rather than sitting below it as diplength's minimum does.
           "anchor": (1600.0, 1800.0, 2000.0, 2200.0, 2400.0),
           # "lengthelse" pins the length at LENGTH_UM and spends its one knob on how convincing a dip has to
           # be, so the rungs are depths. Her 28 ends have a median depth of 0.227 and 12 of them are under
           # 0.10, so this range spans "accept nearly anything" to "only a clear line separation".
           "lengthelse": (0.05, 0.10, 0.15, 0.20, 0.30),
           "lengthclean": (0.05, 0.10, 0.15, 0.20, 0.30)}

# The minimum Field L line length for the length-constrained decodes, in um: 0.90 x the shortest line she has
# drawn (1543 um). Alice, 2026-08-15: *"make the minimum length a little shorter than my shortest length"*.
# Read off her outlines rather than fitted to IoU, which is what lets "lengthelse" spend its only knob on the
# depth ([[minimum-line-length-beats-the-step]]).
LENGTH_UM = 1389.0

# Half-width of the shoulders for the local-minimum test, in um.
DIP_REACH_UM = 160.0

# The shipped absolute floor on a dip's depth: the one knob of the "lengthelse" decode, fitted leave-one-bird-out
# ([[minimum-line-length-beats-the-step]]). Named here because `outer_stops` spends its own knob on the SHARE and
# still needs this value for its fallback, and a second copy of it would drift.
DEPTH_FLOOR = 0.227

# The window a bin is compared against for the local jump, in um. 240 um is 15 bins: wide enough that one
# blood vessel cannot set the baseline, narrow enough that a gradient does not accumulate across it.
#
# AND IT IS A SPIKE, NOT A SETTING. Fitting the fraction leave-one-bird-out at each width:
#
#   120 um 0.755   160 um 0.751   240 um 0.781   320 um 0.757   400 um 0.725   560 um 0.755
#
# 240 beats both of its neighbours by 0.024 and is the only width where all 7 folds agree, which is what
# a noise spike looks like, not what a real scale looks like. Fitting BOTH knobs leave-one-bird-out --
# the honest number for this whole family -- gives 0.764, BELOW the one-knob "first" at 0.774, and 6 of
# the 7 folds then choose 320 um, which scores 0.757 held out. So "jumpfirst 0.50" is not an improvement,
# it is this file's own [[knob-count-is-a-modelling-choice]] lesson happening again.
LOCAL_UM = 240.0

# Decodes with nothing to fit, so their score on all 14 sections IS their honest score -- no leave-one-
# bird-out needed, and no way to overfit 7 birds ([[knob-count-is-a-modelling-choice]]).
KNOBLESS = ("plateau", "jump", "dip", "edge")

# NaN means no tissue, which is a departure of unknown size. Counting it as 1.0 (a doubling) makes it a
# strong vote without letting it swamp every finite run by being infinite.
NAN_DEPARTURE = 1.0

# How far two stops may differ and still count as the same stop, for "plateau", in um. Five bins.
PLATEAU_TOL_UM = 80.0


def deep_profile(along, region, green, across=None, depth_um=None):
    """Mean normalised green inside `region` at each along bin: (profile, positions).

    Bins no region pixel landed in are NaN, which is the honest value: there is no tissue there to have
    a colour. NaN then stops the walk for free, so the tissue edge needs no separate test -- the same
    trick that let one rule cover both of her borders in rule_two_borders.

    `depth_um` CAPS how deep into the region the average reaches, and the picture says it matters. With
    no cap the average runs the region's full depth -- but that depth CHANGES along the band, so the
    average shifts wherever the band gets deeper or shallower even though tissue colour is constant. A
    depth change then reads as a colour change and stops the walk early, which is visibly what happens
    at the high end of Wh175_RH_1-1-9, Wh175_LH_1-2-1 and Gre595: the cut slices across the middle of
    her outline. A fixed depth compares like with like.
    """
    inside = np.isfinite(along) & region & (np.abs(along) <= REACH_UM)

    if depth_um is not None and across is not None:
        inside = inside & (across >= 0) & (across <= depth_um)

    if not inside.any():
        return None

    index = np.rint(along[inside] / ALONG_BIN_UM).astype(np.int64)
    low = int(index.min())
    index = index - low
    count = np.bincount(index)
    total = np.bincount(index, weights=green[inside].astype(float))

    profile = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    positions = (np.arange(len(profile)) + low) * ALONG_BIN_UM

    return profile, positions


def departure_of(profile, positions):
    """|colour / the band's own middle - 1| per bin, with NaN kept as NAN_DEPARTURE. None if unusable."""
    middle = np.abs(positions) <= REFERENCE_HALF_UM
    reference = float(np.nanmedian(profile[middle])) if middle.any() else float("nan")

    if not np.isfinite(reference) or reference <= 0:
        return None

    ratio = profile / reference

    return np.where(np.isfinite(ratio), np.abs(ratio - 1.0), NAN_DEPARTURE)


def scan_order(length, centre, direction, inward):
    """The bin indices to visit, in order. Four cases, and getting them mixed up scans the wrong end.

    outward is centre -> the reach edge, which is what "the first step" means; inward is the reach edge
    -> centre, which is what "the outermost tissue that still looks like NCM" means.
    """
    if direction > 0:
        return list(range(length - 1, centre - 1, -1)) if inward else list(range(centre, length))

    return list(range(0, centre + 1)) if inward else list(range(centre, -1, -1))


def runs_of(flag, order, needed):
    """Maximal runs of True in `flag` at least `needed` long: (leading index, length, indices).

    "Leading" means first in SCAN order, not in canvas order. Scanning outward that is the run's inner
    end -- where the change begins, which is the border. Scanning inward it is the run's outer end,
    which is the border for the opposite reason. Either way it is the index a stop reports.
    """
    found, at = [], 0

    while at < len(order):
        if not flag[order[at]]:
            at += 1
            continue

        end = at

        while end < len(order) and flag[order[end]]:
            end += 1

        if end - at >= needed:
            found.append((order[at], end - at, [order[k] for k in range(at, end)]))

        at = end

    return found


def jump_stop(profile, positions, direction, fraction=None):
    """Where the tissue colour JUMPS relative to the 240 um just inside it, walking outward.

    `fraction` None takes the biggest jump on this side, which has no knob to fit at all; a number takes
    the first jump over it, which is the same decode as "first" on a different feature.

    Bins with no tissue are skipped rather than counted as an infinite jump: the tissue edge would
    otherwise win every argmax, and the region mask already clips there anyway.
    """
    span = max(int(round(LOCAL_UM / ALONG_BIN_UM)), 1)
    centre = int(np.argmin(np.abs(positions)))
    order = scan_order(len(profile), centre, direction, False)
    best, best_at = -np.inf, None

    for step, at in enumerate(order):
        if step < span or not np.isfinite(profile[at]):
            continue

        inside = profile[[order[k] for k in range(step - span, step)]]
        base = float(np.nanmedian(inside)) if np.isfinite(inside).any() else float("nan")

        if not np.isfinite(base) or base <= 0:
            continue

        jump = abs(profile[at] / base - 1.0)

        if fraction is not None:
            if jump > fraction:
                return float(positions[at])
        elif jump > best:
            best, best_at = jump, at

    return None if best_at is None else float(positions[best_at])


def dip_depth(profile, bin_um=ALONG_BIN_UM):
    """Per bin: how much darker than the brightest tissue within DIP_REACH_UM on BOTH sides. NaN if unknown.

    A FRACTION, and that is a real bias: dividing by the local brightest EXPLODES wherever the tissue is dim,
    so Gre595 (whose whole span peaks at 0.206) and Wh175_LH_1-2-1 (a 9x ramp along the line) both find their
    deepest "dip" near the band's centre on drops of 0.03-0.04, where a real line separation elsewhere drops
    0.10 ([[her-ends-are-dark-local-minima]]). The raw drop was the obvious alternative and is not better --
    it has the mirror-image bias, preferring the bright half of a section -- so the fix was not a better depth
    but a constraint on the two ends TOGETHER ([[minimum-line-length-beats-the-step]]).

    Taking the MIN of the two shoulders is what makes this specific to a line rather than to any change:
    tissue simply getting darker outward, which every earlier rule hunted, scores about 0 here. Kept as its
    own function so a picture of the cue reads the same numbers the rule stops on, rather than a
    re-derivation that could quietly disagree with it.

    `bin_um` is the spacing of `profile`, so the SAME cue can be run down a ray ACROSS the band (bins of
    ACROSS_BIN_UM = 8 um) instead of along it. One implementation for both directions on purpose: a
    second copy of this loop for the far border would drift from this one the first time either changed,
    and the whole point of comparing the two directions is that the cue is identical.
    """
    reach = max(int(round(DIP_REACH_UM / bin_um)), 1)
    out = np.full(len(profile), np.nan)

    for at in range(len(profile)):
        if not np.isfinite(profile[at]):
            continue

        inside = profile[max(at - reach, 0):at]
        outside = profile[at + 1:at + 1 + reach]

        if not (np.isfinite(inside).any() and np.isfinite(outside).any()):
            continue

        shoulder = min(np.nanmax(inside), np.nanmax(outside))

        if np.isfinite(shoulder) and shoulder > 0:
            out[at] = 1.0 - profile[at] / shoulder

    return out


def dip_stop(profile, positions, direction, prefer_edge=False, floor_um=None, floor_share=None,
             min_depth=None, else_edge=False):
    """The deepest dark local minimum on this side, walking out from the centre and stopping at any gap.

    `prefer_edge` returns the tissue edge instead: the last bin with tissue in it. Alice, 2026-08-15: *"for
    those two there is no outer tissue so it should just go to the edge, and i dont think you are making
    that distinction right now"* -- and she was right that the distinction was missing, because the earlier
    version asked "did the tissue run out BEFORE my search window ended", which is False exactly when the
    tissue ends at the window's edge. That is the situation on both sections it was written for, so it never
    fired on either. The honest reading of "the edge" turns out to be unconditional, and unconditional is a
    different rule: her end is at the tissue edge on only 5 of 28 ends, and the edge sits a median +267 um
    PAST her end on the other 23 ([[hand-outlines-already-sit-on-the-tissue-edge]] measured this ACROSS the
    band, where it does hold; along the band it does not).

    `floor_um` is her other suggestion, a minimum Field L half-length: no stop closer to the centre than
    this. Both failures stop essentially AT the centre (-16 um, +80 um), so a floor removes them without
    touching a section that already stops past it. It is a fitted knob, so it goes on the ladder and gets
    chosen leave-one-bird-out like any other ([[knob-count-is-a-modelling-choice]]).

    `floor_share` is the same idea measured PER SIDE, as a share of the distance to the tissue edge on that
    side. Alice, 2026-08-15, on seeing the floor drawn: *"the gray is not in the right place which is why
    the orange is getting messed up"* -- and measurement agrees: along = 0 sits a median 139 um from the
    middle of her span and 444 um out on Wh175_LH_1-2-1, the section a symmetric floor hurts most. A
    symmetric floor around a mis-centred origin is tight on one side and slack on the other. A share is
    also scale-free, so it transfers between sections of different size in a way that 750 um cannot.
    """
    depth = dip_depth(profile)
    centre = int(np.argmin(np.abs(positions)))
    inside, last_tissue = [], None

    for at in scan_order(len(profile), centre, direction, False):
        if not np.isfinite(profile[at]):
            # Tissue has run out. The first time it does is the edge she would have drawn to; anything
            # past it is across a gap, so stop looking either way.
            if last_tissue is not None:
                break

            continue

        last_tissue = at
        inside.append(at)

    if prefer_edge:
        return None if last_tissue is None else float(positions[last_tissue])

    # The share is of THIS side's tissue extent, which is why the edge has to be known before the floor
    # can be applied -- hence the two passes.
    floor = floor_um

    if floor_share is not None:
        floor = None if last_tissue is None else floor_share * abs(float(positions[last_tissue]))

    best, best_at = -np.inf, None

    for at in inside:
        if floor is not None and abs(float(positions[at])) < floor:
            continue

        if min_depth is not None and not (np.isfinite(depth[at]) and depth[at] >= min_depth):
            continue

        if np.isfinite(depth[at]) and depth[at] > best:
            best, best_at = depth[at], at

    if best_at is None and else_edge and last_tissue is not None:
        return float(positions[last_tissue])

    return None if best_at is None else float(positions[best_at])


def step_stop(profile, positions, fraction, direction, decode="first"):
    """The um at which the colour step that marks this end of the band begins.

    A step is `DIM_WIDTH_UM` of bins whose colour differs from the band's own middle by more than
    `fraction`, EITHER WAY -- the whole point of this file. NaN counts as a step, so running out of
    tissue is one case of the same rule rather than a special one. `decode` chooses WHICH such step is
    the border; see DECODES.

    Returns None when nothing qualifies, meaning "no stop found, the constant stands".
    """
    if decode == "jump":
        return jump_stop(profile, positions, direction)

    if decode == "jumpfirst":
        return jump_stop(profile, positions, direction, fraction)

    if decode == "edge":
        return dip_stop(profile, positions, direction, prefer_edge=True)

    if decode == "dipfloor":
        return dip_stop(profile, positions, direction, floor_um=fraction)

    if decode == "dipshare":
        return dip_stop(profile, positions, direction, floor_share=fraction)

    if decode == "dipelse":
        return dip_stop(profile, positions, direction, min_depth=fraction, else_edge=True)

    if decode == "dipshare-edge":
        return dip_stop(profile, positions, direction, floor_share=fraction, else_edge=True)

    if decode == "dip":
        return dip_stop(profile, positions, direction)

    departure = departure_of(profile, positions)

    if departure is None:
        return None

    # Start the scan at the band's middle and walk outward, so a step near along = 0 -- inside Field L's
    # own bright patch -- is the first thing found rather than something skipped over.
    centre = int(np.argmin(np.abs(positions)))
    needed = max(int(round(DIM_WIDTH_UM / ALONG_BIN_UM)), 1)

    if decode == "inward":
        # Matching, not stepped, and scanned from the reach edge in, so the FIRST run found is the
        # outermost stretch of tissue that still looks like NCM.
        found = runs_of(departure <= fraction, scan_order(len(departure), centre, direction, True), needed)

        return float(positions[found[0][0]]) if found else None

    found = runs_of(departure > fraction, scan_order(len(departure), centre, direction, False), needed)

    if not found:
        return None

    if decode == "strong":
        # Magnitude x duration: a wide moderate change beats a narrow violent one, which is the
        # difference between a border and a cell cluster.
        best = max(found, key=lambda run: float(np.sum(departure[run[2]])))
        return float(positions[best[0]])

    return float(positions[found[0][0]])


def dip_candidates(profile, positions, direction, guard_um=None):
    """Every dark local minimum on this side as (position, depth), plus that side's tissue edge.

    Walking outward and breaking at the first gap past tissue, so nothing across a separation is offered.

    `guard_um` drops candidates that close to the tissue edge. Alice, 2026-08-15: *"in or99 1-1-1 and or408
    1-2-1, the hippocampus/dip isn't identified correctly"*. Tracing OR99_RH's high side showed the rule
    stopping at +1344 on a "dip" of 0.254 where the profile reads 0.034 -- the tissue FADING OUT at its
    boundary, not a line between two pieces of tissue. The dip is a ratio, so an outer shoulder of 0.046
    manufactures a deep dip out of the blur at the edge. Within one shoulder-width of the edge the outer
    shoulder IS the edge, so the measurement is not meaningful there; the honest answer for that side is the
    tissue edge itself, which `length_stops` already falls back to. Excluding it, OR99_RH's deepest real dip is
    +1024 (depth 0.201), 64 um from her end instead of 389.
    """
    depth = dip_depth(profile)
    centre = int(np.argmin(np.abs(positions)))
    found, last_tissue = [], None

    for at in scan_order(len(profile), centre, direction, False):
        if not np.isfinite(profile[at]):
            if last_tissue is not None:
                break

            continue

        last_tissue = at

        if np.isfinite(depth[at]):
            found.append((float(positions[at]), float(depth[at])))

    edge = None if last_tissue is None else float(positions[last_tissue])

    if guard_um is not None and edge is not None:
        found = [c for c in found if abs(c[0] - edge) > guard_um]

    return found, edge


def length_stops(profile, positions, min_span, min_depth=None, guard_um=None):
    """The deepest dip on each side, chosen JOINTLY so the two are at least `min_span` um apart.

    `min_depth` is Alice's rule, 2026-08-15: *"if the dip is not convincint enough it should resort to
    finding tissue edge, because in those two specific cases, the dip they are looking for isn't there"*. A
    dip shallower than that is not allowed to be an end at all, and a side left with no candidate goes to its
    own tissue edge while the other side still argues from its best dip. Applied PER SIDE, so one end can be
    a line separation and the other the tissue edge -- which is what 7 of her 14 sections look like
    ([[her-ends-are-dark-local-minima]]).

    Alice, 2026-08-15: *"just say if its shorter tahn this you probably wrong keep looking"*, and earlier,
    on a floor measured from the band's centre: *"i dont think the gray ticks are going to work because you
    are not goign to put them in the right place"*. A minimum on the WHOLE line's length is the version of
    her idea that never mentions the centre, so it cannot inherit the 139-444 um error in where along = 0
    sits ([[her-ends-are-dark-local-minima]]). Her own lines run 1543-2602 um.

    Jointly, because "too short" is a fact about the pair: when the low side has a convincing dip and the
    high side does not, the high side is the one that should give ground. Maximising the summed depth over
    all pairs at least `min_span` apart does that on its own, with no rule about which side to move.

    Falls back to the two tissue edges, which is her other rule -- *"if you cant find the dip just go to the
    tissue edge"* -- and needs no knob of its own, since it only fires when no pair of dips can reach.
    """
    low_side, low_edge = dip_candidates(profile, positions, -1, guard_um)
    high_side, high_edge = dip_candidates(profile, positions, +1, guard_um)

    if min_depth is not None:
        low_side = [c for c in low_side if c[1] >= min_depth]
        high_side = [c for c in high_side if c[1] >= min_depth]

        # One side unconvinced: it takes its tissue edge, and the other side still has to reach min_span from
        # there, so a rejected end does not let the surviving one stop short.
        if not low_side and high_side:
            reference = low_edge if low_edge is not None else float(positions[0])
            usable = [c for c in high_side if c[0] - reference >= min_span]

            return reference, (max(usable, key=lambda c: c[1])[0] if usable else high_edge)

        if not high_side and low_side:
            reference = high_edge if high_edge is not None else float(positions[-1])
            usable = [c for c in low_side if reference - c[0] >= min_span]

            return (max(usable, key=lambda c: c[1])[0] if usable else low_edge), reference

    best, pair = -np.inf, None

    for low, low_depth in low_side:
        for high, high_depth in high_side:
            if high - low >= min_span and low_depth + high_depth > best:
                best, pair = low_depth + high_depth, (low, high)

    return pair if pair is not None else (low_edge, high_edge)


def outer_stops(profile, positions, min_span, share, guard_um=None):
    """The OUTERMOST dip on each side that is at least `share` of that side's own deepest dip.

    `diag_end_dips.py`, 2026-08-17, on OR408 -- the one section whose window loses her pixels (274 um short at
    the high end, 40.8% of its wrong pixels):

        low   hers  -840   chosen -1040   nearest dip of all  -832 (depth 0.06)
        high  hers +1570   chosen +1296   nearest dip of all +1568 (depth 0.16)

    A dip sits within 2 um of BOTH ends she drew. They are faint, and `length_stops` maximises summed depth, so
    a shallower candidate can never win no matter how far out it lies -- which also means lowering the absolute
    0.227 floor alone cannot fix this section. The objective has to change, not the candidate set.

    SHARE, not a depth. The floor becomes a fraction of the section's own deepest dip instead of an absolute
    number, which is the move Alice has now been right about twice ([[relative-floor-beats-absolute]],
    [[a-relative-floor-beats-any-absolute-one]]): a faint section's dips are all faint together. `share = 1.0`
    leaves only the deepest candidate and reproduces the deepest-dip behaviour, so it is the control.

    OUTERMOST, because her ends are the LAST dark line before the tissue runs out, and the interior laminae that
    outvote them on depth are all further in ([[continuous-laminae-are-not-her-border]] measured the same thing
    from the other direction: continuity finds laminae, not borders).

    `guard_um` matters much more here than for `length_stops`: reaching outward is exactly where the tissue
    fading at its own boundary manufactures a deep dip out of blur, which is the OR99_RH_1-1-1 failure
    `dip_candidates` documents. Falls back to `length_stops` if the pair cannot reach `min_span`, and to the two
    tissue edges if a side has no dip at all -- *"if you cant find the dip just go to the tissue edge"*.
    """
    low_side, low_edge = dip_candidates(profile, positions, -1, guard_um)
    high_side, high_edge = dip_candidates(profile, positions, +1, guard_um)

    def outermost(candidates, direction):
        if not candidates:
            return None

        best = max(depth for _, depth in candidates)

        # A section whose deepest "dip" is not dark at all has nothing to be a share OF, so it keeps the
        # absolute floor rather than admitting everything at share * a negative number.
        if best <= 0.0:
            return None

        kept = [position for position, depth in candidates if depth >= share * best]

        return (min(kept) if direction < 0 else max(kept)) if kept else None

    low, high = outermost(low_side, -1), outermost(high_side, +1)

    if low is None or high is None or high - low < min_span:
        return length_stops(profile, positions, min_span, min_depth=DEPTH_FLOOR, guard_um=guard_um)

    return low, high


def reach_stops(profile, positions, min_span, share, guard_um=None):
    """Today's pair of dips, then each end EXTENDED outward to a shallower dip worth `share` of it.

    `outer_stops` proved the cue and broke the packaging. Choosing the two ends independently instead of as a
    pair costs 0.013 before the share does anything -- OR99_RH_1-1-4 falls 0.884 to 0.714 -- and the edge guard
    another 0.015, while the share itself lifts OR408 from 0.755 to 0.819. So the gain is real and everything it
    was bundled with is a loss.

    This keeps `length_stops` as the answer and only ever moves an end FURTHER OUT:

        the pair it picks is untouched, so no section can lose the window it has today
        an end moves only if a dip lies beyond it worth `share` of the depth of the dip that end is on
        `share = 1.0` extends nothing and is the control

    RELATIVE TO THE CHOSEN DIP, not to a fixed number and not to the section's deepest: the question is whether
    the thing further out is comparable to the thing we settled on, which is a local comparison and so survives
    a section being faint overall ([[or99-is-a-low-contrast-section]]). On OR408 the high end sits on 0.27 and
    her own line is a 0.16 dip 272 um beyond it, a share of 0.59.

    Only ONE extension per side, to the outermost qualifying dip -- not a walk that chains outward, which would
    be free to cross the whole section on a run of mediocre dips.
    """
    low, high = length_stops(profile, positions, min_span, min_depth=DEPTH_FLOOR, guard_um=guard_um)

    if low is None or high is None:
        return low, high

    def extend(candidates, at, direction):
        if at is None or not candidates:
            return at

        # The depth of the dip this end is standing on. `length_stops` returns a candidate's position, so a
        # miss here means the end came from the tissue-edge fallback, which has no depth to be a share of and
        # is already as far out as the tissue goes.
        here = next((depth for position, depth in candidates if position == at), None)

        if here is None or here <= 0.0:
            return at

        beyond = [position for position, depth in candidates
                  if (position < at if direction < 0 else position > at) and depth >= share * here]

        return (min(beyond) if direction < 0 else max(beyond)) if beyond else at

    low_side, _ = dip_candidates(profile, positions, -1, guard_um)
    high_side, _ = dip_candidates(profile, positions, +1, guard_um)

    return extend(low_side, low, -1), extend(high_side, high, +1)


def anchor_stops(profile, positions, length_um):
    """ONE end from the single deepest dip on the whole line; the other placed `length_um` away.

    Alice, 2026-08-15, asking where two specific sections sit in the range of dip depths, which produced the
    measurement this decode exists for ([[review_dip_levels.py]]). Per section, the depth AT HER TWO ENDS:

        both ends dark (>0.25)            2 of 14   YW113_RH_1-1-3, LBlu59_LH
        one dark, one absent or BRIGHTER  7 of 14   e.g. Purp30 0.599 / -0.068, LBlu59_RH 0.558 / 0.007
        neither (her ends are the tissue edge)      Gre595, and the weak halves above

    A negative depth means her end sits on a bright spot, not a dark line. So the image shows ONE of her two
    ends in most sections, and every decode before this one asked it for both -- `dip` per side, and even
    `length_stops`, which lets the weak end vote and then constrains the pair. Here the strong end is the only
    thing read off the image; the other end is placed by length alone, because a length measured from HER
    OUTLINES is better evidence than a dip that is absent or inverted.

    Clamped to the tissue edge, since her outlines stop where tissue does; the fallback when no dip exists
    anywhere is both tissue edges, which is *"if you cant find the dip just go to the tissue edge"*.
    """
    low_side, low_edge = dip_candidates(profile, positions, -1)
    high_side, high_edge = dip_candidates(profile, positions, +1)

    best, at, side = -np.inf, None, 0

    for candidates, which in ((low_side, -1), (high_side, +1)):
        for position, depth in candidates:
            if depth > best:
                best, at, side = depth, position, which

    if at is None:
        return low_edge, high_edge

    if side < 0:
        high = at + length_um

        return at, high if high_edge is None else min(high, high_edge)

    low = at - length_um

    return (low if low_edge is None else max(low, low_edge)), at


def plateau_stops(profile, positions, base):
    """The two stops from whichever ladder rung is most STABLE, and so has no fitted knob of its own.

    Stability is how many other rungs put both stops within `PLATEAU_TOL_UM`. The reasoning: a stop
    sitting on a real border is where it is because the tissue changed, so nudging the threshold barely
    moves it; a stop sitting on the shoulder of a cell cluster slides as soon as the threshold moves.
    """
    stops = {f: (step_stop(profile, positions, f, -1, base), step_stop(profile, positions, f, +1, base))
             for f in FRACTIONS}

    def agrees(one, other):
        if one is None or other is None:
            return one is None and other is None

        return abs(one - other) <= PLATEAU_TOL_UM

    def stability(fraction):
        return sum(1 for other in FRACTIONS if other != fraction
                   and agrees(stops[fraction][0], stops[other][0])
                   and agrees(stops[fraction][1], stops[other][1]))

    # Ties break toward the LARGEST fraction, which marks the fewest bins as a step and so is the most
    # reluctant to call a cell cluster a border -- the failure this whole decode exists to fix.
    return stops[max(FRACTIONS, key=lambda f: (stability(f), f))]


def step_extent(along, across, green, fraction, half_um=NCM_LENGTH_UM / 2.0, depth_um=DEPTH_UM,
                decode="first", guard=True):
    """The (low, high) along window from the two-sided step test, falling back to the constant."""
    bright = bright_from(green, BLACK_LEVELS[0])
    region = fill_between(bright, along, across, THICKNESS_UM, window=(-REACH_UM, REACH_UM))
    built = deep_profile(along, region, green, across, depth_um)

    if built is None:
        return -half_um, half_um

    profile, positions = built

    if decode == "diplength":
        low, high = length_stops(profile, positions, fraction)
    elif decode == "anchor":
        low, high = anchor_stops(profile, positions, fraction)
    elif decode == "lengthelse":
        low, high = length_stops(profile, positions, LENGTH_UM, min_depth=fraction)
    elif decode == "lengthreach":
        # `fraction` is the SHARE of the chosen dip's own depth that a dip further out must reach to take over.
        low, high = reach_stops(profile, positions, LENGTH_UM, fraction,
                                guard_um=DIP_REACH_UM if guard else None)
    elif decode == "lengthouter":
        # `fraction` is the SHARE here, not a depth: the outermost dip worth that much of the section's own
        # deepest one. Guarded by default, because reaching outward is where the fading tissue edge fakes a dip
        # -- but the guard costs something of its own, so `guard=False` exists to measure it separately.
        low, high = outer_stops(profile, positions, LENGTH_UM, fraction,
                                guard_um=DIP_REACH_UM if guard else None)
    elif decode == "lengthclean":
        low, high = length_stops(profile, positions, LENGTH_UM, min_depth=fraction,
                                 guard_um=DIP_REACH_UM)
    elif decode.startswith("plateau"):
        base = decode.split("-")[1] if "-" in decode else "first"
        low, high = plateau_stops(profile, positions, base)
    else:
        low = step_stop(profile, positions, fraction, -1, decode)
        high = step_stop(profile, positions, fraction, +1, decode)

    low = -half_um if low is None else low
    high = half_um if high is None else high

    # A degenerate window would delete the section from the mean instead of scoring badly on its own
    # merits, which is the more dangerous of the two failures.
    return (low, high) if low < high else (-half_um, half_um)


def score(section, fraction, use_hp, depth_um=DEPTH_UM, decode="first", bridge=False, clamp_um=0.0,
          smooth_um=0.0,
          strip_um=0.0, strip_black=None, strip_edge=False, strip_ratio=None,
          strip_wide=False, strip_aspect=0.0, lift_um=0.0, dome=False, dome_cap_um=0.0,
          dome_balance=False, strip_extra=None, grow=False, smooth_early=False, smooth_inward=False,
          reach=False, angle_floor=0.0, extend=False, extend_rays=3, extend_order=1, bend_floor=0.0,
          pair=False, blind_um=0.0, offset_um=0.0, guard=True):
    along, across, truth = section["along"], section["across"], section["truth"]
    bright = bright_from(section["green"], BLACK_LEVELS[0])
    # --strip-relative wins over --strip-black when both are given: a relative cue with an absolute
    # cutoff bolted on would be two thresholds doing the same job, and the union with BLACK_LEVELS[0]
    # inside relative_bright is already the only absolute part it needs.
    strip_bright = (relative_bright(section["green"], strip_ratio) if strip_ratio is not None
                    else None if strip_black is None
                    else bright_from(section["green"], strip_black))
    floors = strip_um

    # `strip_extra=(ratio, floor_um)` adds a SECOND cue rather than replacing the first, which is the one
    # thing the "relative wins over absolute" rule above cannot express. The two measured cues own different
    # sections, so a ray stops at a long strip found by either -- see `strip_cues` in rule_two_borders.
    if strip_extra is not None:
        ratio, extra_um = strip_extra
        strip_bright = [strip_bright, relative_bright(section["green"], ratio)]
        floors = [strip_um, extra_um]

    window = step_extent(along, across, section["green"], fraction, depth_um=depth_um, decode=decode,
                         guard=guard)

    # `bridge`/`clamp_um` deliberately do NOT reach the fill inside step_extent, which builds the profile
    # the END CUTS are decoded from. Keeping them out of it means any change in the numbers is the far
    # border's, and cannot be a window that moved for a second reason.
    region, stop, width = fill_between(bright, along, across, THICKNESS_UM, window=window,
                                       return_stop=True,
                          bridge=bridge, clamp_um=clamp_um, smooth_um=smooth_um,
                          strip_um=floors,
                          strip_bright=strip_bright, strip_edge=strip_edge,
                          strip_wide=strip_wide, strip_aspect=strip_aspect,
                          lift_um=lift_um, dome=dome, dome_cap_um=dome_cap_um,
                          dome_balance=dome_balance, grow=grow, smooth_early=smooth_early,
                          smooth_inward=smooth_inward, reach=reach,
                          angle_floor=angle_floor, extend=extend,
                          extend_rays=extend_rays, extend_order=extend_order, bend_floor=bend_floor,
                          # `values` only when the pair cue asks for it: ray_table builds the mean-green
                          # table lazily, so leaving it off keeps every other setting's cost unchanged.
                          pair=pair, blind_um=blind_um,
                          # Slides the whole across axis so the fill starts BEHIND the Field L line.
                          # `diag_error_budget.py`: 64% of every wrong pixel is on this side, and the sign
                          # is per section (91.6% one way on OR99_LH_1-1-4, 81.8% the other on YW113 1-1-2),
                          # so this exists to MEASURE the oracle, not because a constant is expected to win.
                          offset_um=offset_um,
                          # The blind-ray pass needs no `values`; it re-reads the strip cues, not a
                          # green profile, so it costs nothing on sections with no blind rays.
                          values=section["green"] if pair else None)

    if use_hp:
        hp = hp_on_canvas(section["name"], section["pixel_size"], truth.shape)

        if hp is not None and hp.any():
            region = region & ~hp

    her = (float(np.nanmin(along[truth])), float(np.nanmax(along[truth])))

    # How far the drawn border moves between neighbouring rays, on rays where both ends really stopped
    # somewhere (stop == width means the ray ran off the end of the table and never stopped at all).
    found = stop < width
    pairs = found[:-1] & found[1:]
    steps = (np.abs(np.diff(stop.astype(float)))[pairs] * ACROSS_BIN_UM if pairs.any()
             else np.zeros(0))

    return {
        "scores": overlap(region, truth),
        "steps": steps,
        "kept": cells_kept(region, truth, section["name"], section["pixel_size"]),
        "window": window,
        "error": (window[0] - her[0], window[1] - her[1]),
    }


def honest_iou(table, sections, fractions):
    """Leave-one-BIRD-out: fit the fraction on the other birds, score the held-out bird. (IoU, picks).

    Leave-one-BIRD-out and not leave-one-section-out: two sections of the same bird share stain, imaging
    and anatomy, so holding out only one of them measures how well the knob transfers between slices
    rather than between animals ([[one-name-per-bird-not-per-section]]).
    """
    birds = sorted({s["stem"].split("_")[0] for s in sections})
    honest, picks = [], []

    for held in birds:
        train = [s for s in sections if s["stem"].split("_")[0] != held]
        best = max(fractions,
                   key=lambda f: np.mean([table[(f, s["stem"])]["scores"]["iou"] for s in train]))
        picks.append(f"{held}:{best:.2f}")
        honest += [table[(best, s["stem"])]["scores"]["iou"]
                   for s in sections if s["stem"].split("_")[0] == held]

    return float(np.mean(honest)), picks


def summarise(rows):
    kept = [r["kept"] for r in rows if r["kept"] is not None]
    errors = np.abs([e for r in rows for e in r["error"]])

    return (f"{np.mean([r['scores']['iou'] for r in rows]):9.3f} "
            f"{np.mean([r['scores']['precision'] for r in rows]):7.3f} "
            f"{np.mean([r['scores']['recall'] for r in rows]):7.3f} "
            f"{100 * np.mean(kept):6.1f}% {np.median(errors):13.0f} um"
            f"{np.mean([np.percentile(r['steps'], 99) for r in rows if len(r['steps'])]):9.0f} um")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=None,
                        help="how far tissue colour must depart from the band's middle, either way")
    parser.add_argument("--decode", default=None, choices=DECODES,
                        help="which qualifying step is the border; omit to compare all of them")
    parser.add_argument("--hp", action="store_true",
                        help="also subtract her HP outline -- an ORACLE, so a ceiling not a result")
    parser.add_argument("--bridge", action="store_true",
                        help="a ray that finds no gap takes its neighbours' border instead of the image edge")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="running median of the stopping points across this many um of\n"
                             "NEIGHBOURING rays. A border is a curve down the band, so a real\n"
                             "separation stops a long run of rays together while speckle stops\n"
                             "one ray at a time -- the median keeps the first and discards the\n"
                             "second. This is the knob that lets the length floor come down.")
    parser.add_argument("--dome-balance", action="store_true",
                        help="after doming, pull the border back in by the average depth the\n                             dome added, so the shape is concave but the region does not grow.")
    parser.add_argument("--dome-cap", type=float, default=0.0,
                        help="the most um the dome may lift ONE ray. 50 is the worst single\n                             ray doming her own borders moves; 0 means no limit.")
    parser.add_argument("--dome", action="store_true",
                        help="force the far border to be one outward dome (its least concave\n                             majorant). Global, and has no knob. Doming her own borders\n                             costs 1.2% of region, ceiling 0.989.")
    parser.add_argument("--lift", type=float, default=0.0,
                        help="the mirror of --clamp: forbid the border from being pulled IN\n"
                             "faster than this many um per ray, so a notch cut by speckle is\n"
                             "lifted back out to meet its neighbours. Her own max step is\n"
                             "192 um and her p99 is 80 um.")
    parser.add_argument("--clamp", type=float, default=0.0,
                        help="um the border may move between neighbouring rays (80 = the p99 of hers)")
    parser.add_argument("--strip", type=float, default=0.0,
                        help="thin black also stops a ray when its connected piece is at least this many "
                             "um long (0 = off). Her ventricle is a median 40 um thick, so the 80 um "
                             "width test walks through it; spottiness inside NCM never reaches 200 um")
    parser.add_argument("--strip-black", type=float, default=None,
                        help="find the STRIP at this idea of black instead of the rule's own 0.005. Her "
                             "border band in YW113 1-1-2 barely registers at 0.02 and is clear at 0.08, "
                             "but thin black goes from ~1%% of the crop to 6.9%% in Gre595 -- so this "
                             "trades ribbons against confetti and has to be fitted with --strip")
    parser.add_argument("--strip-edge", action="store_true",
                        help="a strip only counts if it REACHES the tissue border, i.e. it is an arm of\n"
                             "the outside black rather than speckle floating inside the tissue. Her rule,\n"
                             "and the exact inverse of the rind test -- with --strip 0 it is the only test")
    parser.add_argument("--strip-relative", type=float, default=None,
                        help="find the strip by RELATIVE darkness instead: dark means below this "
                             "fraction of the local 400 um background. Expresses 'black because it is "
                             "against bright tissue', which no absolute cutoff can")
    parser.add_argument("--strip-aspect", type=float, default=0.0,
                        help="a magenta piece must also be at least this many times longer\n"
                             "than it is wide (length / mean width). A cleft is a ribbon; a\n"
                             "cloud of merged speckle is not. Lets the length floor come down\n"
                             "without Gre595 turning into a comb of teeth.")
    parser.add_argument("--strip-wide", action="store_true",
                        help="a strip may be any WIDTH, not just thinner than the gap test. Only\n"
                             "the outside of the brain and its 40 um rind are excluded, because\n"
                             "thinness was only ever a side effect of the rind artefact")
    args = parser.parse_args()

    sections = load()
    fractions = (args.fraction,) if args.fraction is not None else FRACTIONS
    decodes = (args.decode,) if args.decode is not None else DECODES
    summary = []

    for decode in decodes:
        # "plateau" chooses its own rung per section, so sweeping the fraction would print the same
        # row seven times.
        # --fraction FIRST. It used to come last, so a decode with its own LADDER ignored it and then the
        # per-section table printed LADDERS[decode][0] under the heading of the fraction that was asked for
        # -- the 0.05 rule labelled 0.20. An explicit setting has to win over a default ladder.
        rungs = ((args.fraction,) if args.fraction is not None
                 else (FRACTIONS[0],) if decode in KNOBLESS
                 else LADDERS.get(decode, fractions))
        table = {(f, s["stem"]): score(s, f, args.hp, decode=decode, bridge=args.bridge,
                                      clamp_um=args.clamp, smooth_um=args.smooth,
                                      strip_um=args.strip,
                                      strip_black=args.strip_black,
                                      strip_edge=args.strip_edge,
                                      strip_ratio=args.strip_relative,
                                      strip_wide=args.strip_wide,
                                      strip_aspect=args.strip_aspect, lift_um=args.lift,
                                      dome=args.dome, dome_cap_um=args.dome_cap,
                                      dome_balance=args.dome_balance)
                 for f in rungs for s in sections}

        print(f"\n  DECODE {decode}"
              + (f"   + bridge" if args.bridge else "")
              + (f"   + clamp {args.clamp:.0f} um/ray" if args.clamp else "")
              + (f"   + strips >= {args.strip:.0f} um" if args.strip else "")
              + (f" found at black < {args.strip_black:g}" if args.strip_black is not None else "")
              + ("   + must reach the tissue border" if args.strip_edge else "")
              + (f"   RELATIVE < {args.strip_relative:g} x local"
                 if args.strip_relative is not None else "")
              + ("   + strips may be wide" if args.strip_wide else "")
              + (f"   aspect >= {args.strip_aspect:g}" if args.strip_aspect else "")
              + (f"   + lift {args.lift:g} um" if args.lift else "")
              + ("   + dome" if args.dome else "")
              + (f" capped {args.dome_cap:g} um" if args.dome and args.dome_cap else "")
              + (" balanced" if args.dome and args.dome_balance else ""))

        if args.fraction is not None or decode in KNOBLESS:
            print(f"    {'section':26s} {'IoU':>7s} {'prec':>7s} {'recall':>7s} {'cells':>7s} "
                  f"{'window':>18s} {'err low/high':>16s} {'p99 step':>10s}")

            for section in sections:
                row = table[(rungs[0], section["stem"])]
                kept = "-" if row["kept"] is None else f"{100 * row['kept']:6.1f}%"
                print(f"    {section['stem'][:24]:26s} {row['scores']['iou']:7.3f} "
                      f"{row['scores']['precision']:7.3f} {row['scores']['recall']:7.3f} {kept:>7s} "
                      f"{row['window'][0]:8.0f}..{row['window'][1]:<8.0f} "
                      f"{row['error'][0]:+7.0f} /{row['error'][1]:+7.0f}"
                      f"{np.percentile(row['steps'], 99) if len(row['steps']) else 0:8.0f} um")

        print(f"    {'fraction':>9s} {'mean IoU':>9s} {'prec':>7s} {'recall':>7s} {'cells':>7s} "
              f"{'median |end err|':>17s} {'p99 step':>11s}")

        for fraction in rungs:
            label = "own" if decode in KNOBLESS else f"{fraction:.2f}"
            print(f"    {label:>9s} {summarise([table[(fraction, s['stem'])] for s in sections])}")

        if len(rungs) > 1:
            honest, picks = honest_iou(table, sections, rungs)
            print(f"    leave-one-bird-out picks: {', '.join(picks)}")
            print(f"    HONEST mean IoU {honest:.3f}")
        else:
            honest = float(np.mean([table[(rungs[0], s["stem"])]["scores"]["iou"] for s in sections]))
            print(f"    no knob to fit, so this IS honest: mean IoU {honest:.3f}")

        summary.append((decode, honest))

    print("\n  HONEST mean IoU by decode:")

    for decode, honest in sorted(summary, key=lambda pair: -pair[1]):
        print(f"    {decode:10s} {honest:.3f}")

    # The oracle number was measured with the OLD far border, before --bridge and --strip, so it is no
    # longer a ceiling on this table -- 0.801 with strips does not mean her ends have been reached. It
    # stays only as the historical row it was.
    print("\n  baselines: fixed +-950 um 0.759,  dim-only walk 0.764,  step/first 0.774,"
          "  her ends (oracle, OLD far border) 0.802"
          + ("\n  --hp is an ORACLE: it reads her HP outline, so it is a ceiling." if args.hp else ""))


if __name__ == "__main__":
    main()
