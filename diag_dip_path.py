"""A PATH through the relative-darkness score: does continuity ACROSS rays find her dark line?

Alice, 2026-08-17: *"the thing im trying to get you to identify isn't a bright line its the dark line that my
annotations outline in the area i circled"*. `diag_circle_dip.py` says she is right and says exactly why the rule
cannot see it:

  the cue fires   a relatively dark cell sits within 80 um of her line on 97 of her 103 circled rays, at 0.285
                  of the local brightness -- one of the darkest things in the neighbourhood
  and is deleted  the connected piece it belongs to is 16-48 um long. Her line is a DOTTED line in pixel space,
                  and 527-1173 other specks in the section are just as long, so no length floor can keep it
                  ([[thin-black-strips-are-her-border]] measured the collapse: IoU 0.832 -> 0.020)

The specks are not scattered, though: they are continuous ACROSS the band even though the mask is not connected
WITHIN it. That is a different constraint, and the one this tests -- pick one position per ray, and make each
position sit near its neighbour's, instead of asking a blob to be long.

The decode is the same dynamic program that fixed the Field L border when its cue was equally intermittent
([[path-decode-lands-within-112um]], [[border-cue-is-intermittent-not-absent]]): score every cell, then take the
highest-scoring path that moves at most `--slope` steps per ray. No threshold anywhere, which is the point --
[[border-detector-threshold-does-not-transfer]] is what a threshold does across birds.

Graded honestly: median |path - her line| per section against the SHIPPED rule's own error on the same rays. A
path that wins on OR408 and loses on the twelve black-gap sections is not shippable as a replacement, so both
columns matter ([[border-cue-is-calibrated-not-specific]] is the trap).

Usage:
  python diag_dip_path.py
  python diag_dip_path.py --slope 2 --start 300 --only OR408
"""

import argparse

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

from diag_pair_stop import her_border
from review_where_wrong import rule_border
from rule_two_borders import (ACROSS_BIN_UM, ALONG_BIN_UM, BLACK_LEVELS, RELATIVE_WINDOW_UM, load,
                              oriented_valley, rim_edge, valley_score)


def darkness(green, table, shape):
    """Per cell, 1 - (green / its own 400 um neighbourhood): high where a cell is darker than around it.

    The quantity `relative_bright` thresholds, kept CONTINUOUS instead. A path needs a score, not a mask --
    thresholding is what broke her line into 16 um specks in the first place.
    """
    size = max(int(round(RELATIVE_WINDOW_UM / ACROSS_BIN_UM)), 1)
    background = uniform_filter(np.asarray(green, np.float32), size=size)
    ratio = np.asarray(green, np.float32) / np.maximum(background, 1e-6)

    out = np.full(shape, np.inf, np.float32)
    np.minimum.at(out, (table["ray"], table["step"]), ratio[table["rows"], table["columns"]])

    return np.where(np.isfinite(out), 1.0 - out, 0.0).astype(np.float32)


def along_band(score, seen, smooth_um):
    """Average the score ALONG the band, which is the one direction her line does not vary in.

    Alice, 2026-08-17: *"its not dotted though its continuous like its very clear"* -- and `look_circle.py`
    shows why the cue missed it. Her line is thin ACROSS the band and continuous ALONG it; the speckle the
    cue does mark is small and round. A threshold cannot separate those, an anisotropic average can. Weighted
    by `seen` so the cells no pixel landed in do not drag the average down.
    """
    if smooth_um <= 0.0:
        return score

    size = max(int(round(smooth_um / ALONG_BIN_UM)), 1)
    weight = uniform_filter1d(seen.astype(np.float32), size=size, axis=0, mode="nearest")
    total = uniform_filter1d(np.where(seen, score, 0.0), size=size, axis=0, mode="nearest")

    return total / np.maximum(weight, 1e-6)


def tissue_limit(mean, seen, run_um=200.0):
    """Per ray, the step where the section's OUTER SURFACE begins: the first long run of non-tissue.

    A BUG THIS REPLACES, worth spelling out because it silently inverted the specificity column. The first
    version kept only cells brighter than the tissue cutoff, to stop a path running out into the background --
    but a black GAP inside the brain is below that cutoff too, so on the twelve sections whose border is a real
    black gap it deleted her own border from the search, and the percentile it then printed was measured on the
    minority of rays where her border is not black. Her border came out below chance, which is impossible and
    was the giveaway.

    Depth is the right way to say "inside the section": everything before the surface is fair game, black gaps
    included. A run rather than a single cell because speckle and the gaps between cells are dark too.
    """
    rays, width = mean.shape
    empty = (~seen) | (mean < BLACK_LEVELS[0])
    size = max(int(round(run_um / ACROSS_BIN_UM)), 1)
    solid = uniform_filter1d(empty.astype(np.float32), size=size, axis=1, mode="constant", cval=1.0) >= 0.999
    steps = np.arange(width)[None, :]
    out = np.where(solid.any(axis=1), np.argmax(solid, axis=1) - size // 2, width)

    return np.maximum(out, 1)


def best_path(score, usable, slope):
    """The highest-scoring path across the band, moving at most `slope` steps from one ray to the next.

    Plain dynamic programming, kept in one function because there is nothing to reuse: `total[ray, step]` is
    the best score of any path ending at that cell, and `came[ray, step]` remembers which neighbour it came
    from so the answer can be walked back. Cells outside `usable` are -inf, which keeps a path out of the
    shallow tissue and off the cells no pixel landed in without needing a special case.
    """
    rays, width = score.shape
    start = np.where(usable, score, -np.inf)
    total = np.full((rays, width), -np.inf, np.float32)
    came = np.zeros((rays, width), np.int64)
    total[0] = start[0]

    for ray in range(1, rays):
        # The best predecessor within +-slope, found by shifting rather than looping over offsets.
        stack = np.full((2 * slope + 1, width), -np.inf, np.float32)

        for index, shift in enumerate(range(-slope, slope + 1)):
            row = np.roll(total[ray - 1], shift)

            if shift > 0:
                row[:shift] = -np.inf
            elif shift < 0:
                row[shift:] = -np.inf

            stack[index] = row

        pick = np.argmax(stack, axis=0)
        best = np.take_along_axis(stack, pick[None, :], axis=0)[0]
        came[ray] = np.clip(np.arange(width) - (pick - slope), 0, width - 1)
        total[ray] = np.where(np.isfinite(best), best, 0.0) + start[ray]

    path = np.zeros(rays, np.int64)
    path[-1] = int(np.argmax(total[-1]))

    for ray in range(rays - 1, 0, -1):
        path[ray - 1] = came[ray, path[ray]]

    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--slope", type=int, default=2, help="steps a path may move per ray (1 step = 8 um)")
    parser.add_argument("--start", type=float, default=300.0, help="no border shallower than this")
    parser.add_argument("--smooth", type=float, default=0.0, help="along-band averaging in um")
    parser.add_argument("--cue", default="relative", choices=("relative", "valley", "oriented", "rim", "drop"),
                        help="relative: green vs its 400 um box mean. valley: darker than BOTH flanks. "
                             "oriented: a valley that continues ALONG the band minus one that crosses it. "
                             "rim: brighter 160 um outside than inside, de-trended -- the rim's inner edge")
    parser.add_argument("--blind", type=float, default=200.0)
    args = parser.parse_args()

    print(f"\n  median |border - her line| per section: the shipped rule, and a path through the relative\n"
          f"  {args.cue} darkness, slope {args.slope} step(s) per ray, nothing shallower than {args.start:.0f} um, "
          f"along-band smoothing {args.smooth:.0f} um.\n")
    print(f"  {'section':26s} {'rays':>5s} {'her %ile':>9s} {'rule um':>8s} {'path um':>8s} {'change':>8s}")

    rule_error, path_error, ranks = [], [], []

    for section in load():
        if args.only and args.only.lower() not in section["stem"].lower():
            continue

        _, record, stop, width = rule_border(section, blind_um=args.blind)
        table = record["table"]
        hers = her_border(table, section["truth"])
        judged = hers < width

        steps = np.arange(table["mean"].shape[1])[None, :]
        # INSIDE THE SECTION, by DEPTH: everything before the outer surface, black gaps included. Needed here
        # and not only in `usable` below, because a cue that AVERAGES over a window can read the background
        # from a cell that is itself well inside the tissue.
        inside = steps < tissue_limit(table["mean"], table["seen"])[:, None]

        if args.cue in ("rim", "drop"):
            score = rim_edge(table["mean"], table["seen"], smooth_along_um=args.smooth, inside=inside)
            # `drop`: the same measurement with the sign reversed -- brighter INSIDE than outside. Both
            # directions are asked because the first run put her border at the 18th percentile of `rim`, and a
            # score that is reliably at the bottom is a score that is reliably somewhere.
            score = -score if args.cue == "drop" else score
        elif args.cue == "oriented":
            score = oriented_valley(table["mean"], table["seen"])
        elif args.cue == "valley":
            score = valley_score(table["mean"], table["seen"], smooth_along_um=args.smooth)
        else:
            score = along_band(darkness(section["green"], table, table["seen"].shape), table["seen"],
                               args.smooth)
        # The background outside the brain is darker than anything in it, so a path allowed out there runs along
        # the surface -- which is what the first version of this measured (300-500 um out on all fifteen
        # sections, a bug, not a result).
        usable = table["seen"] & inside & (steps >= max(int(round(args.start / ACROSS_BIN_UM)), 1))
        path = best_path(score, usable, max(args.slope, 1))

        rank = []

        for ray in np.nonzero(judged)[0]:
            usable_here = np.nonzero(usable[ray])[0]

            if usable_here.size and hers[ray] in usable_here:
                rank.append(100.0 * float((score[ray, usable_here] <= score[ray, hers[ray]]).mean()))

        first = float(np.median(np.abs(stop[judged] - hers[judged]))) * ACROSS_BIN_UM
        second = float(np.median(np.abs(path[judged] - hers[judged]))) * ACROSS_BIN_UM
        rule_error.append(first)
        path_error.append(second)

        print(f"  {section['stem'][:24]:26s} {int(judged.sum()):5d} "
              f"{(np.median(rank) if rank else np.nan):9.0f} {first:8.0f} {second:8.0f} "
              f"{second - first:+8.0f}", flush=True)
        ranks.append(np.median(rank) if rank else np.nan)

    print(f"\n  {'mean':26s} {'':5s} {np.nanmean(ranks):9.0f} {np.mean(rule_error):8.0f} {np.mean(path_error):8.0f} "
          f"{np.mean(path_error) - np.mean(rule_error):+8.0f}")
    print(f"  a path is only shippable if it wins on OR408 WITHOUT losing the sections whose border is a real")
    print(f"  black gap -- those are already within 8-24 um and have everything to lose.")


if __name__ == "__main__":
    main()
