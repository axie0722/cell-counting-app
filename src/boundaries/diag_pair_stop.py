"""On the rays that find NO black gap, does the dark-bright pair land closer to her border than the bridge?

Alice, 2026-08-16: *"yes they are, you can continue with the next step"* -- the green marks in
`grids/hp seam rays OR408_RH_1-2-1_NCM_Slide.png` are the line she meant, so the cue measured in
`diag_hp_seam_profile.py` is real: a dark lamina with a bright ridge beside it, above chance on all 8
measurable sections ([[hp-border-is-a-dark-bright-pair]]).

WHAT THIS SCRIPT REFUSES TO DO, and why. The far border is already right to 8-48 um on every ray that finds a
black gap ([[far-border-is-solved-where-the-gap-is]]). A new cue applied to all rays can therefore only break
those, which is exactly how the last three border decodes lost to the constant rule
([[border-detector-hurts-the-region]], and [[border-cue-is-calibrated-not-specific]] measured the dip firing on
27% of already-correct rays). So the comparison here is restricted to the rays where `found` is False -- the
ones whose answer is currently COPIED FROM THEIR NEIGHBOURS by `bridge_stops`. Beating a copy is the whole bar.

NOT A FIRST CROSSING, which is the other trap. A test calibrated so that 10% of interior steps pass it will
fire about 10 steps past wherever the search starts, cutting every ray short -- the failure written up at
DIP_WIDTH_UM in rule_two_borders (IoU 0.759 -> 0.652). On a ray with no wall there is no competing evidence, so
the honest decode is the STRONGEST pair along the ray, not the first one above a bar. Same shape as the path
decode that already beat thresholding on the Field L border ([[path-decode-lands-within-112um]]).

THE FRAME IS FREE. In the band frame `across` IS the across-the-border direction, so `ray_table`'s `mean` is
already a cross-border profile per ray -- no resampling, no normals, and the pair test is just "is this step the
local minimum of its neighbourhood, and how much brighter is the local maximum".

Reported as a distance in um from her own border, per section, because an IoU cannot say whether a cue moved a
ray towards her line or past it.

Usage:
  python diag_pair_stop.py
  python diag_pair_stop.py --reach 64 --start 300
"""

import argparse

import numpy as np

from rule_step_stop import THICKNESS_UM, step_extent
from rule_two_borders import (ACROSS_BIN_UM, BLACK_LEVELS, PAIR_REACH_UM, PAIR_START_UM, bridge_stops,
                              bright_from, fill_between, load, pair_stops, relative_bright)

# The winning far-border settings, so this measures a change to the rule that ships rather than to a toy:
# end cuts from `lengthelse` @ 0.227, strips at least 400 um long off a 0.02 cutoff, plus the relative cue at
# 0.8 with its own 608 um floor ([[relative-cue-and-floor-union]]), the barrier grown through connected black
# ([[tapered-gaps-leak-rays]]), and the inward-only median with the 80 um mirror.
DECODE, STEP, FLOOR, BLACK = "lengthelse", 0.227, 400.0, 0.02
EXTRA_RATIO, EXTRA_UM = 0.8, 608.0
LIFT_UM, SMOOTH_UM, ANGLE_FLOOR = 80.0, 560.0, 80.0

# The cue itself now lives in rule_two_borders as `pair_score`/`pair_stops`, next to the rest of the stop
# machinery, so this script and the rule can never drift apart. REACH_UM and START_UM are its constants.
REACH_UM, START_UM = PAIR_REACH_UM, PAIR_START_UM


def her_border(table, truth):
    """Per ray: the step just past the last one she drew as NCM. `table`'s own frame, so nothing is resampled."""
    rays, width = table["black"].shape
    inside = truth[table["rows"], table["columns"]]
    out = np.zeros(rays, np.int64)

    np.maximum.at(out, table["ray"][inside], table["step"][inside] + 1)

    return np.where(out > 0, out, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach", type=float, default=REACH_UM)
    parser.add_argument("--start", type=float, default=START_UM)
    args = parser.parse_args()

    sections = load()

    if not sections:
        raise SystemExit("no scoreable sections")

    print(f"\n  ONLY the rays that found no wall of their own. `bridge` is what the rule does today (copy the")
    print(f"  neighbours); `pair` is the strongest dark-bright pair on the ray. Both as |distance from HER")
    print(f"  border| in um, median over those rays. Lower is better, and `pair` has to beat `bridge`.\n")
    print(f"  {'section':26s} {'rays':>6s} {'gapless':>8s} {'bridge':>7s} {'pair':>7s} {'better':>7s} "
          f"{'pair-hers':>10s}")

    rows = []

    for section in sections:
        along, across, truth = section["along"], section["across"], section["truth"]
        bright = bright_from(section["green"], BLACK_LEVELS[0])
        window = step_extent(along, across, section["green"], STEP, decode=DECODE)

        record = {}
        _, stop, width = fill_between(
            bright, along, across, THICKNESS_UM, window=window, return_stop=True, bridge=True,
            strip_um=[FLOOR, EXTRA_UM],
            strip_bright=[bright_from(section["green"], BLACK),
                          relative_bright(section["green"], EXTRA_RATIO)],
            lift_um=LIFT_UM, grow=True, smooth_um=SMOOTH_UM, smooth_early=True, smooth_inward=True,
            angle_floor=ANGLE_FLOOR, values=section["green"], record=record)

        table, found = record["table"], record["found"]
        gapless = ~found

        if gapless.sum() < 5:
            print(f"  {section['stem'][:24]:26s} {len(found):6d} {int(gapless.sum()):8d}   "
                  f"too few gapless rays to compare")
            continue

        hers = her_border(table, truth)
        bridge = bridge_stops(record["wall"].copy(), width)
        # The limit is the ray's own end: these rays found no wall, so there is nothing nearer to stop at.
        pair = pair_stops(table["mean"], table["seen"], np.full(len(found), width),
                          reach_um=args.reach, start_um=args.start)

        # Rays she never drew at all carry no answer to be judged against, so they are dropped rather than
        # scored against the window edge -- which would reward whichever cue happens to stop later.
        judged = gapless & (hers < width)

        if judged.sum() < 5:
            print(f"  {section['stem'][:24]:26s} {len(found):6d} {int(gapless.sum()):8d}   "
                  f"too few gapless rays inside her outline")
            continue

        bridge_error = np.abs(bridge[judged] - hers[judged]) * ACROSS_BIN_UM
        pair_error = np.abs(pair[judged] - hers[judged]) * ACROSS_BIN_UM
        signed = (pair[judged] - hers[judged]) * ACROSS_BIN_UM

        better = float(np.mean(pair_error < bridge_error))

        print(f"  {section['stem'][:24]:26s} {len(found):6d} {int(judged.sum()):8d} "
              f"{np.median(bridge_error):7.0f} {np.median(pair_error):7.0f} {better:7.0%} "
              f"{np.median(signed):+10.0f}", flush=True)

        rows.append({"stem": section["stem"], "bird": section["stem"].split("_")[0],
                     "bridge": float(np.median(bridge_error)), "pair": float(np.median(pair_error)),
                     "better": better, "signed": float(np.median(signed)), "rays": int(judged.sum())})

    if not rows:
        raise SystemExit("no section had enough gapless rays to compare")

    print(f"\n  {'median over sections':26s} {'':6s} {np.median([r['rays'] for r in rows]):8.0f} "
          f"{np.median([r['bridge'] for r in rows]):7.0f} {np.median([r['pair'] for r in rows]):7.0f} "
          f"{np.median([r['better'] for r in rows]):7.0%} {np.median([r['signed'] for r in rows]):+10.0f}")

    wins = [r for r in rows if r["pair"] < r["bridge"]]
    print(f"\n  the pair beats the bridge on {len(wins)} of {len(rows)} sections: "
          f"{', '.join(r['bird'] for r in wins) if wins else 'none'}")
    print(f"  a positive `pair-hers` means the pair stops PAST her border, a negative one short of it.")
    print(f"\n  Only if this wins is it worth adding to fill_between -- and then only for gapless rays.")


if __name__ == "__main__":
    main()
