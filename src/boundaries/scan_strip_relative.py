"""A length floor RELATIVE to how speckled each section is, instead of one number in um for all 14.

Alice, 2026-08-15: *"another thing you can try is to make the magenta stuff all relative, like in gre595
there is no clear straight line but there is in purp30"*.

WHAT THE DIAGNOSTIC FOUND (`check_strip_lengths.py`), which is the whole reason this can work:

                     black inside her region   speckle pieces   longest speckle   black ON her border
  Purp30_LH_1-1-8              0.1%                   5              16 um             712 um
  Gre595_RH_3-1-4             13.4%                 249             608 um              32 um

In Gre595 her far border carries almost NO black, while its interior speckle forms chains 608 um long. So
no absolute floor can be right for both: 400 um throws away Purp30's real 464 um ribbon, and anything that
keeps it hands Gre595 a comb of teeth built from speckle. The floor has to be per section, and it has to
come from the image rather than from her outline.

THE STATISTIC. Black fraction inside the TISSUE -- not inside her region, which would be an oracle, and
not over the whole image, where the empty slide outside the brain swamps everything. Tissue is
`green >= BLACK_LEVELS[0]`, the same cutoff that replaced the tissue mask everywhere else
([[normalised-green-replaces-tissue-mask]]). A chance chain of length L is likely in proportion to how much
of the tissue is black, so the floor scales with it:

    floor_um = k * 100 * (percent of tissue that is black at 0.02)

One knob, k, fitted leave-one-BIRD-out, and the derived floor per section is printed so the rule can be
read as a sentence rather than trusted as a number. Deliberately no minimum floor: a second constant is how
[[knob-count-is-a-modelling-choice]] happened, and if a floor of 5 um in an ultra-clean section turns out to
cost precision, that is a result worth seeing rather than hiding behind a clip.

Length quantiles were tried first and are the WRONG statistic, recorded so this is not redone: by piece,
Gre595's p90 is 56 um against Purp30's 168, because hundreds of tiny specks hold the quantile down; by
pixel it inverts the other way, because in a clean section the border ribbon is most of the black and sets
its own p90. Both put the sections in the wrong order. Counting black is robust to exactly that.

Usage:
  python scan_strip_relative.py
  python scan_strip_relative.py --lift 0
"""

import argparse

import numpy as np

from rule_step_stop import score
from rule_two_borders import BLACK_LEVELS, bright_from, load

FRACTION, DECODE, BLACK = 0.227, "lengthelse", 0.02

# k, in um of floor per 0.01% of black tissue. 0.5 puts Gre595 at ~670 um (i.e. strips effectively off) and
# Purp30 at ~5 um (i.e. every piece kept).
KS = (0.125, 0.25, 0.5, 1.0, 2.0)

# The absolute floors to beat, from [[shape-test-loses-to-a-400um-floor]]: 400 is the unanimous winner and
# 608 is what is written into the code today.
ABSOLUTE = (100.0, 400.0, 608.0)


def black_percent(green):
    """What percent of the TISSUE is black at 0.02. Label-free: no outline is read."""
    tissue = bright_from(green, BLACK_LEVELS[0])
    dark = ~bright_from(green, BLACK)

    return 100.0 * (dark & tissue).sum() / max(tissue.sum(), 1)


def bird(stem):
    return stem.split("_")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lift", type=float, default=80.0,
                        help="the mirror safeguard, on by default at her p99 of 80 um.")
    args = parser.parse_args()

    sections = load()
    percents = {s["stem"]: black_percent(s["green"]) for s in sections}
    table = {}

    print(f"\n  relative floor = k * 100 * (percent of tissue black at {BLACK:.2f}), lift {args.lift:.0f} um")
    print(f"  {'section':26s} {'black':>7s} " + " ".join(f"{'k=' + str(k):>16s}" for k in KS))

    for section in sections:
        cells = []

        for k in KS:
            floor = k * 100.0 * percents[section["stem"]]
            table[(k, section["stem"])] = score(
                section, FRACTION, False, decode=DECODE, bridge=True, strip_um=floor,
                strip_black=BLACK, lift_um=args.lift)
            cells.append(f"{floor:7.0f}um {table[(k, section['stem'])]['scores']['iou']:.3f}")

        for floor in ABSOLUTE:
            table[(floor, section["stem"])] = score(
                section, FRACTION, False, decode=DECODE, bridge=True, strip_um=floor,
                strip_black=BLACK, lift_um=args.lift)

        print(f"  {section['stem'][:24]:26s} {percents[section['stem']]:6.2f}% " + " ".join(cells),
              flush=True)

    def mean_of(cell):
        return np.mean([table[(cell, s["stem"])]["scores"]["iou"] for s in sections])

    print()

    for k in KS:
        print(f"  k = {k:<6g} mean IoU {mean_of(k):.3f}")

    for floor in ABSOLUTE:
        print(f"  absolute {floor:.0f} um  mean IoU {mean_of(floor):.3f}")

    # Leave-one-BIRD-out over the relative rule's single knob, against the same fit over absolute floors,
    # so the two are compared on equal footing rather than relative-fitted against absolute-fixed.
    birds = sorted({bird(s["stem"]) for s in sections})

    print()

    for label, choices in (("relative k", KS), ("absolute um", ABSOLUTE)):
        chosen, got = [], []

        for held in birds:
            train = [s for s in sections if bird(s["stem"]) != held]
            best = max(choices, key=lambda c: np.mean(
                [table[(c, s["stem"])]["scores"]["iou"] for s in train]))
            chosen.append(f"{held}:{best:g}")
            got += [table[(best, s["stem"])]["scores"]["iou"]
                    for s in sections if bird(s["stem"]) == held]

        agree = "UNANIMOUS" if len({c.split(":")[1] for c in chosen}) == 1 else "folds disagree"
        print(f"  {label:12s} honest {np.mean(got):.3f}   {agree}   {', '.join(chosen)}")


if __name__ == "__main__":
    main()
