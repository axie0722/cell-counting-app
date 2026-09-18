"""The two axes together: a RELATIVE cue for what counts as dark, and a RELATIVE floor for how long.

Alice, 2026-08-15, pointing at `grids/predicted lines lengthelse bridged strip 608 rel 0.8.png`: *"no it
was better before like if you look at predicted lines lengthelse bridged strip 608 rel 0.8"*, then *"i
dont remember what you did there but it was good"*.

WHAT THAT FIGURE WAS. The strip cue was `relative_bright(green, 0.8)` -- a pixel is dark if it is below
0.8x its local reference -- instead of `bright_from(green, 0.02)`, an absolute cutoff on normalised green.
The floor was the old absolute 608 um. It means 0.803 over her 14 sections.

WHY BOTH AXES ARE NEEDED, from her own two examples:

  YW113_RH_1-1-2   0.790 with the relative cue, 0.775 with the absolute one. It is 0.2% black by the
                   absolute test, so there is nothing to follow; the relative test sees its border.
  OR408_RH_1-2-1   0.727 against 0.681, the biggest gap in the table.
  Purp30_LH_1-1-8  0.785 with the relative cue against 0.875 with the relative FLOOR -- the other way
                   round, because its 464 um ribbon is thrown away by a 608 um floor whatever the cue is.

So the cue and the floor fix different sections and have never been crossed. This scans them together, with
one rule about the statistic that matters: the black fraction the relative floor scales with is measured
UNDER THE CUE IN USE ([[relative-floor-beats-absolute]] measured it under 0.02), or the floor would be
scaled by a different notion of dark than the one it filters.

Knobs are fitted ONE AT A TIME, cue held fixed, because fitting both at once on 7 birds is how the folds
started disagreeing before ([[knob-count-is-a-modelling-choice]]).

Usage:
  python review/scan_cue_and_floor.py
"""

import numpy as np

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

from rule_step_stop import score
from rule_two_borders import BLACK_LEVELS, bright_from, load, relative_bright

FRACTION, DECODE, LIFT = 0.227, "lengthelse", 80.0

# (label, absolute black level, relative ratio, k ladder). Exactly one of the middle two is used.
#
# The two cues need DIFFERENT ladders and that is not a fudge: the relative cue calls several times more of
# the tissue dark than the 0.02 cutoff does, so k * 100 * percent is a different number for the same k. The
# first run of this scan had one shared ladder and every relative-cue k landed above every section's longest
# strip, i.e. it silently reproduced the no-strips row (0.789) three times. Each ladder is therefore centred
# so that its middle rung gives floors in the few-hundred-um range that her borders actually live in.
CUES = (("abs 0.02", 0.02, None, (1.0, 2.0, 3.0)),
        ("rel 0.8", None, 0.8, (0.05, 0.1, 0.2, 0.4)))

# 608 is what the picture she liked used; 0 means no strips at all, as the floor of the ladder.
ABSOLUTE = (0.0, 608.0)


def dark_percent(green, level, ratio):
    """What percent of the TISSUE is dark under the cue in use. Label-free."""
    tissue = bright_from(green, BLACK_LEVELS[0])
    dark = ~(relative_bright(green, ratio) if ratio is not None else bright_from(green, level))

    return 100.0 * (dark & tissue).sum() / max(tissue.sum(), 1)


def main():
    sections = load()
    watch = ("YW113_RH_1-1-2", "OR408", "Purp30", "Gre595", "Wh175_LH_1-2-1")
    table, percents = {}, {}

    for label, level, ratio, _ in CUES:
        for section in sections:
            percents[(label, section["stem"])] = dark_percent(section["green"], level, ratio)

        print(f"  {label:9s} dark tissue  " + "  ".join(
            f"{s['stem'].split('_')[0][:6]} {percents[(label, s['stem'])]:4.1f}%" for s in sections),
            flush=True)

    print()

    for label, level, ratio, ks in CUES:
        for floor in ABSOLUTE + ks:
            relative = floor in ks

            for section in sections:
                um = (floor * 100.0 * percents[(label, section["stem"])] if relative else floor)
                table[(label, floor, section["stem"])] = score(
                    section, FRACTION, False, decode=DECODE, bridge=True, strip_um=um,
                    strip_black=level, strip_ratio=ratio, lift_um=LIFT)

            rows = [table[(label, floor, s["stem"])] for s in sections]
            name = f"k={floor:g}" if relative else (f"{floor:.0f} um" if floor else "no strips")

            if relative:
                floors = [floor * 100.0 * percents[(label, s["stem"])] for s in sections]
                print(f"  {label:9s} floor {name:9s} derived {min(floors):.0f}-{max(floors):.0f} um  "
                      + "  ".join(f"{w.split('_')[0][:6]} "
                                  f"{floor * 100.0 * percents[(label, next(s['stem'] for s in sections if w in s['stem']))]:.0f}um"
                                  for w in watch), flush=True)

            print(f"  {label:9s} floor {name:9s} mean IoU {np.mean([r['scores']['iou'] for r in rows]):.3f}"
                  f"   prec {np.mean([r['scores']['precision'] for r in rows]):.3f}"
                  f"   recall {np.mean([r['scores']['recall'] for r in rows]):.3f}   "
                  + "  ".join(f"{w.split('_')[0][:6]} "
                              f"{table[(label, floor, next(s['stem'] for s in sections if w in s['stem']))]['scores']['iou']:.3f}"
                              for w in watch), flush=True)

    birds = sorted({s["stem"].split("_")[0] for s in sections})

    print()

    for label, _, _, ks in CUES:
        for family, choices in (("absolute", ABSOLUTE), ("relative k", ks)):
            chosen, got = [], []

            for held in birds:
                train = [s for s in sections if s["stem"].split("_")[0] != held]
                best = max(choices, key=lambda c: np.mean(
                    [table[(label, c, s["stem"])]["scores"]["iou"] for s in train]))
                chosen.append(f"{held}:{best:g}")
                got += [table[(label, best, s["stem"])]["scores"]["iou"] for s in sections
                        if s["stem"].split("_")[0] == held]

            agree = "UNANIMOUS" if len({c.split(":")[1] for c in chosen}) == 1 else "folds disagree"
            print(f"  {label:9s} {family:11s} honest {np.mean(got):.3f}   {agree}   {', '.join(chosen)}")


if __name__ == "__main__":
    main()
