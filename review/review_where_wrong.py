"""WHERE the far border is wrong, ray by ray, instead of one IoU for the whole section.

Alice, 2026-08-17: *"your predictions before were pretty accurate there was just one area where it was picking
up the wrong thing"*.

That is a statement no score in this repo can confirm or deny. An IoU of 0.83 is consistent with "everywhere
slightly off" and with "perfect except for one 600 um stretch", and those two have completely different fixes:
the first is a constant to refit, the second is one cue firing on one wrong thing in one place. Every scan so
far has been optimising as though it were the first.

So: the rule's own border, painted ray by ray and coloured by HOW FAR IT IS FROM HERS.

  GREEN    within 100 um of her line -- inside her own drawing precision
           ([[hand-outlines-already-sit-on-the-tissue-edge]] measures her hand at ~18 um of the tissue edge, and
           [[far-border-is-solved-where-the-gap-is]] measures the rule at 8-48 um on rays with a gap)
  YELLOW   100-250 um
  ORANGE   250-500 um
  RED      beyond 500 um -- a different structure, not a wobble

THE RULE'S LINE, NOT HERS. Colouring her border by the error says "something went wrong near here"; colouring
the rule's border says where it actually WENT, which is the only version that can identify what it locked onto.
Her outline stays on the picture as grey dashes so both are visible at once.

Rays she never drew on are not painted at all -- scoring them against the window edge would reward whichever
decode happens to stop later, the trap written up in `diag_pair_stop.py`.

The worst CONTIGUOUS stretch is reported rather than the worst ray, because one bad ray is noise in her clicks
and a run of them is a cue firing on the wrong structure.

Usage:
  python review/review_where_wrong.py
  python review/review_where_wrong.py --only OR408
  python review/review_where_wrong.py --only "" --quiet
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

import sys

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. This file can be run as its own
# process, and Python then puts only ITS folder on sys.path -- so `import ps6` at the root, or a
# sibling folder's module, would not be found. app_path.py explains the whole arrangement; the line
# before it is there because app_path is at the root, which is not on the path yet either. The
# condition also covers a flat copy of the app, where app_path sits right here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

from ceiling_lines import DATA_PATH
from diag_pair_stop import her_border
from review_pair import PAD_UM
# THE RULE ITSELF MOVED to `rule_ncm.py`, so the app can import it without importing this figure stack. It is
# re-exported here because fifteen other files import `rule_border` from this module by name.
from rule_ncm import rule_border
from rule_step_stop import score
from rule_two_borders import ACROSS_BIN_UM, load

# The error bands, in um, and their colours. GREEN's edge at 100 um is not arbitrary: it is roughly where the
# rule already sits on rays that find a black gap, so green means "as close as this rule has ever got".
BANDS = [(100.0, np.array([40, 220, 90], np.float32), "green"),
         (250.0, np.array([245, 230, 60], np.float32), "yellow"),
         (500.0, np.array([250, 150, 40], np.float32), "orange"),
         (np.inf, np.array([255, 45, 45], np.float32), "red")]

TINT, TINT_COLOUR = 0.16, np.array([60, 200, 120], np.float32)

# The three sides the ray error cannot be measured on: the Field L line the band frame places, and the two
# end cuts. Blue is the same colour review_predicted_lines.py gives the end cuts, so the two figures can be
# held up against each other.
OUTLINE = np.array([70, 130, 255], np.float32)


def worst_run(error, judged, threshold):
    """(first ray, last ray, mean error) of the longest run of consecutive judged rays over `threshold`.

    Consecutive in RAY ORDER, which is order along the band, so a run is a physically continuous stretch of
    border. Unjudged rays break a run rather than joining it: a gap in her drawing is not evidence of error.
    """
    best, current = None, None

    for ray in range(len(error)):
        bad = judged[ray] and abs(error[ray]) > threshold

        if bad:
            current = (ray, ray) if current is None else (current[0], ray)

            if best is None or (current[1] - current[0]) > (best[1] - best[0]):
                best = current
        else:
            current = None

    if best is None:
        return None

    first, last = best
    inside = np.arange(first, last + 1)

    return first, last, float(np.mean(np.abs(error[inside])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="OR408")
    parser.add_argument("--threshold", type=float, default=250.0,
                        help="the error a ray has to exceed to join a bad run, in um")
    parser.add_argument("--quiet", action="store_true", help="table only, no pictures")
    parser.add_argument("--full", action="store_true",
                        help="draw the COMPLETE region outline: the far border coloured by error plus the "
                             "Field L side and the two end cuts in blue. Alice, 2026-08-17: *\"can i see "
                             "the complete outlines\"*. The far border is the only side this error can be "
                             "measured on ray by ray, so the other three are drawn as themselves rather "
                             "than given a colour they have not earned.")
    parser.add_argument("--blind", type=float, default=0.0,
                        help="a shorter strip length floor for rays that found no wall, in um "
                             "(scan_blind_floor.py: 200 takes OR408 from 0.733 to 0.755 and moves no "
                             "section without blind rays)")
    args = parser.parse_args()

    data = np.load(DATA_PATH)
    greens = {str(s): data["inputs"][i, 0].astype(np.float32) for i, s in enumerate(data["stems"])}

    print(f"\n  The rule's far border against hers, ray by ray. `um of border` is split by error band, so a")
    print(f"  section that is 'accurate except one area' shows most of its length green and a short red run.\n")
    print(f"  {'section':26s} {'rays':>5s} {'median':>7s} {'green':>6s} {'yellow':>7s} {'orange':>7s} "
          f"{'red':>5s} {'worst run':>22s}")

    for section in load():
        if args.only and args.only.lower() not in section["stem"].lower():
            continue

        region, record, stop, width = rule_border(section, blind_um=args.blind)
        table, truth = record["table"], section["truth"]

        hers = her_border(table, truth)
        judged = hers < width

        if judged.sum() < 5:
            print(f"  {section['stem'][:24]:26s} too few rays she drew on")
            continue

        error = (stop.astype(np.float64) - hers) * ACROSS_BIN_UM
        shares = []
        # STARTS BELOW ZERO, not at it. A ray the rule got exactly right has error 0.0, and `> 0.0` puts it in
        # no band at all -- which left ~30% of rays uncounted in the table AND unpainted in the picture, so the
        # figure was missing precisely the stretches that work.
        low = -1.0

        for high, _, _ in BANDS:
            inside = judged & (np.abs(error) > low) & (np.abs(error) <= high)
            shares.append(100.0 * inside.sum() / judged.sum())
            low = high

        run = worst_run(error, judged, args.threshold)
        where = (f"rays {run[0]}-{run[1]}, {run[2]:.0f} um out" if run is not None
                 else f"none over {args.threshold:.0f} um")

        print(f"  {section['stem'][:24]:26s} {int(judged.sum()):5d} "
              f"{np.median(np.abs(error[judged])):7.0f} " + " ".join(f"{s:5.0f}%" for s in shares)
              + f" {where:>22s}", flush=True)

        if args.quiet:
            continue

        # Onto the canvas. Every pixel of the table carries its (ray, step), so the rule's border is the pixels
        # whose step is that ray's stop -- the same cells the score was computed from, not a re-derived line.
        rows, columns, ray, step = table["rows"], table["columns"], table["ray"], table["step"]
        grey = (np.clip(greens[section["stem"]], 0, 1) * 255).astype(np.float32)
        rgb = np.stack([grey] * 3, axis=-1)

        inside = binary_erosion(truth, iterations=1)
        rgb[inside] = (1.0 - TINT) * rgb[inside] + TINT * TINT_COLOUR

        at = np.nonzero(truth & ~inside)
        dash = ((at[0] + at[1]) % 10) < 4
        rgb[at[0][dash], at[1][dash]] = (150, 150, 150)

        on_border = judged[ray] & (step == stop[ray])
        painted = np.zeros(rgb.shape[:2], bool)
        low = -1.0

        for high, colour, _ in BANDS:
            band = on_border & (np.abs(error[ray]) > low) & (np.abs(error[ray]) <= high)
            here = np.zeros(rgb.shape[:2], bool)
            here[rows[band], columns[band]] = True
            here = binary_dilation(here, iterations=1)
            rgb[here] = colour
            painted |= here
            low = high

        # The rest of the closed outline, in blue, so the picture is a REGION and not an arc. Drawn after
        # the error colours and masked by them, so a far-border pixel never loses its colour to it.
        if args.full:
            outline = region & ~binary_erosion(region, iterations=1)
            rest = binary_dilation(outline, iterations=1) & ~painted
            rgb[rest] = OUTLINE

        picture = Image.fromarray(rgb.astype(np.uint8))

        keep = truth | painted | (region if args.full else np.zeros_like(painted))
        r, c = np.nonzero(keep)
        pad = int(round(PAD_UM / 8.0))
        picture = picture.crop((max(int(c.min()) - pad, 0), max(int(r.min()) - pad, 0),
                               min(int(c.max()) + pad, picture.width),
                               min(int(r.max()) + pad, picture.height)))

        grow = max(1, min(3, 1500 // max(picture.width, 1)))
        picture = picture.resize((picture.width * grow, picture.height * grow), Image.NEAREST)

        label = ("where wrong full" if args.full else "where wrong") + (f" blind {args.blind:.0f}"
                                                                       if args.blind else "")
        path = Path("grids") / f"{label} {section['stem'][:24].strip()}.png"
        picture.save(path)
        print(f"      wrote {path}", flush=True)

    print(f"\n  GREEN within 100 um of her line, YELLOW to 250, ORANGE to 500, RED beyond. Grey dashes are")
    print(f"  hers. A short red run on an otherwise green border is one cue locking onto one wrong structure,")
    print(f"  which is a different problem from a constant that needs refitting.")


if __name__ == "__main__":
    main()
