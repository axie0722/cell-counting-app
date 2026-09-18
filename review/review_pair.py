"""The far border with the dark-bright pair OFF and ON, on the section it can actually change.

Alice, 2026-08-16: *"can i see i thought you said it worked"* -- and both halves of that are right, which is
exactly what this picture is for. The CUE works: on OR408's 54 rays that find no wall, the strongest pair lands
a median 112 um from her border where copying the neighbours lands 180 um out (`diag_pair_stop.py`). The REGION
barely moves: +0.001 IoU, +1.5 points of recall, -1.8 of precision (`scan_pair.py`). So the line moves visibly
and the score does not, because 54 rays out of 147 on one section is a thin sliver of area.

RED is the rule as it ships, GREEN is the same rule with the pair on, and red is drawn first so every red pixel
still visible is a real disagreement rather than an overlap. Her own outline is the grey dashed line with the
faint tint inside, so the question "which one is closer to hers" is answerable by eye.

Only the FAR border, not the whole region outline: the end cuts and the Field L side are decoded by rules the
pair does not touch, and drawing them would put unchanged pixels in a comparison picture.

Usage:
  python review/review_pair.py
  python review/review_pair.py --only Wh175 --full
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
from hippocampus_check import overlap
from review_lift import panel
from review_predicted_lines import predicted_lines
from rule_step_stop import THICKNESS_UM, score, step_extent
from rule_two_borders import BLACK_LEVELS, bright_from, fill_between, load, relative_bright
from scan_cue_and_floor import dark_percent

# The shipping stack, the same constants scan_pair.py graded, so the picture cannot be of a different rule
# than the table.
FRACTION, DECODE, LIFT, K, RATIO, EXTRA_UM, SMOOTH = 0.227, "lengthelse", 80.0, 2.0, 0.8, 608.0, 560.0
ANGLE, EXTEND_RAYS, EXTEND_ORDER = 80.0, 8, 2

PAD_UM = 120.0


def region_at(section, pair):
    """The region mask and its far border line, with the pair cue on or off. Everything else identical."""
    green = section["green"]
    bright = bright_from(green, BLACK_LEVELS[0])
    window = step_extent(section["along"], section["across"], green, FRACTION, decode=DECODE)
    floor = K * 100.0 * dark_percent(green, 0.02, None)

    region, stop, width = fill_between(
        bright, section["along"], section["across"], THICKNESS_UM, window=window, return_stop=True,
        bridge=True, strip_um=[floor, EXTRA_UM],
        strip_bright=[bright_from(green, 0.02), relative_bright(green, RATIO)],
        lift_um=LIFT, grow=True, smooth_um=SMOOTH, smooth_early=True, smooth_inward=True, reach=True,
        angle_floor=ANGLE, extend=True, extend_rays=EXTEND_RAYS, extend_order=EXTEND_ORDER,
        # `values` is the green channel the pair profile is read from; harmless when pair is off.
        values=green, pair=pair)

    return region, window, stop, width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="OR408",
                        help="substring of the sections to draw. OR408 is the only one with blind rays.")
    parser.add_argument("--full", action="store_true",
                        help="draw the whole region boundary instead of the far border alone")
    args = parser.parse_args()

    data = np.load(DATA_PATH)
    greens = {str(s): data["inputs"][i, 0].astype(np.float32) for i, s in enumerate(data["stems"])}

    for section in load():
        if args.only and args.only not in section["stem"]:
            continue

        lines = []

        for pair in (False, True):
            region, window, _, _ = region_at(section, pair)
            line = (region & ~binary_erosion(region, iterations=1) if args.full
                    else predicted_lines(region, section["along"], section["across"], window)["far border"])
            lines.append((binary_dilation(line, iterations=2), overlap(region, section["truth"])["iou"]))

        (off_line, off_iou), (on_line, on_iou) = lines

        truth = section["truth"]
        picture = panel(greens[section["stem"]], truth, off_line, on_line)

        # Cropped to everything the comparison is about -- her region plus either border -- so the section's
        # empty half does not shrink the part being judged.
        rows, cols = np.nonzero(truth | off_line | on_line)
        pad = int(round(PAD_UM / 8.0))
        box = (max(int(cols.min()) - pad, 0), max(int(rows.min()) - pad, 0),
               min(int(cols.max()) + pad, picture.width), min(int(rows.max()) + pad, picture.height))
        picture = picture.crop(box)

        grow = max(1, min(3, 1500 // max(picture.width, 1)))
        picture = picture.resize((picture.width * grow, picture.height * grow), Image.NEAREST)

        path = Path("grids") / f"pair {section['stem'][:24].strip()}.png"
        picture.save(path)

        print(f"  {section['stem'][:34]:36s} IoU {off_iou:.3f} -> {on_iou:.3f}   wrote {path}", flush=True)

    print(f"\n  RED is today's rule, GREEN is the same rule with the pair cue on the blind rays, grey dashes "
          f"are hers.")


if __name__ == "__main__":
    main()
