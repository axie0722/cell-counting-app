"""The rule's prediction drawn as LINES, so the decision is visible instead of the answer.

Alice, 2026-08-14, on the filled version: *"you just gave me back my own annotations? i meant like
mark the lines that you are predicting and where they should stop"*. She is right that the previous
picture was useless for her purpose. A green region over her white outline answers "how much overlap
is there", which is what the IoU column already says. What it cannot answer is "which of the four
borders is wrong, and by how far" -- and that is the only question whose answer changes the code.

So: no fill. The region's boundary is split into the three decisions that produced it, by the same
coordinates the rule itself used, and each is drawn in its own colour.

  YELLOW  the Field L line the rule placed, i.e. across = 0. Comes from find_band, not from the gap
          logic, so if this is wrong nothing downstream can be fixed.
  RED     the far border: on each ray, the step where the first wide gap stopped it. This is the
          thing three models failed to learn ([[ncm-far-border-is-the-real-error]]).
  BLUE    the two end cuts along the band, currently the constant +-950 um
          ([[region-nets-collapse-to-a-constant-line]]).

Her outline is DASHED and dim, deliberately: it is the reference to compare against, not the subject.
In the filled version it was the brightest thing in the frame, which is why the picture read as her
own annotation handed back.

THE END CUTS CAN BE THE WALK'S. Alice, 2026-08-14: *"for the most part the light blue is getting it
right"* -- the cyan ticks in `grids/line ends.png`, i.e. the two-sided walk out from the band's centre
rather than the fixed +-950 um. `--half inf@0.50` draws the whole prediction with those cuts, so the
blue lines move to where the walk stopped and the region is filled to them. Everything else is
unchanged, which is the point: the two pictures differ ONLY in where the band ends.

Usage:
  python review/review_predicted_lines.py
  python review/review_predicted_lines.py --half inf@0.50     # the walk's cuts instead of the constant
  python review/review_predicted_lines.py --thickness 160     # a wider gap, to see which stops disappear
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
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
from find_band import NCM_LENGTH_UM
from hippocampus_check import cells_kept, overlap
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS
from rule_step_stop import DECODES, step_extent
from rule_two_borders import (
    BLACK_LEVELS, along_window, bright_from, fill_between, load, relative_bright, today_rule,
)

THICKNESS_UM = 80.0

# How close to across = 0 counts as "this boundary pixel IS the Field L line", and likewise for the
# end cuts. Three canvas pixels: the region's edge is not exactly on the coordinate because the band
# frame is a rotated grid, so an exact test would leave the line dotted.
EDGE_TOLERANCE_UM = 24.0

COLOURS = {
    "field l": (255, 230, 60),
    "far border": (255, 60, 60),
    "end cut": (60, 140, 255),
}


def predicted_lines(region, along, across, window):
    """Split the region's own boundary into the three decisions that made it.

    Classified by coordinate rather than by geometry: a boundary pixel at across ~ 0 is there because
    the rule starts at the Field L line, one at either end of `window` is there because of the end cut,
    and anything else is there because a ray stopped. That is exactly the split
    `show_border_kinds.border_kinds` makes for HER outline, so the two pictures are comparable.

    `window` is the (low, high) along range the region was actually filled to, NOT +-half_um. It has to
    be, now that the walk can move each end independently: testing against the constant would paint the
    walk's own cuts as "far border" and leave a phantom blue line in empty space where the constant used
    to be, which is exactly backwards from what the picture is for.
    """
    boundary = region & ~binary_erosion(region, iterations=1)
    low, high = window

    near = boundary & (across <= EDGE_TOLERANCE_UM)
    at_end = (along <= low + EDGE_TOLERANCE_UM) | (along >= high - EDGE_TOLERANCE_UM)
    end = boundary & at_end & ~near

    return {"field l": near, "end cut": end, "far border": boundary & ~near & ~end}


def panel(green, truth, lines):
    grey = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1).astype(np.float32)

    # Her outline dashed: on every 8th pixel diagonal, so it reads as a reference rather than as the
    # brightest thing in the frame.
    edge = truth & ~binary_erosion(truth, iterations=1)
    rows, columns = np.nonzero(edge)
    dash = ((rows + columns) % 10) < 4
    rgb[rows[dash], columns[dash]] = (170, 170, 170)

    for name, mask in lines.items():
        thick = binary_dilation(mask, iterations=1)

        if thick.any():
            rgb[thick] = COLOURS[name]

    return Image.fromarray(rgb.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thickness", type=float, default=THICKNESS_UM)
    parser.add_argument("--black", type=float, default=BLACK_LEVELS[0])
    parser.add_argument("--half", default=None,
                        help="'LEASH@FRACTION' for the per-section walk, e.g. 'inf@0.50'; omit for the "
                             "fixed +-950 um cuts")
    parser.add_argument("--step", type=float, default=None,
                        help="two-sided step stop instead: tissue colour departing from the band's "
                             "middle by this fraction EITHER way (rule_step_stop, shipped 0.40); for "
                             "--decode lengthelse this is instead the minimum DIP DEPTH, 0.227")
    parser.add_argument("--bridge", action="store_true",
                        help="a ray that finds no gap takes its neighbours' border, not the image edge")
    parser.add_argument("--strip", type=float, default=0.0,
                        help="thin black also stops a ray when its connected piece is this long (608 = "
                             "longer than the longest spotty piece in her sections)")
    parser.add_argument("--strip-relative", type=float, default=None,
                        help="find the strip by relative darkness (below this fraction of the local\n"
                             "400 um background) instead of an absolute cutoff -- the only version\n"
                             "that moves YW113 1-1-2, whose border band is grey by any constant")
    parser.add_argument("--strip-edge", action="store_true",
                        help="a strip only counts if it reaches the tissue border")
    parser.add_argument("--decode", default="first", choices=DECODES,
                        help="which rule_step_stop decode places the two end cuts. It used to be hard-wired "
                             "to 'first', so this picture could not show the rule that was actually winning")
    args = parser.parse_args()

    # Same 'LEASH@FRACTION' spelling as rule_two_borders --half, so a picture and a score row can never
    # be describing two different settings.
    if args.half is not None:
        leash, fraction = args.half.split("@")
        args.half = (float(leash), float(fraction))

    data = np.load(DATA_PATH)
    greens = {str(s): data["inputs"][i, 0].astype(np.float32) for i, s in enumerate(data["stems"])}

    half_um = NCM_LENGTH_UM / 2.0
    panels, captions = [], []

    for section in load():
        bright = bright_from(section["green"], args.black)

        if args.step is not None:
            window = step_extent(section["along"], section["across"], section["green"], args.step,
                                 decode=args.decode)
        elif args.half is not None:
            window = along_window(
                args.half, section["along"], section["across"], bright, section["green"], half_um
            )
        else:
            window = None

        strip_bright = (relative_bright(section["green"], args.strip_relative)
                        if args.strip_relative is not None else None)
        region = fill_between(
            bright, section["along"], section["across"], args.thickness, window=window,
            bridge=args.bridge, strip_um=args.strip, strip_bright=strip_bright,
            strip_edge=args.strip_edge,
        )

        if not region.any():
            continue

        lines = predicted_lines(
            region, section["along"], section["across"], window or (-half_um, half_um)
        )
        rule = today_rule(section["tissue"], section["along"], section["across"])

        new, old = overlap(region, section["truth"]), overlap(rule, section["truth"])
        kept = cells_kept(region, section["truth"], section["name"], section["pixel_size"])

        panels.append(panel(greens[section["stem"]], section["truth"], lines))
        captions.append((
            section["stem"][:30],
            f"IoU {old['iou']:.3f} -> {new['iou']:.3f}"
            + (f",  cells {100 * kept:.0f}%" if kept is not None else ""),
        ))

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    grid_rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new("RGB", (COLUMNS * cell_width, HEADER_HEIGHT + grid_rows * cell_height), "black")
    draw = ImageDraw.Draw(sheet)
    cuts = (f"{args.decode} @ {args.step:.3f}" if args.step is not None
            else "the walk's cuts" if args.half else "fixed +-950 um cuts")
    draw.text((4, 4), f"THE LINES THE RULE PREDICTS, gap width {args.thickness:.0f} um, {cuts}",
              fill="white")
    draw.text((4, 18), "YELLOW = predicted Field L line.  RED = predicted far border (where each ray "
              "stopped).  BLUE = the two end cuts.  DASHED GREY = your outline, for reference.",
              fill=(140, 255, 160))

    for position, (image, (title, note)) in enumerate(zip(panels, captions)):
        row, column = divmod(position, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 2), title, fill="white")
        draw.text((x + 4, y + image.height + 14), note, fill=(140, 255, 160))

    # A separate file per cut rule, so the constant version stays on disk to compare against instead of
    # being silently overwritten by the run that was meant to be compared with it.
    tail = ((" bridged" if args.bridge else "")
            + (f" strip {args.strip:.0f}" if args.strip else "")
            + (f" rel {args.strip_relative:g}" if args.strip_relative is not None else "")
            + (" edge" if args.strip_edge else ""))
    name = (f"predicted lines {args.decode}{tail}.png" if args.step is not None
            else "predicted lines walk.png" if args.half else "predicted lines.png")
    out = Path("grids") / name
    sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
