"""NCM's region, exactly as it ships. One definition, so the app and the figures cannot drift apart.

Lifted UNCHANGED out of `review_where_wrong.py`, which is a picture script: it imports PIL and the whole
figure stack, and fifteen other files were importing the shipped rule THROUGH it. `predict_regions.py` needs
this rule while napari is starting, so it needed a home that is only the rule ([[draw_helpers]] moved
`resolve_image` for the same reason). `review_where_wrong` now imports it from here, so every existing caller
keeps working and there is still only one copy.

WHAT THE RULE IS: fill from the Field L line outward along every ray until the first wide gap, with the stops
decoded as a smooth curve down the band rather than independently per ray. The flag list below is the shipped
setting and nothing else -- every one of those flags is a measurement written up in `rule_two_borders.py`'s
docstring or in one of the `scan_*.py` files, and changing one here silently changes what ships.

The constants live in `review_pair.py` (`FRACTION, DECODE, LIFT, K, RATIO, EXTRA_UM, SMOOTH = 0.227,
"lengthelse", 80.0, 2.0, 0.8, 608.0, 560.0`). That is a bad address for them and worth moving one day; it is
not worth moving in the same edit that changes what the app does.

CMM is NOT here: it is a different rule with a different shape prior, in `rule_cmm_rays.rays`.
"""
import numpy as np  # noqa: F401  -- kept so the module reads like every other rule file

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

from review_pair import ANGLE, DECODE, EXTEND_ORDER, EXTEND_RAYS, EXTRA_UM, FRACTION, K, LIFT, RATIO, SMOOTH
from rule_step_stop import THICKNESS_UM, step_extent
from rule_two_borders import BLACK_LEVELS, bright_from, fill_between, relative_bright
from scan_cue_and_floor import dark_percent


def rule_border(section, blind_um=0.0):
    """(region, record, predicted stop per ray, width) for the rule exactly as it ships.

    `record` rather than `return_stop` alone, because the ray table is what turns a step index into pixels on
    the canvas, and rebuilding it here would risk a different frame than the one the stops were decoded in.
    """
    green = section["green"]
    bright = bright_from(green, BLACK_LEVELS[0])
    window = step_extent(section["along"], section["across"], green, FRACTION, decode=DECODE)
    floor = K * 100.0 * dark_percent(green, 0.02, None)

    record = {}
    region, stop, width = fill_between(
        bright, section["along"], section["across"], THICKNESS_UM, window=window, return_stop=True,
        bridge=True, strip_um=[floor, EXTRA_UM],
        strip_bright=[bright_from(green, 0.02), relative_bright(green, RATIO)],
        lift_um=LIFT, grow=True, smooth_um=SMOOTH, smooth_early=True, smooth_inward=True, reach=True,
        angle_floor=ANGLE, extend=True, extend_rays=EXTEND_RAYS, extend_order=EXTEND_ORDER,
        blind_um=blind_um, values=green, record=record)

    return region, record, stop, width
