"""The mirror safeguard, on every section: the far border with and without it, side by side.

Alice, 2026-08-15: *"can i see what the mirror thing looks like on all the images"*, after *"are there
measures to ensure that the shape is regular because the ups and downs are not normal"*.

WHAT THE MIRROR IS. `clamp_stops` was the only cross-ray safeguard in the rule and it takes a MINIMUM, so
it can only ever pull the border IN: it deletes a ray that ran too far, and a ray that stopped too EARLY
gets spread to its neighbours instead of repaired. Speckle produces early stops, so on the speckled
sections the clamp made the teeth wider. `lift_stops` is the same sweep taking a MAXIMUM -- the border may
not be pulled in faster than 80 um per ray, which is the p99 of her own borders -- so a tooth is lifted
back out to meet the rays either side of it.

  mean IoU 0.799 -> 0.799 at lift 192, 0.797 at lift 80, i.e. the safeguard is free
  mean p99 step 138 um -> 124 / 74 um, against 80 um for her own outlines
  Gre595 0.717 -> 0.723 / 0.733, which is the proof its teeth were notches and not spikes

RED against GREEN is the whole point of the picture: red visible on its own is a stop the mirror decided
was too deep a notch to be real, and green visible on its own is where it put the border instead. Where
the border was already smooth the two lie on the same pixels and only green shows.

Only the FAR border is drawn, not the whole region outline. The end cuts and the Field L side are decoded
by rules the mirror does not touch, and including them would put unchanged pixels in the picture that read
as if they were part of the comparison.

Usage:
  python review_lift.py
  python review_lift.py --lift 192
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from ceiling_lines import DATA_PATH
from hippocampus_check import overlap
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predicted_lines import predicted_lines
from review_predictions import COLUMNS
from rule_step_stop import THICKNESS_UM, step_extent
from rule_two_borders import (ACROSS_BIN_UM, BLACK_LEVELS, bright_from, fill_between, load,
                              relative_bright)
from scan_strip_relative import black_percent

# The winning settings, written as constants so the picture cannot be of a different rule than the table:
# end cuts from `lengthelse` @ 0.227, the far border bridged, and thin black strips at least 400 um long
# read off a 0.02 cutoff counted as barriers ([[shape-test-loses-to-a-400um-floor]]).
DECODE, STEP, FLOOR, BLACK = "lengthelse", 0.227, 400.0, 0.02

PLAIN = (255, 70, 70)
LIFTED = (90, 255, 120)
TINT, TINT_COLOUR = 0.18, np.array([60, 200, 120], np.float32)


def border(bright, section, window, lift_um, floor_um=FLOOR, full=False, extra=None, grow=False,
           smooth_um=0.0, smooth_inward=True, reach=False, angle_floor=0.0, extend=False,
           extend_order=1, bend_floor=0.0):
    """The far border as a mask, plus its IoU and p99 cross-ray step, at one setting.

    `full` draws the WHOLE region boundary instead -- Alice, 2026-08-15: *"can i see the full outline"*.
    The far border alone answers "did this change help", but only the full outline answers "is this the
    region", and the two end cuts and the Field L side are where the remaining error lives on OR408 and
    Wh175_LH_1-2-1 (their cuts are 353 and 217 um out while their far borders are fine).
    """
    strip_bright = bright_from(section["green"], BLACK)
    floors = floor_um

    # `extra=(ratio, floor_um)` is the SECOND cue: relatively dark against its own neighbourhood rather than
    # below 0.02. Both cues go in as barriers and the earlier stop wins ([[relative-cue-and-floor-union]]).
    if extra is not None:
        strip_bright = [strip_bright, relative_bright(section["green"], extra[0])]
        floors = [floor_um, extra[1]]

    region, stop, width = fill_between(
        bright, section["along"], section["across"], THICKNESS_UM, window=window, return_stop=True,
        bridge=True, strip_um=floors, strip_bright=strip_bright, lift_um=lift_um, grow=grow,
        smooth_um=smooth_um, smooth_early=True, smooth_inward=smooth_inward, reach=reach,
        angle_floor=angle_floor, extend=extend, extend_rays=8, extend_order=extend_order,
        bend_floor=bend_floor)

    found = stop < width
    pairs = found[:-1] & found[1:]
    steps = (np.abs(np.diff(stop.astype(float)))[pairs] * ACROSS_BIN_UM if pairs.any()
             else np.zeros(1))

    line = (region & ~binary_erosion(region, iterations=1) if full
            else predicted_lines(region, section["along"], section["across"], window)["far border"])

    return (binary_dilation(line, iterations=1),
            overlap(region, section["truth"])["iou"],
            float(np.percentile(steps, 99.0)))


def panel(green, truth, plain, lifted):
    grey = (np.clip(green, 0, 1) * 255).astype(np.float32)
    rgb = np.stack([grey] * 3, axis=-1)

    inside = binary_erosion(truth, iterations=1)
    rgb[inside] = (1.0 - TINT) * rgb[inside] + TINT * TINT_COLOUR

    at = np.nonzero(truth & ~inside)
    dash = ((at[0] + at[1]) % 10) < 4
    rgb[at[0][dash], at[1][dash]] = (150, 150, 150)

    # Plain first, so the lifted border covers it wherever the two agree and every visible red pixel is a
    # real disagreement.
    rgb[plain] = PLAIN
    rgb[lifted] = LIFTED

    return Image.fromarray(rgb.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="draw just the sections whose stem contains this, at full size.")
    parser.add_argument("--red-floor", type=float, default=FLOOR,
                        help="the floor the RED border uses. 608 is what is shipped today.")
    parser.add_argument("--red-lift", type=float, default=-1.0,
                        help="the lift the RED border uses; -1 means match the green one.")
    parser.add_argument("--full", action="store_true",
                        help="draw the whole region boundary instead of the far border alone.")
    parser.add_argument("--relative", type=float, default=0.0,
                        help="draw the GREEN border with a floor of k * 100 * (percent of tissue black) "
                             "instead of the absolute one, i.e. her relative-floor idea. 2 is the "
                             "unanimous fit.")
    parser.add_argument("--lift", type=float, default=80.0,
                        help="um per ray the border may be pulled IN. 80 is the p99 of her own borders, "
                             "192 her maximum.")
    parser.add_argument("--grow", action="store_true",
                        help="the GREEN border grows the barrier through connected black, so a gap that is "
                             "wide anywhere walls a ray everywhere. Red is then the same rule without it.")
    parser.add_argument("--extra-ratio", type=float, default=0.0,
                        help="add a SECOND strip cue to the green border: dark means below this fraction of "
                             "its own neighbourhood. 0.8 is the measured setting.")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="the GREEN border takes a running median of its stops over this much border, "
                             "BEFORE the mirror, so an isolated spike is deleted rather than smeared. 560 um "
                             "(35 rays) is the fit; red is then the same rule without it.")
    parser.add_argument("--reach", action="store_true",
                        help="the GREEN region keeps only what is joined to the band line without crossing a "
                             "gap that opens to the outside of the section, so it cannot wrap around a cleft "
                             "the rays slipped past. Interior gaps are untouched.")
    parser.add_argument("--extend", action="store_true",
                        help="where a run of at least 8 rays stops on a long thin strip, the GREEN border "
                             "continues that strip's own measured slope along the band until it meets the "
                             "wide gap the rays already found. Only rays that stopped on a wide gap, or on "
                             "nothing, may be overwritten.")
    parser.add_argument("--bend", type=float, default=0.0,
                        help="the GREEN border ROUNDS any bend sharper than this many degrees over 32 um, by "
                             "averaging the ray with its neighbours. Unlike --angle this catches a SEAM "
                             "between two straight stretches, which is not a tip. Measured: it loses.")
    parser.add_argument("--round", action="store_true",
                        help="the GREEN continuation carries the strip run's CURVATURE, not just its slope, so "
                             "it eases off instead of running out straight -- but only where the fit's turning "
                             "point lies ahead of the walk. Implies --extend; red is then the straight one.")
    parser.add_argument("--angle", type=float, default=0.0,
                        help="the GREEN border deletes any corner that turns sharper than this many degrees "
                             "over a 32 um baseline, cutting the tip back to the chord between its "
                             "neighbours. 80 is just inside her own sharpest corner, 83 deg.")
    parser.add_argument("--free-red", action="store_true",
                        help="RED takes the same median but WITHOUT the inward-only rule, so the picture is "
                             "of that rule alone: red is then a border the median pushed OUT past a wall the "
                             "ray had found for itself.")
    parser.add_argument("--extra-um", type=float, default=608.0,
                        help="the length floor for that second cue. Its own, because the two cues call "
                             "very different amounts of the tissue dark.")
    args = parser.parse_args()

    data = np.load(DATA_PATH)
    greens = {str(s): data["inputs"][i, 0].astype(np.float32) for i, s in enumerate(data["stems"])}

    panels, captions = [], []

    for section in load():
        if args.only and args.only not in section["stem"]:
            continue

        bright = bright_from(section["green"], BLACK_LEVELS[0])
        window = step_extent(section["along"], section["across"], section["green"], STEP, decode=DECODE)

        # RED is what the rule does today; GREEN is the change being looked at. With --relative the two
        # differ in the floor and share the lift, so the picture is of the floor alone.
        floor = args.relative * 100.0 * black_percent(section["green"]) if args.relative else FLOOR
        red_lift = args.lift if args.relative else 0.0

        # With --extra-ratio the RED border is the relative floor on its own and green adds the second cue,
        # so red showing through is a border the absolute cue alone never found. Red then uses the same
        # floor and lift as green, or the picture would be of three changes at once.
        extra = (args.extra_ratio, args.extra_um) if args.extra_ratio else None
        red_floor = floor if extra else args.red_floor

        # With --smooth the ONE thing being looked at is the median, so RED gets every other option green has
        # -- the cues, the floor, the grown barrier and the mirror. Otherwise the picture would contain four
        # changes at once and be read as if it were about the median.
        same = dict(extra=extra, grow=args.grow) if args.smooth else {}

        # With --angle the ONE change is the angle test, so RED gets the median, the reach and the cues too.
        if args.angle:
            same = dict(extra=extra, grow=args.grow, smooth_um=args.smooth, reach=args.reach)

        # With --extend the ONE change is the continuation, so RED keeps the angle test too.
        if args.extend:
            same = dict(extra=extra, grow=args.grow, smooth_um=args.smooth, reach=args.reach,
                        angle_floor=args.angle)

        # With --bend the ONE change is the rounding, so RED gets the curved continuation too.
        if args.bend:
            same = dict(extra=extra, grow=args.grow, smooth_um=args.smooth, reach=args.reach,
                        angle_floor=args.angle, extend=True, extend_order=2)

        # With --round both borders continue the strip; the ONE change is straight against curved. So RED gets
        # extend=True at order 1 and the picture is of the roundness alone.
        if args.round:
            same = dict(same, extend=True)

        if args.free_red:
            same = dict(same, smooth_um=args.smooth, smooth_inward=False)
        plain, plain_iou, plain_step = border(bright, section, window,
                                             args.lift if args.smooth
                                             else red_lift if args.red_lift < 0 else args.red_lift,
                                             floor if args.smooth else red_floor, full=args.full, **same)
        lifted, lift_iou, lift_step = border(bright, section, window, args.lift, floor, full=args.full,
                                             extra=extra, grow=args.grow, smooth_um=args.smooth,
                                             reach=args.reach, angle_floor=args.angle,
                                             extend=args.extend or args.round or bool(args.bend),
                                             extend_order=2 if args.round or args.bend else 1,
                                             bend_floor=args.bend)

        panels.append(panel(greens.get(section["stem"], section["green"]), section["truth"],
                            plain, lifted))
        note = ("" if extra else f"floor {args.red_floor:.0f} -> {floor:.0f}um   " if args.relative else "")
        captions.append((section["stem"][:30],
                         note + f"IoU {plain_iou:.3f} -> {lift_iou:.3f}   "
                         f"p99 step {plain_step:.0f} -> {lift_step:.0f} um"))
        print(f"  {section['stem'][:30]:32s} {note}IoU {plain_iou:.3f} -> {lift_iou:.3f}   "
              f"p99 step {plain_step:4.0f} -> {lift_step:4.0f} um", flush=True)

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    grid_rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new("RGB", (COLUMNS * cell_width, HEADER_HEIGHT + grid_rows * cell_height), "black")
    draw = ImageDraw.Draw(sheet)
    if args.bend:
        draw.text((4, 4), f"NO BEND SHARPER THAN {args.bend:.0f} DEGREES: a ray that turns harder than that "
                  "over 32 um is AVERAGED with its neighbours, which rounds a corner -- the running median "
                  "deliberately preserves bends, so it never touched these seams", fill="white")
        draw.text((4, 18), "RED = the border WITHOUT the rounding.  GREEN = with it.  This catches a SEAM "
                  "between two straight stretches, which is not a tip and so is invisible to the angle test. "
                  "Measured over 14 sections it LOSES: 0.827 -> 0.826, every fold votes it off.",
                  fill=(140, 255, 160))
    elif args.round:
        draw.text((4, 4), "A ROUNDER CONTINUATION: the strip run is fitted with a CURVE rather than a straight "
                  "line, so where its black has closed the border eases off the way yours does instead of "
                  "running out at the strip's steepest slope", fill="white")
        draw.text((4, 18), "RED = the straight continuation.  GREEN = the curved one.  The curve is carried "
                  "only where its turning point lies AHEAD of the walk; where the fit would steepen instead "
                  "there is nothing to follow and the line is walked, so red and green agree.",
                  fill=(140, 255, 160))
    elif args.extend:
        draw.text((4, 4), "FOLLOW THE STRIP WHERE ITS BLACK HAS CLOSED: a run of at least 8 rays that stops "
                  "on a long thin strip continues that strip's own measured slope along the band, until it "
                  "meets the wide gap the rays had already found", fill="white")
        draw.text((4, 18), "RED = the border WITHOUT that.  GREEN = with it.  Only rays that stopped on a "
                  "WIDE GAP -- the brain's surface -- or on nothing at all may be overwritten, so a border "
                  "found from a strip of its own is never replaced by an extrapolation.", fill=(140, 255, 160))
    elif args.angle:
        draw.text((4, 4), "NO CORNER SHARPER THAN YOU DRAW: any ray whose tip turns sharper than "
                  f"{args.angle:.0f} degrees over 32 um is cut back to the straight line between its "
                  "neighbours -- your own sharpest corner anywhere is 83 degrees", fill="white")
        draw.text((4, 18), "RED = the border WITHOUT that.  GREEN = with it.  A stub is narrow AND deep at "
                  "once, which is what an angle measures and neither the median nor the mirror can see. The "
                  "ends are padded with no depth, so the corner where the border meets the end cut counts.",
                  fill=(140, 255, 160))
    elif args.reach:
        draw.text((4, 4), "ONE PATCH, NO CROSSING A GAP: the region keeps only what is joined to the Field L "
                  "line without stepping over black that opens to the outside of the section -- so it cannot "
                  "wrap around a cleft the rays slipped past", fill="white")
        draw.text((4, 18), "RED = the region WITHOUT that.  GREEN = with it.  Red showing on its own is black "
                  "space, or tissue only reachable across black. Interior gaps do not reach the outside, so "
                  "the holes you draw over stay filled.", fill=(140, 255, 160))
    elif args.smooth and args.free_red:
        draw.text((4, 4), "THE MEDIAN MAY PULL A RAY IN, NEVER PUSH IT OUT past a wall that ray found for "
                  "itself -- its neighbours vote on the SHAPE of the border, they are not evidence about "
                  "this ray", fill="white")
        draw.text((4, 18), "RED = the median voting freely.  GREEN = with the inward-only rule.  Red showing "
                  "on its own is a run of rays that stopped at real black and were outvoted out of it, so "
                  "the region wrapped around the black instead of stopping at it.", fill=(140, 255, 160))
    elif args.smooth:
        draw.text((4, 4), "SPIKES DELETED, NOT CAPPED: each ray's stop is replaced by the median of its "
                  f"neighbours over {args.smooth:.0f} um of border, before the mirror -- so a run of rays "
                  "too short to outvote its window is thrown away", fill="white")
        draw.text((4, 18), "RED = the border WITHOUT the median.  GREEN = with it.  A wide, deep border has "
                  "the votes to survive; an isolated wedge does not, and the mirror never sees it, so it is "
                  "never smeared onto its correct neighbours.", fill=(140, 255, 160))
    elif args.grow:
        draw.text((4, 4), "A GAP THAT TAPERS IS STILL A GAP: black that is wide ANYWHERE now stops a ray "
                  "along its whole connected extent, so a separation that narrows below 80 um no longer "
                  "leaks", fill="white")
        draw.text((4, 18), "RED = the border WITHOUT that.  GREEN = with it.  Red showing on its own is "
                  "where rays poured through the narrow part of a wall and ran on to the tissue edge.",
                  fill=(140, 255, 160))
    elif extra:
        draw.text((4, 4), "BOTH CUES AT ONCE: a ray stops at a long black strip found EITHER below 0.02 "
                  f"(your relative floor, k={args.relative:g}) OR below {args.extra_ratio:g}x its own "
                  f"neighbourhood (floor {args.extra_um:.0f}um)", fill="white")
        draw.text((4, 18), "RED = your relative floor ALONE.  GREEN = with the second, relative-brightness "
                  "cue added.  Red showing on its own is a border only the relative cue can see -- there is "
                  "no black there by an absolute test.", fill=(140, 255, 160))
    elif args.relative:
        draw.text((4, 4), f"A RELATIVE LENGTH FLOOR: a black strip must be longer than "
                  f"{args.relative:g} x 100 um per 1% of the tissue that is black, instead of a flat "
                  f"{FLOOR:.0f} um for every section", fill="white")
        draw.text((4, 18), f"RED = the far border at the flat {FLOOR:.0f} um floor.  GREEN = at this "
                  "section's own floor.  Both are mirrored at "
                  f"{args.lift:.0f} um, so the only difference here is the floor.", fill=(140, 255, 160))
    else:
        draw.text((4, 4), f"THE MIRROR SAFEGUARD: the border may not be pulled IN faster than "
                  f"{args.lift:.0f} um per ray (your own p99 is 80 um, your max 192)", fill="white")
        draw.text((4, 18), "RED = the far border WITHOUT it.  GREEN = WITH it.  Red showing on its own is "
                  "a tooth it decided was too deep a notch to be real; where the border was already "
                  "smooth only green shows.", fill=(140, 255, 160))
    draw.text((4, 32), "GREY DASHES = your outline.  " + ("The WHOLE region boundary is drawn, so the end "
              "cuts and the Field L side are included -- those are decoded by other rules and are the same "
              "in both colours." if args.full else "Only the far border is drawn -- the end cuts and the "
              "Field L side are decoded by rules this does not touch."), fill=(230, 230, 230))

    for position, (image, (title, note)) in enumerate(zip(panels, captions)):
        row, column = divmod(position, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 2), title, fill="white")
        draw.text((x + 4, y + image.height + 14), note, fill=(140, 255, 160))

    out = Path("grids") / ((args.only + " " if args.only else "")
                           + (f"bend {args.bend:.0f}deg" if args.bend
                              else "rounder" if args.round
                              else "extend" if args.extend
                              else f"angle {args.angle:.0f}deg" if args.angle
                              else ("reach" if args.reach else "")
                              + (f"median {args.smooth:.0f}um"
                               + (" inward" if args.free_red else "")) if args.smooth
                              else "grown barrier" if args.grow
                              else f"both cues k{args.relative:g} rel {args.extra_ratio:g}" if extra
                              else f"relative floor k{args.relative:g}" if args.relative
                              else f"lift {args.lift:.0f}") + (" full" if args.full else "") + ".png")
    sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
