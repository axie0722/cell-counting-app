"""CMM as a convex polygon that cannot collapse: RADII about the box centre, not absolute offsets.

`diag_polygon_collapse.py` names the bug in review_polygon.py's version. On the two sections it
ruins, nearly EVERY direction cuts into her region at once (Gre595_RH_3-1-4: 15 of 16 negative,
-5 to -418 um; relaxing the worst one alone moves x0.30 to x0.31). That is not a shape error and
not one rogue half-plane -- it is the whole polygon shrinking toward the frame's origin.

WHY IT SHRINKS. Each offset there is an absolute position in the band's frame, fitted as
`a_k * tissue_reach_k + b_k`. One number is being asked to carry two facts at once -- where CMM
sits and how big it is -- so a section whose tissue reaches less far than the training average has
every side pulled inward together. `find_band`'s box already places CMM well (72% of cells at
x1.00 the area, `which_error.py`); only its EXTENT is wrong.

SO SEPARATE THE TWO. Take the origin from the box's own centre and predict a RADIUS per direction:

  centre_k = how far the box's centroid lies along direction k        <- placement, from find_band
  radius_k = how far CMM's boundary sits beyond it in that direction  <- extent, fitted
  offset_k = centre_k + radius_k

Intersecting `projected[k] <= offset[k]` is still a convex polygon by arithmetic. What is new is
that a radius floor makes it a polygon that always CONTAINS A DISC around the box centre, so
"collapsed to a sliver" is no longer a reachable state -- the same trick as convexity itself,
one line lower down.

Two priors are measured, because which one is right is a question about the data:

  median   the median radius over the training birds. A constant SHAPE about a predicted centre.
           Not the same thing as review_polygon's constant polygon, which was a constant shape in
           a FIXED frame -- that one loses because it cannot move.
  reach    radius = a_k * (how far the tissue reaches beyond the centre in direction k) + b_k,
           least squares on training birds only, slope clipped to [0, 1.5] as before.

and each is measured with and without ShapeNet leashed onto it, in radius space.

AN AREA RATIO CANNOT SEE A DISPLACED REGION, so `out` is reported beside it: the share of the
prediction lying OUTSIDE her outline. YW113_RH_1-1-2 is x0.91 of her area with 44% of it off her
region, and YW113_RH_1-1-3 is x1.14 with 87% off it -- both read as "about the right size" and
neither is in the right place. That is a PLACEMENT error in find_band's box centre, and no rule
about black separations can reach it (see `separation_cuts`, measured and kept off).

Usage:
  python regions/polygon_solid.py
  python regions/polygon_solid.py --detail
  python regions/polygon_solid.py --label 1
"""

import argparse

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label

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

from review_pair import EXTRA_UM, K, RATIO
from review_polygon import (
    DIRECTIONS,
    LEASH_UM,
    MIN_LABEL_PX,
    NET_PERCENTILE,
    leashed,
    offsets_containing,
    rasterise,
    sections,
)
from rule_first_gap import gap_mask
from rule_step_stop import THICKNESS_UM
from rule_two_borders import bright_from, long_strips, relative_bright
from scan_cue_and_floor import dark_percent
from score_region_cells import CANVAS_UM

# No side may come in closer than this to the box centre. Her CMM is ~1 mm across at its
# narrowest, so 250 um is a disc the real region always contains -- it is a floor against
# collapse, not a fitted size.
MIN_RADIUS_UM = 250.0


def black_barrier(green):
    """Where a black separation runs: a WIDE black gap, or a thin black strip that is LONG.

    Alice, 2026-08-18, looking at the sheet: *"do you see how in like yw113 1-1-3 and gre 595 3-1-4 the
    cmm goes past the hippocampus where there is a clear black separation, can you prevent that"*. She is
    describing the same structure the NCM rule already stops on, so this is the same barrier, built from
    the same two cues and the same floors -- absolute black with a floor that scales with how dark the
    section is, plus relative dark with its own ([[relative-cue-and-floor-union]]). Nothing new is fitted.

    The polygon needs it because it is clipped to the TISSUE MASK, and the tissue mask cannot see a black
    separation at all -- a ventricle between CMM and hippocampus is tissue-coloured to a 0.10 threshold on
    raw green ([[dark-laminae-are-real-borders]]).
    """
    cues = ((bright_from(green, 0.02), K * 100.0 * dark_percent(green, 0.02, None)),
            (relative_bright(green, RATIO), EXTRA_UM))
    barrier = np.zeros(green.shape, bool)

    for bright, floor in cues:
        barrier |= gap_mask(bright, THICKNESS_UM) | long_strips(bright, THICKNESS_UM, max(floor, 8.0))

    return barrier


def box_centre(section, name, fallback):
    """find_band's box centroid as a canvas (row, column)."""
    box = section["box"][name]
    rows, columns = np.nonzero(box if box.any() else fallback)

    return float(rows.mean()), float(columns.mean())


def separation_cuts(region, section, name, barrier=None, keep_um=MIN_RADIUS_UM):
    """Cut the polygon along the LINE of each long black strip that reaches into it. MEASURED, KEPT OFF.

    VERDICT FIRST, because this function is a dead end and should not be re-derived. With the floor
    it changes nothing the connected-piece test does not already do. Without the floor it is a coin
    flip -- it moves four of fifteen sections, two right and two wrong:

      YW113_RH_1-1-2  x0.91 -> x0.71,  44% -> 28% outside her outline   the one she pointed at
      Wh175_LH_1-2-1  x1.07 -> x1.01,  27% -> 23%
      OR99_RH_1-1-4   x1.28 -> x0.41                                    over-cut to a sliver
      YW113_RH_1-1-3  x1.14 -> x0.64,  87% -> 100% outside              cut clean off her region

    Combined with the piece test and no floor, one section reaches x0.24 -- the exact collapse
    MIN_RADIUS_UM exists to make impossible. So `carve_at_black` is not what review_combined.py
    calls; it calls `stop_at_black` alone.

    WHY A LINE AND NOT THE STRIP. On YW113_RH_1-1-2 the separation between CMM and hippocampus is
    real and visible but it does not span the polygon -- it is a ~240 um strip with tissue either
    end of it -- so `stop_at_black`'s connected-component test finds one piece and cuts nothing
    (measured: 0%). Continuing the strip is the same move the NCM rule makes when its own black
    closes ([[follow-the-strip-where-black-closes]], [[tapered-gaps-leak-rays]]).

    A straight continuation is the right one HERE for a reason that does not apply there: CMM is a
    convex polygon, so cutting it with a half-plane leaves it convex, and every side it already has
    came from exactly this kind of cut. The strip's own principal axis supplies the direction; her
    outline is never consulted.

    TWO GUARDS, both off MIN_RADIUS_UM, the disc this polygon is already guaranteed to contain:

      length   the piece must be at least as long as that disc. A separation shorter than the disc
               cannot be the border of a region containing it, so this is the same statement made
               twice rather than a second knob.

               NOT the NCM rule's darkness-scaled floor, which is what the first version used and
               why it cut nothing: that floor is 608 um on YW113_RH_1-1-2 and 1211 um on
               Gre595_RH_3-1-4, sized for a strip that has to separate NCM from the tissue beyond
               across a whole ray fan. Her separation here is 280 um long. Different job, and
               borrowing a constant across jobs is not the same as not fitting one.

      distance `keep_um`: the cut is PUSHED OUT to this far from the box centre rather than refused
               when it comes closer. Refusing was the other reason nothing was cut: find_band's box
               centre on YW113_RH_1-1-2 sits 89 um from her real border, so "too close to the
               centre to be a border" threw away a correct cut.

               But pushing out to MIN_RADIUS_UM does not rescue that section either -- the polygon
               only reaches 160 um past her separation, so a line held 250 um off the centre lands
               BEYOND the polygon's own edge and cuts 3 pixels. The disc and the correct cut are in
               direct conflict there, which is why `keep_um` is a parameter and main() measures 250
               against 0 on all fifteen sections. The floor protects against a FITTED radius being
               wrong; a cut at observed black is evidence rather than a fit, so whether it should
               be bound by the same guarantee is a question for the data.
    """
    green = section["green"]
    barrier = black_barrier(green) if barrier is None else barrier

    pieces, count = label(barrier & binary_dilation(region, iterations=1))

    if not count:
        return region

    centre = box_centre(section, name, region)
    rows, columns = np.indices(green.shape, dtype=np.float32)
    out = region.copy()

    for index, window in enumerate(find_objects(pieces), start=1):
        piece = pieces[window] == index
        length = max(window[0].stop - window[0].start, window[1].stop - window[1].start) * CANVAS_UM

        if length < MIN_RADIUS_UM:
            continue

        # The strip's own direction, as the principal axis of its pixels. `eigh` returns ascending
        # eigenvalues, so the last vector is the long one and its perpendicular is the cut's normal.
        spots = np.stack([rows[window][piece], columns[window][piece]])
        anchor = spots.mean(axis=1, keepdims=True)
        _, vectors = np.linalg.eigh((spots - anchor) @ (spots - anchor).T)
        normal = np.array([-vectors[1, -1], vectors[0, -1]])

        offset = normal[0] * (centre[0] - anchor[0, 0]) + normal[1] * (centre[1] - anchor[1, 0])

        # Signed so that the box centre is at +abs(offset) and the far side of the strip is negative:
        # keeping `signed >= 0` keeps the centre's side. The floor moves the line AWAY from the centre
        # when it would otherwise cut inside the disc, and never pulls it in.
        signed = (normal[0] * (rows - anchor[0, 0])
                  + normal[1] * (columns - anchor[1, 0])) * np.sign(offset)
        out &= signed >= min(0.0, abs(offset) - keep_um / CANVAS_UM)

    return out


def carve_at_black(region, section, name, keep_um=0.0):
    """Both black tests, in the order that makes each one able to help.

    The half-plane cuts first, because they are what turn a strip that only reaches PART WAY across
    the polygon into a boundary; the component test second, to drop anything the cuts leave stranded
    on the far side of a wall that does close.
    """
    return stop_at_black(separation_cuts(region, section, name, keep_um=keep_um), section, name)


def stop_at_black(region, section, name):
    """The part of `region` on the box centre's side of every black separation.

    A CONNECTED COMPONENT and not a per-ray walk, because a region has to close: whatever is cut off
    behind the barrier is cut off however the polygon reaches it. Holes are filled afterwards, never
    before -- she draws straight over interior speckle, so a gap INSIDE CMM is still CMM
    ([[regions-must-be-continuous-patches]]), and filling before the split would bridge the separation
    itself.
    """
    allowed = region & ~black_barrier(section["green"])

    if not allowed.any():
        return region

    pieces, count = label(allowed)

    if count <= 1:
        return binary_fill_holes(allowed)

    centre = box_centre(section, name, region)

    # The piece nearest find_band's box centre, measured to the piece rather than at a single pixel:
    # the centre itself can land in the separation, and then no piece contains it.
    here = np.nonzero(allowed)
    nearest = int(np.argmin((here[0] - centre[0]) ** 2 + (here[1] - centre[1]) ** 2))

    return binary_fill_holes(pieces == pieces[here[0][nearest], here[1][nearest]])


def centre_projections(section, name, mask=None):
    """How far a mask's centroid lies along each direction, in um. Defaults to find_band's box.

    The mean of a LINEAR projection over a mask is the projection of that mask's centroid, so
    this needs no separate centroid calculation and cannot disagree with one.
    """
    if mask is None:
        mask = section["box"][name]

    if not mask.any():
        mask = section["tissue"]

    return np.array([section["projected"][k][mask].mean() for k in range(DIRECTIONS)])


def radii_of(section, mask, centre, percentile=100.0):
    """A mask's convex outline as radii measured out from `centre`."""
    return offsets_containing(section["projected"], mask, percentile) - centre


def reach_beyond(section, centre):
    """How far the tissue itself reaches past the centre in each direction, in um."""
    return np.array(
        [section["projected"][k][section["tissue"]].max() for k in range(DIRECTIONS)]
    ) - centre


def fit_radius_prior(training, label, name, kind, centre_for=None):
    """(slopes, intercepts) for radius = slope * tissue_reach_beyond_centre + intercept.

    `kind == "median"` returns zero slopes and the median radius as the intercept, so both priors
    go through the same two lines of arithmetic below and cannot drift apart.

    `centre_for` is the SAME callable the prediction uses, so a radius is always measured from the
    origin it will be added back to. Fitting radii about one centre and rasterising about another
    would flatter any change to the centre for a reason that has nothing to do with placement.
    """
    radii, reaches = [], []

    for section in training:
        centre = centre_for(section) if centre_for else centre_projections(section, name)
        radii.append(radii_of(section, section["labels"] == label, centre))
        reaches.append(reach_beyond(section, centre))

    radii, reaches = np.array(radii), np.array(reaches)

    if kind == "median":
        return np.zeros(DIRECTIONS), np.median(radii, axis=0)

    slopes = np.zeros(DIRECTIONS)
    intercepts = np.zeros(DIRECTIONS)

    for k in range(DIRECTIONS):
        x, y = reaches[:, k], radii[:, k]
        spread = ((x - x.mean()) ** 2).sum()
        slopes[k] = (
            float(np.clip(((x - x.mean()) * (y - y.mean())).sum() / spread, 0.0, 1.5))
            if spread > 0
            else 0.0
        )
        intercepts[k] = y.mean() - slopes[k] * x.mean()

    return slopes, intercepts


def solid_offsets(section, training, label, name, net_mask=None, leash_um=LEASH_UM,
                  kind="reach", min_radius_um=MIN_RADIUS_UM, centre_for=None):
    """The K half-plane offsets of the collapse-proof polygon."""
    centre = centre_for(section) if centre_for else centre_projections(section, name)
    slopes, intercepts = fit_radius_prior(training, label, name, kind, centre_for)
    radius = slopes * reach_beyond(section, centre) + intercepts

    if net_mask is not None and net_mask.sum() >= 20:
        net = radii_of(section, net_mask, centre, NET_PERCENTILE)
        radius = radius + np.clip(net - radius, -leash_um, leash_um)

    # THE FLOOR IS THE WHOLE POINT: with every radius at least this, the polygon contains a disc
    # of that size about the box centre, so it cannot be a sliver however badly the fit misses.
    return centre + np.maximum(radius, min_radius_um)


def tally(pool, label_value, name, offsets_for=None, carve="", keep_um=MIN_RADIUS_UM,
          region_for=None, punch=""):
    """Cells kept, area ratio, density and the per-section area ratios, held out by bird.

    `carve` is "" / "piece" / "line" / "both", so the two black tests can be attributed separately:
    a change in the mean cannot say which of them moved which section.

    `region_for(section, training)` returns a MASK directly, for rules that are not 16 radii about a
    centre -- the trapezoid off the band line, for one. Exactly one of the two must be given, so a
    new rule is graded by the same code as the old one rather than by a copy of it.
    """
    cells = kept = area = drawn = 0
    per_section, ratios, outside, ious = {}, {}, {}, {}

    for section in pool:
        training = [s for s in pool if s["bird"] != section["bird"]]
        region = (region_for(section, training) if region_for is not None else
                  rasterise(section["projected"], offsets_for(section, training),
                            section["tissue"], True))

        if carve in ("line", "both"):
            region = separation_cuts(region, section, name, keep_um=keep_um)

        if carve in ("piece", "both"):
            region = stop_at_black(region, section, name)

        truth = section["labels"] == label_value

        # PUNCHING THE BLACK OUT, which is a different use of `black_barrier` than every row above:
        # there it is a WALL that stops the far border, here it is a HOLE removed from the finished
        # region. Alice, 2026-08-19: *"if you identified the black separation in the image why didn't
        # you cut it out?"*.
        #
        # "region" punches the prediction only, and that number is NOT honest: density is
        # kept / (predicted area / HER area), so removing empty pixels from one side of that ratio and
        # not the other raises density for free. "both" removes it from her outline too, which is the
        # like-for-like comparison -- she draws straight over interior gaps
        # ([[regions-must-be-continuous-patches]]), so her area contains the same black.
        if punch in ("region", "both"):
            region = region & ~black_barrier(section["green"])

        if punch == "both":
            truth = truth & ~black_barrier(section["green"])

        area += int(region.sum())
        drawn += int(truth.sum())
        per_section[section["stem"]] = region
        ratios[section["stem"]] = region.sum() / max(int(truth.sum()), 1)
        # The share of the prediction that is NOT hers. An area ratio cannot see a displaced region:
        # YW113_RH_1-1-2 is x0.91 of her area with 44% of it outside her outline.
        outside[section["stem"]] = (region & ~truth).sum() / max(int(region.sum()), 1)
        # IoU, because area-ratio and outside-hers can move in OPPOSITE directions -- a smaller
        # region leaks less and misses more -- and then neither one alone can say which is better.
        ious[section["stem"]] = (region & truth).sum() / max(int((region | truth).sum()), 1)

        if len(section["rows"]):
            inside = truth[section["rows"], section["columns"]]
            cells += int(inside.sum())
            kept += int((inside & region[section["rows"], section["columns"]]).sum())

    share = kept / max(cells, 1)
    ratio = area / max(drawn, 1)

    return share, ratio, share / max(ratio, 1e-9), per_section, ratios, outside, ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=2, help="2 = CMM, 1 = NCM")
    parser.add_argument("--detail", action="store_true", help="per-section table for the cut floors")
    parser.add_argument("--floors", action="store_true", help="sweep the radius floor")
    args = parser.parse_args()

    saved = dict(np.load("shape_masks.npz"))
    name = "CMM" if args.label == 2 else "NCM"
    found = sections()
    pool = [s for s in found if (s["labels"] == args.label).sum() >= MIN_LABEL_PX]

    print(f"\n  {name}, {len(pool)} sections, held out by bird. 'worst' is the smallest area ratio\n"
          f"  any single section gets -- the collapse this is meant to make impossible.\n")
    print(f"  {'polygon':52s} {'kept':>5s} {'area':>6s} {'dens':>6s} {'out':>4s} {'IoU':>5s} "
          f"{'worst':>7s}  {'section':24s} {'leakiest':21s} she pointed at (area/outside)")

    rows = [
        ("review_polygon (absolute offsets)",
         lambda s, t: leashed(s, t, args.label, saved.get(f"{s['stem']}|{name}"), LEASH_UM)),
        ("radii, median prior, no net",
         lambda s, t: solid_offsets(s, t, args.label, name, None, kind="median")),
        ("radii, median prior + net leashed",
         lambda s, t: solid_offsets(s, t, args.label, name,
                                    saved.get(f"{s['stem']}|{name}"), kind="median")),
        ("radii, reach prior, no net",
         lambda s, t: solid_offsets(s, t, args.label, name, None, kind="reach")),
        ("radii, reach prior + net leashed",
         lambda s, t: solid_offsets(s, t, args.label, name,
                                    saved.get(f"{s['stem']}|{name}"), kind="reach")),
    ]

    # The two sections she pointed at, printed on every row: they are the ones where the polygon runs
    # past a black separation into hippocampus, and a mean cannot show whether that stopped.
    WATCH = ("YW113_RH_1-1-2", "Gre595_RH_3-1-4")

    # The floor on how close a black cut may come to the box centre, measured rather than argued:
    # 250 um keeps the anti-collapse disc intact, 0 lets the observed black have the last word.
    CARVES = [("", MIN_RADIUS_UM, ""),
              ("piece", MIN_RADIUS_UM, "  + the piece test"),
              ("line", MIN_RADIUS_UM, "  + line cuts, 250um floor"),
              ("line", 0.0, "  + line cuts, NO floor"),
              ("both", MIN_RADIUS_UM, "  + both, 250um floor"),
              ("both", 0.0, "  + both, NO floor")]

    for title, offsets_for in rows:
        sweep = CARVES if title.startswith("radii, reach prior +") else CARVES[:1]

        for carve, keep_um, suffix in sweep:
            share, ratio, density, _, ratios, out, ious = tally(pool, args.label, name, offsets_for,
                                                               carve, keep_um)
            worst = min(ratios, key=ratios.get)
            leakiest = max(out, key=out.get)
            watched = "  ".join(
                f"{stem.split('_')[0][:6]} x{ratios[next(k for k in ratios if stem in k)]:.2f}"
                f"/{out[next(k for k in out if stem in k)]:.0%}"
                for stem in WATCH)
            print(f"  {(title + suffix):52s} {share:5.0%} x{ratio:5.2f} "
                  f"{density:6.0%} {np.mean(list(out.values())):4.0%} "
                  f"{np.mean(list(ious.values())):5.3f} x{ratios[worst]:6.2f}  "
                  f"{worst[:22]:24s} {out[leakiest]:3.0%} {leakiest[:16]:18s} {watched}",
                  flush=True)

    if args.floors:
        # WHY SWEEP THIS AND NOT THE PRIOR. A 250 um radius floor guarantees the polygon contains a
        # 250 um disc, so it cannot be narrower than 500 um across. Her CMM is 402 / 444 / 445 um
        # across the band on YW113_RH_1-1-3, YW113_RH_1-1-2 and Wh175_LH_1-1-7 -- and those are
        # exactly her three worst sections for off-it (86% / 44% / 43%). The floor that makes
        # collapse impossible is what pushes the polygon off her thinnest regions.
        #
        # Alice's own rule for a floor, 2026-08-15: *"make the minimum length a little shorter than
        # my shortest length"*. Her narrowest CMM half-width is ~201 um, so the sweep brackets it.
        widths = {s["stem"]: float(np.ptp(s["across"][s["labels"] == args.label])) for s in pool}

        print(f"\n  radius floor swept. Her narrowest {name} across the band is "
              f"{min(widths.values()):.0f} um, so a floor above {min(widths.values()) / 2:.0f} um "
              f"cannot fit it.\n")
        print(f"  {'floor':>7s} {'kept':>5s} {'area':>6s} {'dens':>6s} {'out':>5s} {'IoU':>6s} "
              f"{'worst':>7s}  {'thin three: 1-1-3 / 1-1-2 / Wh175_LH_1-1-7 (area, off it)'}")

        for floor in (0.0, 100.0, 150.0, 200.0, 250.0, 350.0):
            def offsets_for(section, training, floor=floor):
                return solid_offsets(section, training, args.label, name,
                                     saved.get(f"{section['stem']}|{name}"),
                                     min_radius_um=floor)

            share, ratio, density, _, ratios, out, ious = tally(pool, args.label, name,
                                                               offsets_for, "piece")
            worst = min(ratios, key=ratios.get)
            thin = "  ".join(
                f"x{ratios[k]:.2f}/{out[k]:.0%}"
                for stem in ("YW113_RH_1-1-3", "YW113_RH_1-1-2", "Wh175_LH_1-1-7")
                for k in [next(k for k in ratios if stem in k)])
            print(f"  {floor:6.0f}u {share:5.0%} x{ratio:5.2f} {density:6.0%} "
                  f"{np.mean(list(out.values())):5.0%} {np.mean(list(ious.values())):6.3f} "
                  f"x{ratios[worst]:6.2f}  {thin}", flush=True)

    if args.detail:
        # PER SECTION, because two of the three numbers above are means over fifteen sections and a
        # mean cannot say whether a change helped one section and hurt another by the same amount.
        settings = [(c, k, s) for c, k, s in CARVES if c in ("", "line")]
        columns = [tally(pool, args.label, name, rows[-1][1], c, k)[4:6] for c, k, _ in settings]

        print(f"\n  per section: area ratio and the share of the prediction OUTSIDE her outline\n")
        print(f"  {'section':30s} " + "  ".join(f"{s.strip()[2:][:20]:>20s}" for _, _, s in settings))

        for stem in sorted(columns[0][0]):
            cells = "  ".join(f"{'x%.2f / %3.0f%% out' % (r[stem], 100 * o[stem]):>20s}"
                              for r, o in columns)
            print(f"  {stem[:28]:30s} " + cells)

    print(f"\n  find_band's box    CMM 73% x1.00 73% | NCM 83% x1.03 81%")
    print(f"  radius floor {MIN_RADIUS_UM:.0f} um, leash {LEASH_UM:g} um, {DIRECTIONS} directions.")


if __name__ == "__main__":
    main()
