"""Find the Field L band and split the tissue -- by looking, not by learning.

THE REPLACEMENT FOR train_band.py. That network learned a constant: the same vertical line in
all 14 sections (angle -90.0 +/- 0.3 deg against a true spread of 113 deg) with the band
collapsed to zero width. A fixed vertical line drawn with no model at all beat it. See
region-nets-collapse-to-a-constant-line.

WHY A WRITTEN RULE IS THE RIGHT TOOL HERE. ps6.compute_tissue is the precedent: outlining the
brain, the part of this problem that LOOKS hardest, is a blur and a threshold. A network earns
its keep when nobody can write down what makes an answer correct -- "is this blob a PS6-positive
cell" is genuinely like that, which is why the counting model works. "Where is the bright
stripe" is not like that. And a rule cannot collapse to a constant, because there is no loss for
it to shortcut.

THE SIX STEPS, and the measurement behind each:

  1. BRIGHT PATCH. Blur, then keep tissue pixels above the 75th percentile of tissue
     brightness, then take the largest connected component. 67% of that patch lands inside
     Alice's Field L gap (median; worst 36%), 6% on her NCM, 18% on her CMM.

  2. ANGLE from the patch's long axis, WEIGHTED BY HOW BRIGHT AND HOW GRANULAR each pixel is
     rather than by the bare mask. Median error 7.0 deg, 14/14 within 20 -- and it VARIES with
     the section instead of being constant, which is the whole point. Brightness weighting is
     what fixed OR99_RH, the low-contrast section (33.0 deg to 8.7): stitching seams are bright
     but THIN, so once brightness counts for something they stop out-voting the band. Granularity
     is what fixed YW113_RH_1-1-2 (44.6 deg to 3.8). See band_axis and texture.

  3. THE TWO BORDERS from where the patch starts and stops along the perpendicular.

  4. WHICH SIDE IS NCM from the dorsal marker Alice already clicks. Both regions are dark, so
     brightness cannot tell them apart -- one bit has to come from outside the image, and this
     bit is already being collected for the dNCM/vNCM split, so it costs her nothing.

  5. EVERY EXTENT IS A CONSTANT. The band's width, each region's depth and each region's length
     are fixed in microns; only position and angle come from the image. This is not a
     simplification, it is what the measurements said: the patch's width correlates +0.12 with
     Alice's real gap and its length +0.46 with NCM's, so those readings are mostly noise, and
     a noisy reading is worse than a good constant. It took CMM's reported density from 36% of
     the truth to 72%.

  6. WHERE ALONG THE BAND each box sits: NCM on the band's centre, CMM midway between that and
     the centroid of the tissue behind its border. This was the limiting error once the sizes
     were standardised, and it is where the obvious measurement pointed the wrong way -- see
     regions_for.

USE A PERCENTILE, NOT OTSU. Otsu put the cut out in the bright tail -- 0.899 on one section,
0.649 on the next, 0.432 on a third -- leaving "bright patches" of 4 to 10 PIXELS on 6 of the 14
sections. Those specks sat inside the gap and scored a meaningless 100%, which is what my first
reported figure was. A percentile is scale-free per section and guarantees a real region. The
25% is not arbitrary: Alice's gap IS a median 25% of tissue area.

KNOWN LIMITATION, stated rather than hidden. This gives both borders ONE shared angle, and
field_l_band.py measured that Alice's two borders are 2.6 deg apart (worst 5.4) and that forcing
them parallel costs cells -- NCM 99%->94%, CMM 100%->98%. So a shared angle is a real if small
cost, accepted here because the angle is the thing that was broken. Per-border angles are a
later refinement, not a reason to keep the network.

Usage:
  python find_band.py            # score every section on CELLS, against the null baseline
  python find_band.py --pixels   # also report the pixel figures, for continuity with training
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

import ps6
from ceiling_lines import DATA_PATH
from field_l_band import signed_distance, split_tissue
from score_region_cells import CANVAS_UM, cells_for, to_canvas
from score_rungs_cells import canvas_indices

# Blur before thresholding, in canvas pixels (8 um each, so 48 um). The band is a REGION, and
# without this the threshold would pick out individual bright cells instead -- exactly the
# distinction ps6.compute_tissue's own blur is making.
BLUR_SIGMA = 6.0

# Keep the brightest quarter of tissue. Matched to Alice's gap being a median 25% of tissue
# area, so the threshold is asking for a region the size of the thing being looked for.
BRIGHT_PERCENTILE = 75.0

# Where the band's edges sit in the patch's spread along the normal. Not 0 and 100: the patch
# has ragged ends and a few stray pixels would set the borders, so the extremes are trimmed.
EDGE_PERCENTILES = (10.0, 90.0)

# THE BAND IS A STANDARD WIDTH, not the width the patch happens to have. Measured across the 14
# sections, the patch's own width does NOT track Alice's real gap -- correlation +0.12, so it is
# noise -- while producing 1165 um bands that swallow CMM whole. Her gap is a median 550 um
# (296-937). Fixing the width therefore discards nothing and removes the failure mode.
BAND_WIDTH_UM = 550.0

# EACH REGION IS A STANDARD SIZE. Alice, 2026-08-11: "could you write a general rule where the
# field l region is pretty standard in size across all images to prevent some of the really big
# regions". Only the band's position and angle vary per section; all four extents are constants.
#
# DEPTH = how far the region runs away from its border. Without a depth limit each region is a
# half-plane and swallows everything on its side: CMM's border has a median 1183 um of tissue
# behind it while Alice's CMM only reaches 550 um, and that alone IS the 2.3x area.
#
# LENGTH = how far it runs along the band, centred on the band's own centre. The band's measured
# span is a worse guide than a constant -- it is a median 1220 um against NCM's real 1757 -- and
# it correlates only +0.46 with NCM's extent, so it is mostly noise, exactly like the width.
#
# INDEPENDENTLY CONFIRMED, which matters because these constants are the one place this file
# could overfit. band_atlas.py throws the boxes away: it re-expresses every pixel as (um along
# the band, um across it) and lets 14 sections VOTE on which spots are NCM. That free-form vote
# comes out RECTANGULAR, spanning +0 to +1300 um across and -1000 to +900 um along -- i.e. it
# rederives NCM_DEPTH_UM = 1300 and NCM_LENGTH_UM = 1900 exactly, with no box in the procedure.
# Held out one section at a time it also scores the same as these boxes (NCM 86% of cells at
# x1.05 area against 83% at x1.03) while needing Alice's outlines, which is why the atlas stays a
# check on the constants rather than a stage in the pipeline. See shape-atlas-needs-the-band-as-
# its-frame.
#
# HOW THESE FOUR NUMBERS WERE CHOSEN. Not
# by best average score: that criterion picked a 700 um CMM depth, which excluded all 5 counted
# cells of OR408_RH and so reported 0% of its density. They were chosen by the SECTION-TO-SECTION
# SWING in the reported-density factor, because a factor that is merely low is a constant scale
# the lab can live with, while a factor that lurches per section breaks the comparison the
# counting is for. Measured coefficient of variation:
#
#     depth 1150/700   NCM cv 19%   CMM cv 46%   <- best average, worst consistency
#     depth 1300/850   NCM cv 16%   CMM cv 21%   <- chosen
#
# For reference, the underlying measurements these land near: her region depths are NCM 645-1300
# (median 1012) and CMM -31-1377 (median 550); the deepest counted cell sits 1397 um into NCM and
# 868 um into CMM; her lengths are NCM 1213-2307 (median 1757) and CMM 765-1565 (median 1208).
#
# A fixed depth is as good as one scaled to the section's size -- fraction-of-tissue-reach and
# fixed-um land within 1-2 points of each other on every figure -- so the simpler rule wins.
NCM_DEPTH_UM = 1300.0
NCM_LENGTH_UM = 1900.0

# CMM IS SCALED TO THE SECTION, NOT FIXED. Alice, 2026-08-11, on the two sections where a fixed
# CMM came out 2.5x and 3.5x too big: "they actually are small and a other images will have small
# cmm's too". So a constant CMM size is wrong by construction and would stay wrong on new
# sections. It scales with how much tissue lies behind the CMM border -- her depth correlates
# +0.87 with that reach -- and expressing both extents as a fraction of it is measurably steadier
# than fixing them: depth cv 48% against 63%, length cv 16% against 19%.
#
# Chosen on her statement rather than on the score, and the score is worse in one respect: this
# costs 12 points of CMM cells kept (78% -> 66%) and raises the section-to-section swing from
# 21% to 27%, while taking CMM's area from x1.53 to x1.00 and her two small sections from
# x3.5/x2.5 to x1.8/x1.8. The steadiness of the fixed rule came from over-claiming everywhere by
# a similar amount, which flatters a swing measured on 7 sections and generalises badly.
#
# NCM stays fixed: scaling it the same way cost 9 points of cells for no gain in either figure.
CMM_DEPTH_FRACTION = 0.50
CMM_LENGTH_FRACTION = 0.46

# WHERE ALONG THE BAND EACH BOX IS CENTRED: 1.0 = on the band's own centre, 0.0 = on the centroid
# of the tissue behind that region's border. See regions_for for why these differ per region and
# why the obvious measurement pointed the wrong way.
NCM_CENTRE_BLEND = 1.0
CMM_CENTRE_BLEND = 0.5

# HOW HARD BRIGHTER PIXELS COUNT. The weight is (brightness - cut) to this power, so 0 would be
# the old binary mask and larger numbers hand the decision to a smaller, brighter core. Measured
# angle error, median / worst / within 20 deg: binary 13.5 / 37.3 / 11, ^1 12.5 / 43.1 / 12,
# ^2 13.1 / 44.6 / 13, ^3 10.2 / 44.6 / 13. Not pushed past 3: by then the effective vote sits on
# a few hundred pixels, and there is no measurement here that would notice it going too far.
WEIGHT_POWER = 3.0

# TEXTURE SCALES, in canvas pixels of 8 um. The fine sigma is the neighbourhood the local
# standard deviation is taken over -- 2 px = 16 um, about one cell, so it measures cell-scale
# speckle rather than a region's overall level. The coarse sigma then smooths that speckle map to
# region scale, matching BLUR_SIGMA so the two fields being multiplied together are comparably
# smooth. See texture().
TEXTURE_FINE_SIGMA = 2.0
TEXTURE_COARSE_SIGMA = 6.0

# SET THIS TO FALSE TO GO BACK TO BRIGHTNESS ALONE. It is a real switch, not a leftover, because
# the evidence for granularity is split: it clearly improves the ANGLE (median 10.2 deg -> 7.0,
# and 14/14 sections within 20 deg instead of 13/14) while being a WASH on cells (NCM 84%/83%
# density -> 83%/84%, CMM 69%/77% -> 73%/69%). It is on because three of the five sections whose
# angle it fixes have no counted cells, so the cell score cannot see its main benefit -- but that
# is a judgement, and a judgement should be easy to reverse. One known regression: Wh175_RH_1-1-9
# keeps 71% of its NCM cells instead of 84%.
USE_TEXTURE_FOR_ANGLE = True

# How far the brightness-weighted axis is allowed to move the unweighted one, in degrees. See
# band_axis -- a near-90 deg swing means the patch is round and has no axis, not that the
# weighting found something.
MAX_WEIGHT_SWING_DEG = 45.0

# A patch smaller than this is not a band. Guards the one failure worth guarding: if a section
# has no findable band, saying so beats returning a confident line through a speck.
MIN_PATCH_PIXELS = 2000


def texture(green):
    """How GRANULAR the image is at cell scale, smoothed to a region scale.

    Local standard deviation: the spread of brightness in a small neighbourhood, not its level.
    Field L is packed with small cells, so it is speckly as well as bright, and speckle is a
    different measurement from brightness -- which is exactly why it is worth having. Alice,
    2026-08-11, on the sections where the band is hard to see: "can you use other metrics of the
    image to generalize?".

    ON ITS OWN THIS IS A BETTER FIELD L DETECTOR AND A WORSE ANGLE. The granularity blob lands a
    median 81% inside Alice's Field L gap against brightness's 76%, and its WORST section is 72%
    against 40% -- it rescues YW113_RH_1-1-2, the section she asked about, from 40% to 96%. But
    its long axis is a median 25 deg off against brightness's 10 (worst 85), because it comes out
    a wider, blobbier shape: 922 um across against 718. Swapping brightness for it outright made
    everything worse (NCM density 82% -> 65%) and flipped the sides on one section. A patch can
    sit squarely inside the right region and still point the wrong way.

    Also tried and NOT kept: texture divided by brightness, to get a granularity measure that
    does not care how bright the tissue is. It collapses -- a median 43% of that blob lands on
    NCM and only 57% in the gap, worst 2%. So it is the raw speckle that matters, not the speckle
    relative to the level.
    """
    mean = gaussian_filter(green, TEXTURE_FINE_SIGMA)
    variance = np.maximum(gaussian_filter(green * green, TEXTURE_FINE_SIGMA) - mean * mean, 0.0)

    return gaussian_filter(np.sqrt(variance), TEXTURE_COARSE_SIGMA)


def patch_and_weight(field, tissue):
    """The largest connected blob of high values inside the tissue, and how high each pixel is.

    Returns (patch, weight) or None if the blob is too small to be a band.

    THE WEIGHT IS THE POINT OF THE SECOND RETURN VALUE. The mask alone says only "above the
    cut", which throws away how far above -- so a dim fringe pixel and the blazing core of
    Field L count the same. Alice, 2026-08-11: "i do think its clear there is a patch brighter
    than the rest that is the actual field l ... can you consider relative brightness having
    brighter things have more weight for where field l is?". weight = (value - cut) cubed, so
    the core sets the angle and the fringe barely votes. See band_axis for what it bought.
    """
    # The percentile is taken over TISSUE pixels only. Including background would put most of
    # the distribution at zero and drag the threshold down onto ordinary tissue.
    cut = np.percentile(field[tissue], BRIGHT_PERCENTILE)

    tags, count = label(tissue & (field > cut))

    if not count:
        return None

    sizes = np.bincount(tags.ravel())
    sizes[0] = 0
    patch = tags == int(sizes.argmax())

    if patch.sum() < MIN_PATCH_PIXELS:
        return None

    weight = np.zeros_like(field, dtype=float)
    weight[patch] = np.maximum(field[patch] - cut, 0.0) ** WEIGHT_POWER

    return patch, weight


def bright_patch(green, tissue):
    """Two Field L candidates: the bright one, and the bright-AND-granular one.

    Returns ((patch, weight), (patch, weight)) or None. The first is brightness alone and is
    what every BORDER is read off, because it is the one whose edges were measured against
    Alice's outlines. The second multiplies brightness by granularity and is used ONLY for the
    angle -- see band_axis, and see texture for why granularity cannot be trusted with more
    than that.

    WHY THE PRODUCT AND NOT THE INTERSECTION OF THE TWO BLOBS. The intersection scored about as
    well on the angle (median 7.6 deg against 7.0) but on one section it shrank below the
    minimum patch size and had to be abandoned. A product keeps every pixel and just ranks them,
    so it cannot run out of region.
    """
    smooth = gaussian_filter(green, BLUR_SIGMA)

    bright = patch_and_weight(smooth, tissue)

    if bright is None:
        return None

    if not USE_TEXTURE_FOR_ANGLE:
        return bright, bright

    combined = patch_and_weight(smooth * texture(green), tissue)

    return bright, (combined if combined is not None else bright)


def axis_from(weight):
    """The long axis of a weighted pixel cloud, as an angle in degrees.

    Split out from band_axis only so the weighted and unweighted answers can be compared
    against each other, which is what the 45 degree guard below needs.
    """
    ys, xs = np.nonzero(weight)
    share = weight[ys, xs] / weight[ys, xs].sum()

    mean_y = float((share * ys).sum())
    mean_x = float((share * xs).sum())
    dy, dx = ys - mean_y, xs - mean_x

    cov = np.array(
        [
            [float((share * dy * dy).sum()), float((share * dy * dx).sum())],
            [float((share * dy * dx).sum()), float((share * dx * dx).sum())],
        ]
    )

    values, vectors = np.linalg.eigh(cov)
    long_axis = vectors[:, int(np.argmax(values))]

    return float(np.degrees(np.arctan2(long_axis[0], long_axis[1])))


def refine(prior, candidate):
    """Accept a better-informed angle only if it agrees with the trusted one within 45 degrees.

    THE ONE RULE THIS FILE APPLIES TWICE, because both refinements have the same failure mode: a
    patch with no real long axis, whose principal axis is a coin flip between two perpendicular
    answers. Wh175_LH_1-2-1's bright patch is 1.16 times as long as it is wide -- effectively
    round -- and both refinements below tried to flip it by ~90 deg (19.9 deg of error becoming
    75.4). A near-90 deg jump is that coin landing the other way, not a correction, and no amount
    of extra evidence fixes a shape that has no direction. So the cruder answer stays.
    """
    swing = (candidate - prior + 90.0) % 180.0 - 90.0

    return candidate if abs(swing) <= MAX_WEIGHT_SWING_DEG else prior


def band_axis(bright, combined):
    """A unit normal perpendicular to the band's long axis. SIGN IS ARBITRARY here.

    The long axis is the direction the patch is most stretched along, which for a stripe is the
    stripe. Which WAY the normal points is meaningless until orient() fixes it, and relying on
    the sign numpy happens to return would be a bug waiting for the first new section.

    THREE ANSWERS, EACH ALLOWED TO CORRECT THE ONE BEFORE IT BY AT MOST 45 DEG:

      1. the bright patch's SHAPE, every pixel counting the same          median error 13.5 deg
      2. ...reweighted by HOW BRIGHT each pixel is                                        10.2
      3. ...reweighted again by brightness TIMES GRANULARITY                               7.0

    and 11 / 13 / 14 of the 14 sections within 20 deg, worst 37.3 / 44.6 / 19.9.

    STEP 2 IS THE SEAM FIX. Alice, 2026-08-11: "i do think the seams are altering it a bit ...
    there are these vertical lines running down the image". Stitching seams are bright but THIN,
    so they carry few pixels and lose their vote once brightness counts for something, while an
    unweighted mask let them pull as hard as the band. OR99_RH, the low-contrast section she
    raised it about, went from 33.0 deg wrong to 8.7.

    STEP 3 IS THE HARD-SECTION FIX. Alice, same day: "can you use other metrics of the image to
    generalize?". Multiplying by granularity rescued YW113_RH_1-1-2 from 44.6 deg to 3.8 and made
    every section land within 20 deg for the first time.

    HONESTLY, STEP 3 DOES NOT SHOW UP IN THE CELL SCORES -- NCM 84%/83% density becomes 83%/84%,
    CMM 69%/77% becomes 73%/73%. It is kept anyway, because three of the five sections it helps
    most have no counted cells at all, so the cell metric cannot see the thing it fixes. That is
    a stated reason to trust the geometry over the score, not a claim the score improved.

    Tried and NOT kept: choosing between the two candidate axes by which one has a brightness
    cliff across it (bright strip, darker flanks). On the round section the cliff test preferred
    the wrong axis too, so it added machinery and fixed nothing.
    """
    patch, weight = bright

    shape_only = np.zeros_like(weight)
    shape_only[patch] = 1.0

    degrees = refine(axis_from(shape_only), axis_from(weight))
    degrees = refine(degrees, axis_from(combined[1]))

    radians = np.radians(degrees)

    long_axis = np.array([np.sin(radians), np.cos(radians)])
    normal = np.array([-long_axis[1], long_axis[0]])

    return normal / max(np.linalg.norm(normal), 1e-9)


def orient(normal, patch, dorsal, shape):
    """Flip the normal so it points at NCM, using the dorsal click.

    Both regions are dark, so no brightness rule can do this -- a threshold puts 100% of CMM on
    the same side as NCM in 13 of 14 sections. One bit must come from outside the image.

    Verified 14/14, and verified NON-DEGENERATELY: with the incoming sign left as numpy returned
    it, NCM happened to land on the same side in all 14 sections, so a rule that always answered
    "that side" would also have scored 14/14. Flipping the sign on alternate sections split the
    truth 7/7 and the dorsal rule still got every one. Without that check this function would
    have looked correct while doing nothing.
    """
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0])
    dorsal_signed = normal[0] * (dorsal[0] - centre[0]) + normal[1] * (dorsal[1] - centre[1])

    signed = signed_distance(normal, shape)
    middle = float(np.mean(np.percentile(signed[patch], EDGE_PERCENTILES)))

    return normal if dorsal_signed > middle else -normal


def is_left_hemisphere(name):
    """True if this section is a LEFT hemisphere, and so must be mirrored into the frame.

    ONE place, because getting it wrong is silent and total: an unmirrored LH section trains as a
    mirror-image brain, and nothing downstream complains. The old test was `"_LH_" in stem`, which
    misses `Gre595_LH-1-1-10` -- a hyphen instead of an underscore -- and would have quietly
    mistrained the first file Alice was asked to add. So match the LH/RH token however it is
    punctuated, and refuse to guess when neither appears rather than defaulting to right.

    Note the section numbers in these names ("1-1-7") are a naming convention only: Alice,
    2026-08-12, "the 1-1-7 is just a naming convention it doesnt mean the image is mirrored".
    Two sections sharing those digits are NOT a matched pair and NOT the same level.
    """
    import re

    stem = Path(name).stem

    if re.search(r"[_\-]LH([_\-]|$)", stem):
        return True

    if re.search(r"[_\-]RH([_\-]|$)", stem):
        return False

    raise ValueError(
        f"{stem!r} says neither LH nor RH, so the hemisphere cannot be determined. "
        f"Rename it with an _LH_ or _RH_ token."
    )


def dorsal_on_canvas(image_name, pixel_size_um, scale_um):
    """The dorsal click in canvas coordinates, or None if it was never made.

    The same single scale factor score_region_cells.to_canvas uses -- prepare_regions.py
    resamples to 8 um/px and pads TOP-LEFT, so there is no offset to undo. Getting this wrong
    is quiet and total: my first attempt divided by the section height instead, which put every
    dorsal point thousands of pixels off a 480-pixel canvas and made the side rule score 1/14.
    """
    hits = list(Path(".").rglob(image_name))

    if not hits:
        return None

    point = ps6.load_dorsal_point(ps6.dorsal_path(hits[0]))

    if point is None:
        return None

    return np.asarray(point, dtype=float) * (pixel_size_um / scale_um)


def find_band(green, tissue, dorsal):
    """The whole finder: image in, band dict out, in field_l_band's format.

    Returns None if no band was found, rather than a guess. A refusal is actionable -- the
    person draws that one section by hand -- while a confident wrong line silently corrupts a
    density.
    """
    found = bright_patch(green, tissue)

    if found is None or dorsal is None:
        return None

    bright, combined = found
    patch = bright[0]

    # EVERY BORDER COMES OFF THE BRIGHTNESS PATCH, not the granularity-boosted one -- only the
    # ANGLE uses granularity. Reading the borders off the combined patch instead cost 6 points of
    # NCM cells, 12 of CMM, and put 13% of CMM's cells on the wrong side. See texture().
    normal = orient(band_axis(bright, combined), patch, dorsal, tissue.shape)

    signed = signed_distance(normal, tissue.shape)

    # ANCHOR AT THE NCM-FACING EDGE and step back a standard width -- do not use both of the
    # patch's own edges. Measured against Alice's borders along this same normal, the NCM edge
    # lands within a median 92 um while the CMM edge is off by 181 um and systematically pushed
    # 181 um INTO CMM (worst -729). Same asymmetry as the 6%-vs-18% patch leak, so keep the edge
    # that is measured and derive the one that is not.
    #
    # THE REASON THIS COMMENT USED TO GIVE WAS WRONG. It said CMM has no brightness cliff. Alice:
    # *"how is it different than ncm? it looks the same"* -- and diag_generalize_borders.py agrees
    # with her: the step across her CMM border is +0.183 of the tissue median against +0.178 across
    # her NCM border, positive on 12/15 and 13/15. The cliff is the SAME SIZE on both sides. What
    # differs is the base LEVEL beyond it, 1.64x the tissue median past CMM against 1.37x past NCM,
    # so an ABSOLUTE threshold does not stop at an equally large step that happens above its cutoff
    # and the blob keeps growing. The blindness is in the thresholding, not in the image -- and a
    # fitted depth+brightness read of the same pixels gains 22 um on this border, unanimous across
    # all 7 birds ([[cmm-side-of-field-l-has-no-cue]]).
    high = float(np.percentile(signed[patch], EDGE_PERCENTILES[1]))
    low = high - BAND_WIDTH_UM / CANVAS_UM

    # How far the band RUNS, on the perpendicular axis. Carried in the same dict so the end
    # cuts and the borders can never come from different fits of the same patch.
    along = signed_distance(np.array([-normal[1], normal[0]]), tissue.shape)
    span_low, span_high = np.percentile(along[patch], EDGE_PERCENTILES)

    # The normal points at NCM, so NCM is beyond the FAR edge and CMM is behind the near one.
    # Both borders share this normal; see the docstring on what that costs.
    return {
        "normal_y": float(normal[0]),
        "normal_x": float(normal[1]),
        "ncm_offset": float(high),
        "cmm_normal_y": float(normal[0]),
        "cmm_normal_x": float(normal[1]),
        "cmm_offset": float(low),
        "span_low": float(span_low),
        "span_high": float(span_high),
    }


def along_coordinate(band, shape):
    """Distance ALONG the band, perpendicular to its normal, from the canvas centre.

    This is the axis the end cuts live on. Measured from the centre for the same reason
    field_l_band.signed_distance is: a corner-referenced coordinate swings under rotation.
    """
    normal = np.array([band["normal_y"], band["normal_x"]])

    return signed_distance(np.array([-normal[1], normal[0]]), shape)


def regions_for(band, tissue):
    """The two regions as BOXES of standard size, not half-planes.

    Four bounds per region, and only the first is read off this section:

      * the band-facing border  -- from the bright patch, the one edge that is measured well
      * the depth limit         -- a constant behind that border
      * the two ends            -- a constant length, centred on the TISSUE behind that border

    So the section decides WHERE each region sits and at what ANGLE; it does not decide how big
    the region is. That is the point of the standard sizes: the parts of the geometry the image
    reports reliably are used, and the parts it reports badly are replaced by constants.

    Filled and reduced to the largest blob, as ceiling_lines.reconstructions does: a region cut
    by a dark fissure would otherwise come back as two pieces, and the smaller piece is usually
    on the wrong side of the fissure entirely.
    """
    from scipy.ndimage import binary_fill_holes

    from ceiling_lines import largest_blob

    signed = signed_distance(np.array([band["normal_y"], band["normal_x"]]), tissue.shape)
    along = along_coordinate(band, tissue.shape)

    band_centre = (band["span_low"] + band["span_high"]) / 2.0

    regions = {}

    # The normal points at NCM, so NCM lies ABOVE ncm_offset and CMM BELOW cmm_offset.
    for name, offset, sign in (
        ("NCM", band["ncm_offset"], +1.0),
        ("CMM", band["cmm_offset"], -1.0),
    ):
        near = (signed > offset) if name == "NCM" else (signed < offset)
        behind = tissue & near

        if not behind.any():
            regions[name] = behind
            continue

        if name == "NCM":
            depth = NCM_DEPTH_UM / CANVAS_UM
            length = NCM_LENGTH_UM / CANVAS_UM
        else:
            # How much tissue this section actually offers behind the border, in both
            # directions. Measured on the tissue mask, so it costs nothing and cannot be
            # circular -- it never looks at Alice's outlines.
            reach = abs(signed[behind].min() - offset)
            span = along[behind].max() - along[behind].min()

            depth = CMM_DEPTH_FRACTION * reach
            length = CMM_LENGTH_FRACTION * span

        # WHERE ALONG THE BAND THE BOX SITS -- the limiting error once the sizes were
        # standardised, because a correctly sized box parked at the wrong end of the region is
        # invisible in an area ratio. Two candidate anchors, mixed per region by CENTRE_BLEND:
        # the band's own centre, and the centroid of the tissue behind this region's border.
        #
        # THE PROXY AND THE REAL METRIC DISAGREED HERE, and the real metric won. Against the
        # geometric centre of Alice's outlines the tissue centroid is clearly the better anchor
        # -- median absolute deviation 104 um against the band centre's 236 for CMM -- so I
        # switched both regions to it. Scored on CELLS it was WORSE: NCM 84% -> 81% kept, CMM
        # 67% -> 66%. Predicting where the middle of a region is is not the same job as covering
        # the cells inside it, which is the same trap as grade-regions-on-cells-not-area.
        #
        # So NCM keeps the band centre outright, and CMM takes the midpoint of the two. CMM's
        # blend was chosen on a BROAD PLATEAU -- 0.25, 0.50 and 0.75 all score 76-77% density
        # against 76% for the band centre alone -- so the midpoint is a safe pick rather than a
        # fitted one, and the whole gain is 3 points of cells. A sharp optimum here would have
        # been a reason to distrust it, on 7 sections.
        blend = NCM_CENTRE_BLEND if name == "NCM" else CMM_CENTRE_BLEND
        centre = blend * band_centre + (1.0 - blend) * float(along[behind].mean())

        far = (signed * sign) < (offset * sign + depth)
        ends = (along >= centre - length / 2.0) & (along <= centre + length / 2.0)

        regions[name] = largest_blob(binary_fill_holes(behind & far & ends))

    return regions


def vertical_baseline():
    """A fixed vertical line, no model, no image. THE NUMBER EVERY RESULT IS QUOTED AGAINST.

    Exists because "fraction of pixels on the correct side" has a base rate near 100% for NCM,
    and without this line beside it a useless model reads as a good one -- which is exactly how
    BandNet's 90% got reported as progress.
    """
    return {
        "normal_y": 0.0,
        "normal_x": -1.0,
        "ncm_offset": 0.0,
        "cmm_normal_y": 0.0,
        "cmm_normal_x": -1.0,
        "cmm_offset": 0.0,
        # Symmetric about zero, so regions_for centres the standard box on the canvas centre.
        # The ruler gets exactly the same standard sizes the found band gets, which is the
        # point: the standard-size rules must not take credit for what a ruler could have done.
        "span_low": -1e9,
        "span_high": 1e9,
    }


def score(sides, labels, ys, xs):
    """Counts per region for one section, given the two region masks. Counts, not fractions.

    Fractions get formed later over pooled totals: averaging per-section percentages gives a
    7-cell section the same weight as a 51-cell one, and these sections differ that much.

    `other` and `unlabelled` are kept apart on purpose. A cell Alice put in the OTHER region is
    a real side error; a cell she put in neither is reach into unlabelled brain, which inflates
    area without being a side error. Pooling them makes a permissive half-plane look as bad as a
    misplaced one.
    """
    values = {"NCM": 1, "CMM": 2}

    rows = []

    for name, value in values.items():
        drawn = labels == value
        other = labels == values["CMM" if name == "NCM" else "NCM"]

        if not drawn.any():
            continue

        in_drawn = drawn[ys, xs]
        in_other = other[ys, xs]
        inside = sides[name][ys, xs]

        rows.append(
            {
                "region": name,
                "cells": int(in_drawn.sum()),
                "kept": int((in_drawn & inside).sum()),
                "other": int((inside & in_other).sum()),
                "unlabelled": int((inside & ~in_drawn & ~in_other).sum()),
                "area_ratio": sides[name].sum() / max(drawn.sum(), 1),
                "pixels_right": (sides[name] & drawn).sum() / max(drawn.sum(), 1),
            }
        )

    return rows


def main():
    show_pixels = "--pixels" in sys.argv

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    found = {"found": [], "baseline": [], "ruler+box": []}
    refused = []

    print(f"{'section':30s} {'cells':>6s}   {'FOUND kept / side / area':32s} {'baseline kept':>14s}")

    for index, (stem, name) in enumerate(zip(stems, images)):
        green = data["inputs"][index, 0]
        tissue = data["inputs"][index, 1] > 0.5
        labels = data["labels"][index]

        pixel_size = float(data["pixel_sizes"][index])
        points, source = cells_for(name)

        if not len(points):
            print(f"{stem[:30]:30s} no cells to score")
            continue

        ys, xs = canvas_indices(points, pixel_size, tissue.shape)

        band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size, scale_um))

        if band is None:
            refused.append(stem)
            print(f"{stem[:30]:30s} {len(points):6d}   REFUSED -- no band found")
            continue

        rows = score(regions_for(band, tissue), labels, ys, xs)

        ncm_side, cmm_side = split_tissue(vertical_baseline(), tissue)
        base = score({"NCM": ncm_side, "CMM": cmm_side}, labels, ys, xs)
        boxed = score(regions_for(vertical_baseline(), tissue), labels, ys, xs)

        found["found"].extend(rows)
        found["baseline"].extend(base)
        found["ruler+box"].extend(boxed)

        summary = "  ".join(
            f"{r['region']} {r['kept'] / max(r['cells'], 1):.0%}/{r['other'] / max(r['cells'], 1):.0%}"
            f"/x{r['area_ratio']:.1f}"
            for r in rows
        )
        base_summary = "  ".join(
            f"{r['kept'] / max(r['cells'], 1):.0%}" for r in base
        )
        print(f"{stem[:30]:30s} {len(points):6d}   {summary:32s} {base_summary:>14s}")

    print("\n" + "=" * 78)
    print("POOLED OVER ALL CELLS -- 'kept' is the number that matters: the fraction of the")
    print("cells Alice put in a region that the found region also contains.\n")
    print(
        f"{'':11s} {'region':6s} {'cells':>7s} {'kept':>7s} {'wrong side':>11s} "
        f"{'unlabelled':>11s} {'area':>7s} {'density':>8s}"
    )

    for which in ("found", "ruler+box", "baseline"):
        for region in ("NCM", "CMM"):
            rows = [r for r in found[which] if r["region"] == region]

            if not rows:
                continue

            cells = sum(r["cells"] for r in rows)
            kept = sum(r["kept"] for r in rows) / max(cells, 1)
            area = float(np.mean([r["area_ratio"] for r in rows]))

            print(
                f"{('FOUND' if which == 'found' else which):11s} {region:6s} {cells:7d} "
                f"{kept:6.0%} "
                f"{sum(r['other'] for r in rows) / max(cells, 1):10.0%} "
                f"{sum(r['unlabelled'] for r in rows) / max(cells, 1):10.0%} "
                f"x{area:5.2f} {kept / max(area, 1e-9):7.0%}"
            )

    print(
        "\n'density' = kept / area: the fraction of the TRUE cells-per-mm2 this region would\n"
        "report. It is the number the lab actually uses, and it is the only one of these that\n"
        "punishes a region for being too big and too small at once."
    )
    print(
        "'ruler+box' = a fixed vertical line given the SAME standard depths and end cuts, so\n"
        "the standard-size rules cannot take credit for what a ruler could have done."
    )

    if show_pixels:
        print("\npixel figures, for continuity with train_band.py's log:")
        for which in ("found", "baseline"):
            for region in ("NCM", "CMM"):
                rows = [r for r in found[which] if r["region"] == region]
                if rows:
                    print(
                        f"  {which:9s} {region}: median "
                        f"{np.median([r['pixels_right'] for r in rows]):.0%} of drawn pixels "
                        f"on the right side"
                    )

    if refused:
        print(f"\nrefused {len(refused)} section(s) -- no band found, draw these by hand:")
        for stem in refused:
            print(f"  {stem}")


if __name__ == "__main__":
    main()
