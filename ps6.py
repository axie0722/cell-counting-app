"""Shared pieces of the PS6 counting pipeline.

The detector geometry was measured, not guessed: averaging the radial intensity
profile of the hand-clicked cells gives a dim center out to about 4 px, a bright
ring peaking near 11 px, and background past about 22 px.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import oaconvolve
from scipy.spatial import cKDTree
from skimage.draw import polygon2mask
from skimage.feature import peak_local_max
from skimage.transform import resize

# Regions to count in, drawn by hand. There is no anatomical counterstain in
# these sections -- one activity channel and no Nissl or DAPI -- so no algorithm
# can find the NCM border from intensity. Drawing them keeps the judgement
# visible and re-runnable instead of hidden inside a segmentation step.
REGION_NAMES = ["dNCM", "vNCM", "CMM"]

# One entry per annotated section. `region` fences off tissue that was not
# annotated thoroughly; None means the whole tissue area was reviewed.
#
# `quality` marks how much a section should be trusted as training data. A weak
# section is worth flagging rather than quietly dropping: including it can drag
# the decision boundary toward artifacts, and holding it out reads as "the model
# does not transfer" when the real problem is the tissue. Marking it lets that be
# measured either way.
#   "good"      full trust
#   "marginal"  usable, but excluded from held-out evaluation by default
#
# `name` is the BIRD, not the section, and several sections deliberately share one.
# train_cnn.py holds out by name, so two sections of the same brain must never land
# on opposite sides of that split: they share a staining batch, a scan session and
# an animal, so holding one out while training on the other measures memorisation
# and reports it as generalization. Sections are told apart by `image`.
#
# `annotations` is None for a section with no clicked cells. That is the normal
# state for anything uploaded after 2026-08-03: cell labels now come from the
# exhaustive grid windows in blind_grid.py, because whole-section clicking is what
# poisoned the archive's negatives (38% of patches on a real cell read as negative).
# Such a section still contributes regions, area for grid windows, and counts.
BIRDS = [
    {
        "name": "YW113",
        "image": "YW113_RH_1-1-7_NCM_Slide 3.TIF",
        "annotations": "YW113_RH_1-1-7_NCM_Slide 3 cell centers.csv",
        "region": "cutoff.csv",
        # Two cells found during the exhaustive-box precision check.
        "extra_annotations": "YW113 exhaustive box y3850 x500.csv",
        "quality": "good",
    },
    {
        "name": "OR99",
        "image": "OR99_LH_1-1-4_NCM_Slide 3.TIF",
        "annotations": "OR99_LH_1-1-4_NCM_Slide 3 cell centers.csv",
        "region": None,
        "quality": "good",
    },
    {
        "name": "Gre595",
        "image": "Gre595_RH_3-1-4_NCM_Slide 1.TIF",
        "annotations": "Gre595_RH_3-1-4_NCM_Slide 1 cell centers.csv",
        "region": "gre_cutoff.csv",
        "quality": "good",
        # The 89 assisted clicks this section carried before Alice's blank-canvas pass
        # on 2026-08-06, which replaced them with 62. 44 of the 89 have no fresh click
        # within the match radius: 22 fall inside her training cutoff and are dropped
        # anyway, but 22 do not, and those would otherwise become confident NEGATIVES
        # -- the exact poisoning extract_windows.py exists to avoid. They are not
        # promoted to positives either, since assisted clicks run about 38% precise
        # (see model-precision-is-38pct-not-74pct). Held out instead, like duplicates.
        "disputed_annotations": "Gre595_RH_3-1-4_NCM_Slide 1 cell centers.archive.csv",
    },
    {
        "name": "OR408",
        "image": "OR408_RH_1-2-1_NCM_Slide 2.TIF",
        "annotations": "OR408_RH_1-2-1_NCM_Slide 2 cell centers.csv",
        "region": None,
        "quality": "good",
    },
    {
        # Annotated with model assistance. Flagged as poorer tissue, which is why
        # it was skipped initially.
        "name": "Purp30",
        "image": "birds/Purp30_LH_1-1-8_NCM_Slide 1.TIF",
        "annotations": "birds/Purp30_LH_1-1-8_NCM_Slide 1 cell centers.csv",
        "region": None,
        "quality": "marginal",
    },
    # Uploaded 2026-08-04. No cell labels by design -- see the note above. Regions
    # still need drawing before any of them can be counted or sampled.
    #
    # Three of these are further YW113 sections and so carry the existing YW113 name.
    # That does NOT make them redundant: Alice's point (2026-08-04) is that slices
    # from one bird look substantially different from each other, and different
    # hemispheres more so, since rostrocaudal level changes region shape and size
    # while section handling changes staining. They add real appearance variation and
    # labelling area. What they cannot do is serve as a held-out test, because they
    # share an animal and a staining batch with a section the model trained on --
    # hence the shared name, which keeps them on one side of the holdout split.
    # LBlu59 and Wh175 are the new animals, taking the count from 5 birds to 7 of 8.
    {
        "name": "LBlu59",
        "image": "birds/LBlu59_RH_3-1-8_NCM_Slide 3_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    # The one hemisphere in the project the CNN has never seen from a bird it HAS seen.
    # Measured 2026-08-08 (compare_hemispheres.py): LH vs RH within a bird differs about
    # as much as two different birds do -- median 1.30 against 1.05 between birds -- so
    # this is a real holdout rather than a near-duplicate of the RH section above, which
    # was Alice's point ("they look pretty different from each other even on the same
    # bird"). Keeps the LBlu59 name regardless: it is the same animal and the same
    # staining batch, so a bird-level holdout split must not put it on the other side.
    {
        "name": "LBlu59",
        "image": "birds/LBlu59_LH_3-1-7_NCM_Slide 1_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "Wh175",
        "image": "birds/Wh175_LH_1-1-7_NCM_Slide 3.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "Wh175",
        "image": "birds/Wh175_LH_1-2-1_NCM_Slide 4_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "Wh175",
        "image": "birds/Wh175_RH_1-1-9_NCM_Slide 1.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "YW113",
        "image": "birds/YW113_RH_1-1-2_NCM_Slide 3_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "YW113",
        "image": "birds/YW113_RH_1-1-3_NCM_Slide 3_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        "name": "YW113",
        "image": "birds/YW113_RH_1-1-10_NCM_Slide 3_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        # Registered 2026-08-06, the Gre595 section that had NOT been annotated. The
        # right-hemisphere section above already carried 89 clicks; this is the one
        # Alice meant. Shares the Gre595 name so both stay on one side of the holdout
        # split, per the note above.
        "name": "Gre595",
        "image": "birds/Gre595_LH-1-1-10_Slide 1_TD_p00_0_A01f00d1.TIF",
        "annotations": "birds/Gre595_LH-1-1-10_Slide 1_TD_p00_0_A01f00d1 cell centers.csv",
        "region": None,
        "quality": "good",
    },
    {
        # Registered 2026-08-09. The second unseen hemisphere of a trained bird:
        # OR99_LH_1-1-4 is in training, this is its RH counterpart, so it repeats the
        # LBlu59 LH setup on a different animal. That matters because the unseen-
        # hemisphere cost (56.7% recall against 66.1%, precision holding near 90%)
        # currently rests on one section, and one section cannot separate "hemispheres
        # generalize poorly" from "that particular slice was hard".
        #
        # Keeps the OR99 name: same animal and staining batch as the trained LH
        # section, so a bird-level holdout must not put them on opposite sides.
        "name": "OR99",
        "image": "birds/OR99_RH_1-1-1_NCM_Slide 1_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
    {
        # Registered 2026-08-16, when Alice drew its hippocampus and `prepare_regions`
        # refused to cache it: "is not registered in ps6.BIRDS, so its bird is unknown
        # and it cannot be held out safely". That refusal is the point -- see
        # one-name-per-bird-not-per-section.
        #
        # A THIRD OR99 section, and it adds no fold. Same animal as OR99_LH_1-1-4 and
        # OR99_RH_1-1-1, and the same hemisphere as the latter, so all three carry the
        # OR99 name and move together through a bird-level holdout. What it adds is
        # outlines: a 15th NCM/CMM/HP example, which is worth having when seven birds is
        # the bottleneck (seven-birds-is-the-region-bottleneck) and no unlabelled image
        # can add a bird.
        #
        # `annotations` is None: no cells have been clicked on it, and it must not become
        # a source of confident negatives.
        "name": "OR99",
        "image": "birds/OR99_RH_1-1-4_NCM_Slide 1_TM_p00_0_A01f00d0.TIF",
        "annotations": None,
        "region": None,
        "quality": "good",
    },
]


def annotated_birds(include_marginal=True):
    """Birds whose annotation file exists, optionally excluding weak sections."""
    from pathlib import Path as _Path

    birds = []

    for bird in BIRDS:
        # `annotations` is None for sections registered without clicked cells, which
        # is the intended state for anything uploaded after 2026-08-03. Checked before
        # the path test because _Path(None) raises.
        if not bird["annotations"]:
            continue

        if not _Path(bird["annotations"]).exists():
            continue

        if not include_marginal and bird.get("quality") == "marginal":
            continue

        birds.append(bird)

    return birds

# Zone radii in pixels, as measured on the sections scanned at REFERENCE_PIXEL_SIZE_UM.
CENTER_RADIUS = 4
RING_INNER_RADIUS = 8
RING_OUTER_RADIUS = 15
BACKGROUND_INNER_RADIUS = 22
BACKGROUND_OUTER_RADIUS = 32

SMOOTHING_SIGMA = 1.5

TISSUE_SIGMA = 8

# An ABSOLUTE cutoff on raw green, so it silently means "0.10 for a section as bright as the ones this
# was tuned on". Measured across the 20 sections (check_tissue_threshold.py), it lands at the 13th
# percentile of a bright section but the 35th of Gre595_RH_3-1-4, the dimmest -- where it discards
# 15.9% of Alice's own drawn NCM as "not tissue". Alice, 2026-08-14: *"the areas where the dector
# stops detecting look really dark almost like background"*.
#
# LEFT AT 0.10 ON PURPOSE. This has 25 callers including extract_patches.py, count_cells.py and
# score_blind_grid.py; lowering it here would change cell detection and invalidate the blind grid,
# which is the only honest metric in the project. The region path passes its own value instead --
# see REGION_TISSUE_THRESHOLD in prepare_regions.py.
TISSUE_THRESHOLD = 0.10

# The pixel size the radii above were measured at, and the size most sections were
# scanned at. Used for area and density, and as the baseline for CANDIDATES_PER_MM2.
#
# Deliberately NOT used to rescale the detector radii: measurement showed apparent
# cell size does not follow it (see the Geometry docstring). Scale tolerance comes
# from training augmentation instead.
REFERENCE_PIXEL_SIZE_UM = 0.3088427408547889

# Candidates kept per megapixel of reviewed tissue.
#
# Neither an absolute threshold nor a score percentile transfers across
# sections. An absolute threshold fails on brightness: Gre595 is dim enough that
# a value tuned on YW113 drops it to 39% recall. A percentile fails on tail
# shape: OR99 has bright fibers and vessels that fill its top percentiles, so
# p98.5 lands at 0.058 while the median score at a real OR99 cell is 0.038 --
# the threshold sat above most of its own cells, giving 18% recall.
#
# Fixing the count per megapixel sidesteps both. Measured recall at 15 px:
#   150/Mpx -> YW113 93%, OR99 23%, Gre595 93%
#   350/Mpx -> YW113 96%, OR99 95%, Gre595 98%
#   700/Mpx -> YW113 96%, OR99 96%, Gre595 98%
# Recall plateaus by 350 while the patch count keeps growing, so this is the
# knee of the curve.
CANDIDATES_PER_MEGAPIXEL = 350

# The same budget per unit of real tissue, which is what transfers when sections
# are scanned at different magnifications: a megapixel of coarse scan covers more
# tissue than a megapixel of fine scan and should get proportionally more
# candidates. 350 per Mpx at the reference pixel size works out to this.
CANDIDATES_PER_MM2 = CANDIDATES_PER_MEGAPIXEL / (REFERENCE_PIXEL_SIZE_UM / 1000.0) ** 2 / 1e6

# Peaks below this percentile of the score are never considered, which keeps
# peak_local_max from enumerating the whole background before the top-N cut.
FLOOR_PERCENTILE = 80.0

# The two closest clicked cells in YW113 are 29 px apart.
MIN_DISTANCE = 12

# A candidate this close to a clicked center is taken to be that cell.
MATCH_RADIUS = 15

# A cell centered just outside the reviewed region can pull a candidate just
# inside it, and that candidate was never reviewed.
BOUNDARY_MARGIN = 20

# Cells are 24-30 px across, so 64x64 holds a cell plus enough surround for the
# CNN to judge its background. This is what the CNN consumes.
PATCH_HALF_WIDTH = 32

# Patches are stored wider than the CNN needs, so training can crop to a range of
# widths and resize to 64 -- teaching scale tolerance instead of assuming it.
#
# Cells vary in apparent size between sections by about 10% (measured ring radii
# 11.56-12.68 px), and that variation does not track the recorded pixel size, so it
# cannot be corrected for. Without augmentation the CNN is a knife edge: a 6.7%
# size change drops confident cells from 45% to 4%.
#
#   apparent scale   no augmentation   with augmentation
#     1.000 native         45%               55%
#     1.067                 4%               49%
#     1.143                 1%               39%
#     0.889                11%               49%
#
# The cost is real -- same-scale 3-bird mean AP falls 0.370 to 0.296 -- but AP is
# only ever measured at each bird's own scale, so it cannot credit robustness. What
# decides usability is OR408: 18 confident proposals without augmentation, 249 with.
# Below Purp30's ~2.1/mm2 yardstick a section is not worth opening.
#
# 40 gives an 80x80 store, so crops of 56-72 px are available, spanning apparent
# scales 0.889-1.143 -- wider than the 1.097 the sections actually exhibit.
PATCH_CONTEXT_HALF_WIDTH = 40

# Widest and narrowest crop, in output pixels, used as training augmentation.
MIN_TRAIN_CROP = 56
MAX_TRAIN_CROP = 72


def load_green(path):
    """Return (full image, green channel as float32 in 0-1).

    All PS6 signal is in the green channel; red and blue are identically 0.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        image = tif.series[0].asarray()

    if image.ndim == 3 and image.shape[-1] >= 3:
        return image, image[..., 1].astype(np.float32) / 255.0

    if image.ndim == 2:
        return image, image.astype(np.float32) / np.iinfo(image.dtype).max

    raise ValueError(f"Unexpected image shape: {image.shape}")


class Geometry:
    """Detector distances in pixels, and the pixel size used for area.

    Detector radii are NOT scaled by the metadata pixel size, though an earlier
    version did scale them. Measuring the ring radius directly from the pixels
    (check_cell_size.py) showed the metadata does not predict apparent cell size:

        section   um/px   ring radius   measured   metadata predicts
        YW113    0.3088     11.56 px      1.000          1.000
        OR99     0.3088     12.68 px      1.097          1.000
        Gre595   0.3088     12.31 px      1.065          1.000
        OR408    0.3395     12.66 px      1.095          0.910

    OR408's cells are 9.5% LARGER than YW113's, where its pixel size predicted 9%
    smaller -- wrong in direction, not just magnitude. And the three sections
    sharing an identical 0.3088 spread over 1.000-1.097, as wide as the effect the
    scaling claimed to correct. Section thickness, staining and focus dominate;
    PhysicalSizeX carries no usable signal about apparent size. Scaling by it took
    OR408 from 214 confident proposals to 6.

    Pixel size is still used for area and density, which is what it does measure.
    Scale tolerance is handled where it belongs, by training augmentation over a
    range of apparent sizes (see train_cnn.scale_jitter).
    """

    def __init__(self, pixel_size=None):
        self.pixel_size = pixel_size or REFERENCE_PIXEL_SIZE_UM

        # Pinned to 1: see the class docstring. Kept as an attribute so the call
        # sites stay honest about the fact that a scale exists conceptually.
        self.scale = 1.0

        # Radii keep sub-pixel precision: annulus_kernel rasterises them, and
        # rounding here would quantise the zones twice.
        self.center_radius = CENTER_RADIUS * self.scale
        self.ring_inner_radius = RING_INNER_RADIUS * self.scale
        self.ring_outer_radius = RING_OUTER_RADIUS * self.scale
        self.background_inner_radius = BACKGROUND_INNER_RADIUS * self.scale
        self.background_outer_radius = BACKGROUND_OUTER_RADIUS * self.scale

        self.smoothing_sigma = SMOOTHING_SIGMA * self.scale
        self.tissue_sigma = TISSUE_SIGMA * self.scale

        # Distances used for counting and matching must be whole pixels.
        self.min_distance = max(1, int(round(MIN_DISTANCE * self.scale)))
        self.match_radius = MATCH_RADIUS * self.scale
        self.boundary_margin = BOUNDARY_MARGIN * self.scale

        self.patch_half_width = PATCH_HALF_WIDTH

        # The wider patch actually stored. A candidate needs this much room, not
        # just patch_half_width, or the stored patch runs off the image.
        self.patch_context_half_width = PATCH_CONTEXT_HALF_WIDTH

    @property
    def is_reference(self):
        """True when this section was scanned at the reference pixel size."""
        return abs(self.pixel_size - REFERENCE_PIXEL_SIZE_UM) < 1e-9

    def candidates_wanted(self, reviewed_pixels):
        """Candidate budget for an area, held constant per unit of real tissue.

        Unlike the detector radii, this genuinely should follow pixel size: a
        megapixel of coarse scan covers more tissue and holds proportionally more
        cells. Area is what PhysicalSizeX measures reliably.
        """
        mm2 = reviewed_pixels * (self.pixel_size / 1000.0) ** 2

        return int(round(CANDIDATES_PER_MM2 * mm2))

    def describe(self):
        note = "reference" if self.is_reference else "radii unscaled by design"

        return f"{self.pixel_size:.4f} um/px ({note})"


def geometry_for(image_path):
    """Geometry for an image, read from its own metadata."""
    return Geometry(pixel_size_um(image_path))


def pixel_size_um(path):
    """Microns per pixel, from the OME-XML the scanner embedded.

    Needed because raw counts are not comparable between regions of different
    size; density is. Returns None when the tag is missing, so callers can report
    counts alone rather than inventing a scale.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        description = tif.pages[0].tags.get("ImageDescription")

        if description is None:
            return None

        match = re.search(r'PhysicalSizeX="([^"]+)"', str(description.value))

        if match:
            return float(match.group(1))

        # Fall back to the plain TIFF resolution tags.
        resolution = tif.pages[0].tags.get("XResolution")

        if resolution is None:
            return None

        numerator, denominator = resolution.value

        if not numerator:
            return None

        # ResolutionUnit 2 is inches, 3 is centimetres.
        unit = tif.pages[0].tags.get("ResolutionUnit")
        pixels_per_unit = numerator / denominator
        microns_per_unit = 25400.0 if getattr(unit, "value", 2) == 2 else 10000.0

        return microns_per_unit / pixels_per_unit


def sidecar_path(image_path, suffix):
    """Where a sidecar for `image_path` belongs: BESIDE THE IMAGE, not in the working
    directory.

    Every helper below goes through this, so the rule lives in one place.

    WHY THIS EXISTS. These helpers used to return a bare filename -- "<stem> regions.csv"
    with no folder -- which resolves against whatever directory the process happens to be
    started in. That worked only because everything was always run from the project
    folder. It meant the TIFs in birds/ had their outlines written to the top level, so the
    docstrings claiming sidecars sit "beside the image" were describing an intention the
    code did not implement, and no folder was self-contained: copying birds/ elsewhere
    copied the images and left every outline and count behind.

    THE COST OF THIS RULE: filename stems must be unique, because the stem is a section's
    only identity -- there is no database. Two images with the same stem in different
    folders are the same section as far as every name here is concerned, and each would
    claim its own copy of the sidecars. See app_state.find_sections, which resolves that
    by keeping the shallowest path when a stem appears more than once.
    """
    image = Path(image_path)

    # with_name rather than parent / name: it keeps whatever kind of path came in, and a
    # bare "X.TIF" (parent ".") stays bare instead of becoming "./X regions.csv" -- which
    # would be the same file but would not compare equal to the strings already on disk.
    return str(image.with_name(f"{image.stem}{suffix}"))


def region_path(image_path):
    """Where the hand-drawn ROIs for an image live."""
    return sidecar_path(image_path, " regions.csv")


def areas_path(image_path):
    """Where a section's cached region areas live.

    Here rather than in app_table.py because count_cells.py writes this file and the app
    reads it: two modules on opposite sides of the project agreeing on a filename is
    exactly what the other path helpers exist to prevent.
    """
    return sidecar_path(image_path, " region areas.csv")


def counts_path(image_path):
    """Where a section's kept cell coordinates go.

    count_cells.points_path() delegates here and app_state.sidecar_paths() reads it, so
    the writer and the reader cannot drift apart -- app_state used to rebuild this name by
    hand with a comment explaining why it could not import the writer.
    """
    return sidecar_path(image_path, " counted cells.csv")


def edits_path(image_path):
    """Where hand corrections to a count are recorded, one row per cell added or removed.

    APPEND-ONLY, AND NEVER READ BY THE COUNTER. show_counts.py rewrites the counted-cells
    file when Alice adds or deletes a cell, and that file only ever shows the CURRENT answer.
    This one keeps what changed and when, which is the part that cannot be recovered from the
    result: a deleted cell is a place the model was confidently wrong, and an added one is a
    place it missed. Those are the two most valuable kinds of training example there are, and
    they would otherwise exist only as an absence.

    Separate from counts_path's ".previous.csv" (see count_cells.save_previous) on purpose.
    That is one generation deep and exists to undo an accidental recount; this is the whole
    history and exists to be learned from.
    """
    return sidecar_path(image_path, " cell edits.csv")


# Column order for the cached area file, beside the path for the same reason: two writers
# now (count_cells.py during a count, app_table.py --areas as a backfill) and one reader.
AREA_COLUMNS = ["region", "area_mm2", "tissue_px"]


def training_cutoff_path(image_path):
    """Where a TRAINING-ONLY exclusion polygon lives, if one was drawn.

    Alice drew one on Gre595's RH section (2026-08-06) over tissue whose cells she
    judged unreliable, and was explicit about its scope: "by cutoff i mean just for
    training the model, like when u input the cells to train". So this excludes
    patches from the training set and nothing else -- count_cells.py still measures
    density over the full drawn regions, and the blind grid stays fully scoreable.

    Distinct from the per-bird `region` entry in BIRDS, which is an older cutoff LINE
    marking tissue that should not be looked at at all.
    """
    return sidecar_path(image_path, " cutoff.csv")


def load_training_cutoff(image_path, shape):
    """Mask of tissue to keep out of the TRAINING set, or None if none was drawn.

    Napari writes a Shapes layer with one row per vertex and an `index` column
    numbering the shapes, so several polygons can be drawn in one layer; they are
    unioned here. load_region() cannot read this: it takes every row as a single
    polygon, so two drawn shapes would join into one wrong outline, and
    load_region_vertices() groups on a `region` column a Shapes export does not have.

    Feed the result to label_candidates(excluded=...). Nothing else should use it --
    density and the blind grid stay measured over the full regions.
    """
    path = training_cutoff_path(image_path)

    if not Path(path).exists():
        return None

    table = pd.read_csv(path)

    if not len(table):
        return None

    # A layer with no `index` column is a single shape.
    grouper = table["index"] if "index" in table.columns else np.zeros(len(table))
    mask = np.zeros(shape, bool)
    shapes = 0

    for _, group in table.groupby(grouper, sort=False):
        vertices = group[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)

        if len(vertices) < 3:
            print(f"  WARNING: cutoff shape with {len(vertices)} vertices; skipping")
            continue

        mask |= polygon2mask(shape, vertices)
        shapes += 1

    if not shapes:
        return None

    print(
        f"  training cutoff: {shapes} polygon(s) covering "
        f"{mask.mean():.1%} of the image, excluded from training only"
    )

    return mask


def load_region_vertices(path):
    """Load named ROI polygons as {name: vertex array}, without rasterising.

    split_ncm needs the vertices, not the mask: the Field L axis comes from the
    length of a drawn edge, which rasterising throws away.
    """
    if not Path(path).exists():
        return {}

    table = pd.read_csv(path)

    return {
        str(name): group[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)
        for name, group in table.groupby("region", sort=False)
    }


def load_regions(path, shape):
    """Load named ROI polygons as {name: boolean mask}.

    The file stores one row per vertex with a `region` column, which keeps the
    polygons editable: when a boundary is redrawn, counting re-runs without
    anything else changing.
    """
    masks = {}

    for name, vertices in load_region_vertices(path).items():
        if len(vertices) < 3:
            print(f"  WARNING: {name} has {len(vertices)} vertices; skipping")
            continue

        masks[name] = polygon2mask(shape, vertices)

    return masks


def dorsal_path(image_path):
    """Where the single dorsal-side marker click lives."""
    return sidecar_path(image_path, " dorsal.csv")


def field_l_ends_path(image_path):
    """Where the two clicks marking where the Field L line should START and STOP live.

    Alice, 2026-08-14: *"i dont know if you want me to mark where the lines should end so you know?"*
    Yes -- because until now the target for the line's length was INFERRED, as the along extent of her
    NCM polygon's longest edge (field_l_direction). That proxy answers "how long did the polygon's Field
    L side turn out to be", which is not the same question as "how long should the line be": the polygon
    edge ends where she chose to close the shape, so a corner cut for convenience reads as an anatomical
    end. Two deliberate clicks are the quantity itself.

    Its own sidecar rather than two more rows in regions.csv: everything that reads regions.csv treats a
    region as a POLYGON, and a two-vertex "region" would be silently dropped by some readers
    (draw_regions skips shapes with fewer than 3 vertices) and misread as a degenerate area by others.
    """
    return sidecar_path(image_path, " field l ends.csv")


def load_field_l_ends(path):
    """The two clicks bounding the Field L line as a (2, 2) array of (row, column), or None.

    Order does not matter and is not relied on anywhere -- the pair is used as an unordered interval
    along the band, so "which end did she click first" is not a fact the pipeline can accidentally depend
    on.
    """
    if not Path(path).exists():
        return None

    table = pd.read_csv(path)

    if len(table) < 2:
        return None

    return table[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)[:2]


def load_dorsal_point(path):
    """The one click marking the dorsal (hippocampus) side, or None.

    Dorsal is identified anatomically -- the side nearest the hippocampus -- not
    as a fixed image direction, because sections are not always mounted upright.
    One click carries that information without assuming orientation.
    """
    if not Path(path).exists():
        return None

    table = pd.read_csv(path)

    if not len(table):
        return None

    return table[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)[0]


def field_l_direction(ncm_vertices, ratio_required=2.0):
    """Unit vector along the Field L border of a drawn NCM polygon.

    NCM is traced as many short segments following the curved tissue outline, plus
    one long straight run along the Field L boundary -- the anatomical landmark. So
    the longest edge IS the Field L line, with no extra drawing needed. Measured on
    the five drawn sections it is 5806-7382 px and 3.8x to 8.9x the second-longest,
    so the margin is comfortable.

    Raises when that margin is absent, since a Field L border traced as two or three
    segments would make "the longest edge" an arbitrary pick among them, and a
    silently arbitrary axis would rotate the dNCM/vNCM split.
    """
    vertices = np.asarray(ncm_vertices, dtype=np.float64)

    if len(vertices) < 3:
        raise ValueError(f"NCM polygon has only {len(vertices)} vertices.")

    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    order = np.argsort(-lengths)

    longest, runner_up = lengths[order[0]], lengths[order[1]]

    if longest < ratio_required * runner_up:
        raise ValueError(
            f"No clear Field L edge: longest NCM edge is {longest:.0f} px against "
            f"{runner_up:.0f} px for the next, a ratio of {longest / runner_up:.1f} "
            f"(need {ratio_required}). Field L was probably traced as several "
            f"segments; redraw it as one straight run."
        )

    return edges[order[0]] / longest


def split_axis(ncm_mask, dorsal_point, tissue=None, ncm_vertices=None):
    """The cut direction, its origin, and the offset at which NCM is halved.

    Separate from split_ncm so the cut can be drawn for confirmation without
    recomputing it a second, possibly different, way. Returns
    (axis, centroid, midpoint): `axis` is a unit vector pointing dorsally, and the
    cut is the line of points p with dot(p - centroid, axis) == midpoint.
    """
    inside = ncm_mask if tissue is None else (ncm_mask & tissue)
    rows, columns = np.nonzero(inside)

    if not len(rows):
        raise ValueError("NCM region is empty.")

    centroid = np.array([rows.mean(), columns.mean()])

    toward_dorsal = np.asarray(dorsal_point, dtype=np.float64) - centroid
    length = np.hypot(*toward_dorsal)

    if length < 1e-6:
        raise ValueError("Dorsal click sits on the NCM centroid; move it dorsally.")

    toward_dorsal = toward_dorsal / length

    if ncm_vertices is not None:
        # Measure ALONG the Field L border, which puts the cut ACROSS it. The cut is
        # always perpendicular to `axis`, so using the Field L direction here -- rather
        # than its perpendicular -- is what makes the drawn line perpendicular to
        # Field L. Field L runs dorsoventrally, so this measures NCM's dorsoventral
        # extent, which is the thing being halved.
        axis = field_l_direction(ncm_vertices)

        # An edge has no inherent direction; take the end pointing dorsally.
        if np.dot(axis, toward_dorsal) < 0:
            axis = -axis
    else:
        axis = toward_dorsal

    projection = (rows - centroid[0]) * axis[0] + (columns - centroid[1]) * axis[1]
    midpoint = (projection.min() + projection.max()) / 2

    return axis, centroid, midpoint


def split_ncm(ncm_mask, dorsal_point, tissue=None, ncm_vertices=None):
    """Cut NCM in half with a line perpendicular to its Field L border.

    Field L is a visible landmark, so a cut across it is reproducible: it does not
    move when the dorsal click lands somewhere slightly different. Cutting at the
    midpoint of the extent gives two halves of equal HEIGHT along that axis; their
    areas can differ where NCM is wider at one end, which is the intended reading of
    "split NCM in half".

    The dorsal click is still required, but only for its sign: the drawn edge has two
    ends and nothing in the pixels says which one is the hippocampus side.

    Without `ncm_vertices` this falls back to the older centroid-to-dorsal-click
    axis, which only approximates the Field L direction, so it is a compatibility
    path for callers holding only a mask rather than an equivalent rule.

    Passing `tissue` restricts the extent to real tissue, so a polygon drawn a
    little loose does not drag the midpoint outward.
    """
    inside = ncm_mask if tissue is None else (ncm_mask & tissue)
    rows, columns = np.nonzero(inside)

    axis, centroid, midpoint = split_axis(
        ncm_mask, dorsal_point, tissue, ncm_vertices
    )

    projection = (rows - centroid[0]) * axis[0] + (columns - centroid[1]) * axis[1]

    dorsal = np.zeros_like(ncm_mask)
    ventral = np.zeros_like(ncm_mask)

    dorsal[rows[projection >= midpoint], columns[projection >= midpoint]] = True
    ventral[rows[projection < midpoint], columns[projection < midpoint]] = True

    return dorsal, ventral


def resolve_regions(image_path, shape, tissue=None):
    """Turn the drawn polygons plus the dorsal click into the countable regions.

    You draw NCM and CMM. dNCM and vNCM are derived, so the subdivision comes
    from a stated rule applied identically everywhere rather than from where a
    hand happened to fall.
    """
    drawn = load_regions(region_path(image_path), shape)

    if not drawn:
        raise FileNotFoundError(
            f"No regions drawn for {Path(image_path).name}.\n"
            f"Run: python regions/draw_regions.py {Path(image_path).name!r}"
        )

    regions = {}

    if "CMM" in drawn:
        regions["CMM"] = drawn["CMM"]

    if "NCM" in drawn:
        dorsal_point = load_dorsal_point(dorsal_path(image_path))

        if dorsal_point is None:
            raise FileNotFoundError(
                f"NCM needs a dorsal marker to split. Re-run draw_regions.py and "
                f"click once on the hippocampus side."
            )

        # Vertices as well as the mask: the split axis is perpendicular to the
        # drawn Field L edge, whose length only survives in the vertices.
        vertices = load_region_vertices(region_path(image_path)).get("NCM")

        dorsal, ventral = split_ncm(
            drawn["NCM"], dorsal_point, tissue, ncm_vertices=vertices
        )
        regions["dNCM"] = dorsal
        regions["vNCM"] = ventral

    # Any other drawn polygon is passed through, so extra regions need no code
    # change here.
    for name, mask in drawn.items():
        if name not in ("NCM", "CMM"):
            regions[name] = mask

    return regions


def annulus_kernel(inner_radius, outer_radius):
    radius = int(np.ceil(outer_radius))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.hypot(y, x)

    mask = ((distance >= inner_radius) & (distance <= outer_radius)).astype(np.float32)

    return mask / mask.sum()


def zone_mean(image, inner_radius, outer_radius):
    """Mean over an annulus, via FFT convolution.

    scipy.ndimage.convolve is direct and does not finish on a 70 megapixel image
    with a kernel this size; oaconvolve takes about a second.
    """
    return oaconvolve(
        image,
        annulus_kernel(inner_radius, outer_radius),
        mode="same",
    ).astype(np.float32)


def compute_score(green, geometry=None, scales=None):
    """Require the ring to be brighter than the center AND than the background.

    Ring-minus-center alone fires on any dark spot inside bright tissue, so
    unstained nuclei and vessel lumens score as highly as real cells. Taking the
    minimum against a background term removes them: a dark hole has no ring
    standing above its surroundings. On YW113 this lifted recall at 12 px from
    45% to 88%.

    `scales` multiplies every radius, and the responses are combined by taking the
    pixel-wise maximum. None and (1.0,) both mean the shipped single-scale filter, and
    the arithmetic in that case is unchanged -- every radius is multiplied by exactly
    1.0. It exists because this filter looks for ONE ring size, 8-15 px, and the cells
    the detector never proposes include annuli visibly larger than that (see
    detector_ceiling.py and grids/never proposed cells.png): a peak then lands on the
    bright arc instead of the dark centre, 16-21 px from the click.

    A caution on comparing scales. Every term here is a MEAN intensity, not an integral,
    so the responses are roughly comparable across radii without scale normalisation --
    but only roughly, and detect_candidates ranks peaks against each other to pick its
    top N. If a larger scale sweeps a brighter neighbourhood it can win on magnitude
    rather than on fit and crowd out real cells at the native scale. That is why the
    ceiling is measured for each scale ALONE as well as combined: a combination that
    scores worse than its own best member is this effect, not evidence against the idea.

    The smoothing sigma is scaled along with the radii. A larger ring template on
    unchanged smoothing would be a different filter shape, not the same filter looking
    for a bigger cell, and the point is to vary one thing.
    """
    geometry = geometry or Geometry()

    responses = []

    for scale in scales or (1.0,):
        smoothed = gaussian_filter(green, geometry.smoothing_sigma * scale)

        center = zone_mean(smoothed, 0, geometry.center_radius * scale)
        ring = zone_mean(
            smoothed,
            geometry.ring_inner_radius * scale,
            geometry.ring_outer_radius * scale,
        )
        background = zone_mean(
            smoothed,
            geometry.background_inner_radius * scale,
            geometry.background_outer_radius * scale,
        )

        responses.append(np.minimum(ring - center, ring - background))

    if len(responses) == 1:
        return responses[0]

    return np.maximum.reduce(responses)


def compute_tissue(green, geometry=None, threshold=None):
    """Blur, then threshold. `threshold=None` keeps the shipped TISSUE_THRESHOLD.

    The parameter exists so the region path can use a lower cutoff without moving the constant out
    from under the detector -- see the note on TISSUE_THRESHOLD. Adaptive alternatives were measured
    and lost to a plain smaller constant: rescaling by each section's own p99.5 left 0.70% of her
    drawn NCM outside the mask, a fixed per-section percentile 1.26%, Otsu 45%, while a flat 0.06
    reached 0.23%. Exposure correction was the intuitive fix and it was not the effective one.
    """
    geometry = geometry or Geometry()

    if threshold is None:
        threshold = TISSUE_THRESHOLD

    return gaussian_filter(green, geometry.tissue_sigma) > threshold


def load_annotations(path, extra_path=None):
    """Load clicked centers, optionally merging a supplementary file."""
    table = pd.read_csv(path)

    points = None
    for columns in (["axis-0", "axis-1"], ["y", "x"]):
        if set(columns).issubset(table.columns):
            points = table[columns].to_numpy(dtype=np.float64)
            break

    if points is None:
        raise ValueError(f"No coordinates in {path}. Columns: {table.columns.tolist()}")

    if extra_path and Path(extra_path).exists():
        extra_table = pd.read_csv(extra_path)

        # The exhaustive-box file stores crop-local and full-image coordinates.
        if {"image-axis-0", "image-axis-1"}.issubset(extra_table.columns):
            extra = extra_table[["image-axis-0", "image-axis-1"]].to_numpy()
        else:
            extra = extra_table[["axis-0", "axis-1"]].to_numpy()

        # Keep only genuinely new points, so re-clicked cells are not doubled.
        distance, _ = cKDTree(points).query(extra, k=1)
        new = extra[distance > MATCH_RADIUS]

        if len(new):
            print(f"  merged {len(new)} extra cells from {Path(extra_path).name}")
            points = np.vstack([points, new])

    return points


def load_region(path, shape):
    """Boolean mask of the reviewed area, or None when no region is defined.

    A two-vertex line is read as a cutoff, extended to a full half-plane: the
    side holding the image center is kept. Three or more vertices are read as a
    closed polygon.
    """
    if path is None:
        return None

    if not Path(path).exists():
        print(f"  WARNING: no region file at {path}; using the whole tissue area")
        return None

    vertices = pd.read_csv(path)[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)
    height, width = shape

    if len(vertices) < 2:
        raise ValueError(f"Region in {path} needs at least 2 vertices.")

    if len(vertices) == 2:
        (y0, x0), (y1, x1) = vertices
        y, x = np.mgrid[0:height, 0:width]

        # Signed distance from the line, positive on the image-center side.
        normal_y, normal_x = x1 - x0, -(y1 - y0)
        side = normal_y * (y - y0) + normal_x * (x - x0)
        center_side = normal_y * (height / 2 - y0) + normal_x * (width / 2 - x0)

        mask = (side > 0) if center_side > 0 else (side < 0)
        print(f"  region: cutoff line, keeping {mask.mean():.0%} of the image")

        return mask

    print(f"  region: polygon with {len(vertices)} vertices")

    return polygon2mask(shape, vertices)


def distance_to_edge(mask):
    """Pixels from each point inside `mask` to the nearest point outside it.

    Exposed because tissue-edge distance is a real covariate, not just an
    intermediate: PS6 signal near the boundary is less trustworthy, so anything
    grading the detector should be able to report edge and interior separately
    rather than averaging them together.
    """
    return distance_transform_edt(mask)


def shrink(mask, margin):
    """Erode a mask by a margin, via distance transform."""
    return distance_to_edge(mask) > margin


def detect_candidates(green, reviewed, geometry=None, scales=None):
    """Return (candidate yx coordinates, score map, threshold).

    Keeps a fixed number of candidates per unit of reviewed tissue area rather
    than thresholding on a score value or percentile; see
    CANDIDATES_PER_MEGAPIXEL for why.

    `scales` is passed to compute_score; None is the shipped behaviour.
    """
    geometry = geometry or Geometry()

    score = compute_score(green, geometry, scales)

    # Subsample for the percentile to keep it cheap on 70 megapixels.
    floor = np.percentile(score[::4, ::4][reviewed[::4, ::4]], FLOOR_PERCENTILE)

    peaks = peak_local_max(
        np.where(reviewed, score, -np.inf),
        min_distance=geometry.min_distance,
        threshold_abs=floor,
        exclude_border=False,
    )

    wanted = geometry.candidates_wanted(reviewed.sum())

    if len(peaks) <= wanted:
        print(
            f"  WARNING: only {len(peaks)} peaks above the floor, "
            f"wanted {wanted}; raise FLOOR_PERCENTILE"
        )
        return peaks, score, floor

    peak_scores = score[peaks[:, 0], peaks[:, 1]]
    best = np.argsort(-peak_scores)[:wanted]

    # Restore raster order, so downstream output is not sorted by score.
    best = np.sort(best)

    return peaks[best], score, peak_scores[best].min()


def label_candidates(
    candidates, annotations, reviewed, shape, geometry=None, excluded=None
):
    """Assign binary labels and mark which candidates are usable for training.

    Several candidates can land on one cell. Only the closest becomes a
    positive; the rest are off-center views of a real cell, so they are neither
    clean positives nor honest negatives and are held out.

    `excluded` is a mask of tissue to keep out of TRAINING entirely -- see
    training_cutoff_path(). Candidates inside it are struck from `usable` including
    POSITIVES, which is the point: a cell in tissue Alice does not trust should not
    teach the model either. It has to act on `usable` directly rather than through
    `reviewed`, because the interior test below is bypassed for positives.
    """
    geometry = geometry or Geometry()
    height, width = shape

    annotation_tree = cKDTree(annotations)
    distance_to_cell, nearest_cell = annotation_tree.query(candidates, k=1)
    near_a_cell = distance_to_cell <= geometry.match_radius

    labels = np.zeros(len(candidates), dtype=np.int8)
    is_duplicate = np.zeros(len(candidates), dtype=bool)

    for cell_index in np.unique(nearest_cell[near_a_cell]):
        group = np.flatnonzero(near_a_cell & (nearest_cell == cell_index))
        best = group[np.argmin(distance_to_cell[group])]
        labels[best] = 1
        is_duplicate[np.setdiff1d(group, [best])] = True

    # Negatives must sit clear of the region boundary, since a cell centered
    # just outside can pull a candidate just inside.
    interior = shrink(reviewed, geometry.boundary_margin)
    in_interior = interior[candidates[:, 0], candidates[:, 1]]

    # The stored patch is the wider one, so that is what must fit inside the image.
    half = geometry.patch_context_half_width

    has_full_patch = (
        (candidates[:, 0] >= half)
        & (candidates[:, 0] < height - half)
        & (candidates[:, 1] >= half)
        & (candidates[:, 1] < width - half)
    )

    usable = has_full_patch & ~is_duplicate & (in_interior | (labels == 1))

    # After the fact, and on `usable` rather than on `reviewed`, so it catches the
    # positives that `| (labels == 1)` above deliberately waves through.
    if excluded is not None:
        usable &= ~excluded[candidates[:, 0], candidates[:, 1]]

    return labels, usable, is_duplicate, distance_to_cell


# Sections with no unassisted annotation pass, so none of their labels can grade a
# model. Purp30 was annotated with model proposals from the start; OR408 began from
# an empty CSV in assisted_annotate.py. Both have ` backup.csv` files, but those are
# snapshots of assisted work, not first-pass clicks.
#
# NOT on this list, deliberately: the two sections Alice labelled on 2026-08-06 with
# `assisted_annotate.py <image> 1.01`. A threshold above 1.0 makes the proposal set
# empty, so those passes started from a blank canvas with only the detector's
# opinion-free candidate rings as a guide -- no model proposal was ever shown, let
# alone accepted, so their positives are honestly scoreable. That is the whole reason
# the passes were done: Gre595 and OR408 were previously ungradeable.
# Sections swept from a BLANK CANVAS: run through assisted_annotate.py at threshold
# 1.01, which makes `probabilities >= threshold` empty, so not one model proposal was
# ever drawn on screen. The only guide was the detector's candidate rings, which carry
# no opinion about whether a candidate is a cell.
#
# Two consequences, and the second is the reason this list exists separately from
# ASSISTED_FROM_THE_START:
#
#   * Their positives are honestly scoreable (nothing circular to exclude).
#   * Their NEGATIVES are trustworthy, which is the valuable part. Everywhere else in
#     a whole-section pass, a candidate with no click is a GUESS -- 38% of archive
#     negatives sitting on a real cell are mislabelled, and fixing exactly that is
#     what moved mean AP 0.467 -> 0.653. Here the sweep was exhaustive, so "no click"
#     means "looked at it, not a cell".
#
# Weaker than a grid window, in fairness: a window is exhaustive by construction,
# whereas a 55 Mpx section is exhaustive by Alice having scanned it. Treated as an
# experiment to be graded on the blind grid, not as a free assumption.
BLANK_CANVAS_SECTIONS = frozenset(
    {
        "Gre595_RH_3-1-4_NCM_Slide 1",
        "Gre595_LH-1-1-10_Slide 1_TD_p00_0_A01f00d1",
    }
)


def is_blank_canvas(image_path):
    """Whether this section was swept with no model proposals on screen."""
    return Path(image_path).stem in BLANK_CANVAS_SECTIONS


ASSISTED_FROM_THE_START = frozenset(
    {
        "Purp30_LH_1-1-8_NCM_Slide 1 cell centers.csv",
        "OR408_RH_1-2-1_NCM_Slide 2 cell centers.csv",
    }
)


def model_proposed(coordinates, labels, birds, first_pass_annotations):
    """Which positives entered the label set as accepted model proposals.

    On 2026-07-31 every section was re-reviewed in assisted_annotate.py, which shows
    the model's p>=0.98 candidates for confirmation. That added 176 cells and removed
    29 mislabelled ones. The removals are Alice's own judgement and correcting them
    is a clean gain. The additions are not: scoring a model against points it
    proposed is circular, because those points are high-scoring by construction.
    Measured cost of ignoring this -- mean AP reads 0.687 instead of 0.485.

    So the additions stay in TRAINING, where a confirmed cell is a real example, and
    come out of SCORING. They are excluded rather than relabelled negative: they are
    genuinely cells, and calling them negatives would punish the model for finding
    them. Same treatment duplicates already get.

    `first_pass_annotations` maps bird name to the clicks that existed before the
    re-review, or to None for a section that has no unassisted pass at all. A
    positive counts as first-pass if a pre-review click sits within match_radius.

    Only positives are considered. A negative was never proposed by anything -- it
    is a candidate nobody clicked -- so marking negatives here would empty the test
    set of exactly the points that make precision meaningful.

    `labels` is required for that reason: without it the distance test marks every
    negative as proposed, since no click sits near one by definition.
    """
    proposed = np.zeros(len(coordinates), dtype=bool)

    for name, annotations in first_pass_annotations.items():
        mine = np.flatnonzero((birds == name) & (labels == 1))

        if not len(mine):
            continue

        if annotations is None or not len(annotations):
            # No unassisted pass means none of this section's labels can serve as an
            # honest test set, however many cells its CSV holds.
            proposed[mine] = True
            continue

        distance, _ = cKDTree(annotations).query(coordinates[mine], k=1)
        proposed[mine] = distance > MATCH_RADIUS

    return proposed


def first_pass_path(annotation_path):
    """Clicks made before any model assistance, or None if there were none.

    A ` backup.csv` file is written the first time assisted_annotate.py saves, so
    for a section whose FIRST annotation pass was already assisted the backup is
    itself model-derived and must not be treated as clean. Purp30 and OR408 are both
    such sections -- Purp30 was annotated with assistance throughout, and OR408
    started from an empty CSV -- so both return None here despite having backups.
    """
    if Path(annotation_path).name in ASSISTED_FROM_THE_START:
        return None

    # BESIDE THE ANNOTATION, not in the working directory. This used to build a bare
    # filename -- `f"{stem} backup.csv"` with no folder -- which is the old rule that
    # `sidecar_path` exists to replace: it only found the backup while every cell
    # centres file sat at the top level, and would silently return None (meaning "this
    # section has no clean first pass") for any label file kept next to its image in a
    # subfolder. Returning None there is the bad kind of wrong: nothing fails, the
    # section just quietly stops contributing unassisted clicks to scoring.
    annotation = Path(annotation_path)
    backup = annotation.with_name(f"{annotation.stem} backup.csv")
    return str(backup) if backup.exists() else None


def to_output_size(patch, output_width):
    """Resize a square patch, or return it untouched when already the right size.

    Kept separate and used by both training augmentation and inference, so the
    interpolation the model trains on is the same operation it later meets.
    """
    if patch.shape[0] == output_width:
        return patch

    return resize(
        patch,
        (output_width, output_width),
        order=1,
        anti_aliasing=patch.shape[0] > output_width,
        preserve_range=True,
    ).astype(np.float32)


def cut_patch(green, y, x, geometry=None, half_width=PATCH_HALF_WIDTH):
    """Square patch of half_width * 2 raw pixels around (y, x).

    Cut in raw pixels, with no magnification correction: apparent cell size does
    not track the metadata pixel size (see Geometry), and resampling by it made
    detection worse. The CNN handles size variation through scale augmentation.

    half_width selects what the caller wants: PATCH_HALF_WIDTH for something the
    CNN can consume directly, PATCH_CONTEXT_HALF_WIDTH for the wider stored patch
    that training crops out of.
    """
    half = half_width

    patch = green[y - half : y + half, x - half : x + half]

    # A point within `half` of an edge gives a truncated slice, which negative
    # numpy indices turn into a silently wrong crop. Callers filter these out with
    # a has_full_patch mask; say so plainly rather than resampling a partial cell.
    if patch.shape != (2 * half, 2 * half):
        raise ValueError(
            f"({y}, {x}) is within {half} px of the edge of a "
            f"{green.shape[0]}x{green.shape[1]} image, so it has no full patch. "
            f"Filter candidates on {half} px first."
        )

    return to_output_size(patch, 2 * half_width)


def center_crop(patches, width):
    """Center crop a stack of square patches to `width`.

    The cell sits dead center by construction, so cropping around the middle keeps
    it centered while changing how much surround comes with it.
    """
    stored = patches.shape[-1]

    if width == stored:
        return patches

    if width > stored:
        raise ValueError(f"cannot crop {stored} px patches to {width} px")

    start = (stored - width) // 2

    return patches[..., start : start + width, start : start + width]


def to_model_input(patches, width=None):
    """Center crop a stored patch stack and resize it to what the CNN expects.

    Inference path for a stored 80x80 patch: crop to 64 and feed it. Passing a
    different width is how training presents the same cell at another apparent
    scale.
    """
    output = 2 * PATCH_HALF_WIDTH
    cropped = center_crop(patches, width or output)

    if cropped.shape[-1] == output:
        return cropped

    return np.stack([to_output_size(patch, output) for patch in cropped])
