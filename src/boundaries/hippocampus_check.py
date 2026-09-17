"""Does the hippocampus outline explain NCM's far border? Runs on however many HP polygons exist.

Alice, 2026-08-13: *"its still not identifying the hippocampus region"* / *"thats the most important
part"*. She is drawing `HP` polygons in `draw_regions.py` (key 5). This file is the go/no-go test,
written before the labels exist so the first saved polygon produces an answer instead of a to-do.

WHY HP AND NOT "OUTSIDE NCM". She asked whether to just outline everything outside NCM. That is the
complement of a polygon she has already drawn -- computable in one line, zero new information, and it
lumps hippocampus with Field L, CMM and white matter into one class that cannot teach appearance.
Naming HP changes the problem from LINE DETECTION (a few thousand thin border pixels, precision
required, failed three ways in [[border-detector-hurts-the-region]]) to REGION CLASSIFICATION (tens
of thousands of interior pixels, tolerant of sloppy edges). The far border then falls out as the edge
of the HP mask, and no line is ever detected.

Two questions, in order, because the second only matters if the first passes:

  1. GEOMETRY, no model. How close is her NCM far border to her HP outline? If HP's edge IS that
     border, the median distance is small and this whole approach reduces to `rule AND NOT HP`.
  2. WORTH. Score `rule AND NOT HP` on cells and on IoU against the plain rule. An oracle -- it uses
     her own HP outline -- so it is the CEILING for the approach, not a result. If the ceiling is not
     above the rule's 79.3% cells / 0.743 IoU, do not build a hippocampus model at all.

Usage:
  python hippocampus_check.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
)
from skimage.draw import polygon as raster_polygon

import ps6
from band_atlas import band_frame
from ceiling_lines import DATA_PATH, largest_blob
from find_band import NCM_LENGTH_UM, dorsal_on_canvas, find_band, is_left_hemisphere
from review_band import CAPTION_HEIGHT, HEADER_HEIGHT
from review_predictions import COLUMNS
from review_rule import CANVAS_UM, solid_tissue
from score_region_cells import cells_for, to_canvas
from show_border_kinds import border_kinds, canvas_field_l

# "Coincides" has to mean something. Her own drawing precision is ~18 um and the two outlines are
# drawn in separate passes, so 100 um is the honest bar -- the same one used for the path decode.
CLOSE_UM = 100.0

# Sections shown but NOT scored, with the reason. Alice, 2026-08-13: *"the hippocampus for gre595
# 3-1-4 rh has fallen onto the ncm region"* -- a sectioning artifact, so the hippocampal tissue is
# lying over NCM instead of bordering it. Both questions here assume HP is where it belongs: Q1 asks
# whether HP's edge coincides with NCM's far border, and Q2 subtracts HP from the region. A displaced
# piece answers neither, and with only a handful of HP outlines one such section would dominate the
# mean. Its panel is still drawn, because the footprint is a real fact about that image.
ARTIFACTS = {"Gre595_RH_3-1-4": "hippocampus has fallen onto NCM"}


def artifact_reason(stem):
    for fragment, reason in ARTIFACTS.items():
        if fragment in stem:
            return reason

    return None


def hp_on_canvas(image_path, pixel_size_um, shape):
    """Her HP polygon(s) rasterised straight onto the 8 um canvas.

    Rasterising at canvas scale rather than at full resolution and downsampling: a full-res mask for
    these sections is ~100x the pixels for an answer identical to within one canvas pixel.
    """
    vertices = ps6.load_region_vertices(ps6.region_path(image_path))
    drawn = vertices.get("HP")

    if drawn is None or len(drawn) < 3:
        return None

    canvas = to_canvas(np.asarray(drawn, float), pixel_size_um)
    rows, columns = raster_polygon(canvas[:, 0], canvas[:, 1], shape=shape)

    mask = np.zeros(shape, bool)
    mask[rows, columns] = True

    return mask


def rule_region(green, tissue, name, pixel_size_um, scale_um):
    """The shipping rule: Field L line + tissue edge + the two constant end cuts."""
    band = find_band(green, tissue, dorsal_on_canvas(name, pixel_size_um, scale_um))

    if band is None:
        return None

    along, across = band_frame(band, tissue, is_left_hemisphere(name))
    inside = (
        solid_tissue(tissue)
        & (across >= 0.0)
        & (along >= -NCM_LENGTH_UM / 2.0)
        & (along <= NCM_LENGTH_UM / 2.0)
    )

    if not inside.any():
        return None

    return largest_blob(binary_fill_holes(inside))


def overlap(predicted, truth):
    both = float((predicted & truth).sum())

    return {
        "iou": both / max(float((predicted | truth).sum()), 1.0),
        "recall": both / max(float(truth.sum()), 1.0),
        "precision": both / max(float(predicted.sum()), 1.0),
    }


def cells_kept(mask, truth, image_path, pixel_size_um):
    """Cells of hers inside `mask`, as a fraction of the cells inside her outline. None if uncounted."""
    points, _ = cells_for(image_path)

    if not len(points):
        return None

    canvas = to_canvas(points, pixel_size_um).round().astype(int)
    keep = (
        (canvas[:, 0] >= 0) & (canvas[:, 0] < mask.shape[0])
        & (canvas[:, 1] >= 0) & (canvas[:, 1] < mask.shape[1])
    )
    rows, columns = canvas[keep, 0], canvas[keep, 1]

    in_truth = truth[rows, columns]
    total = int(in_truth.sum())

    if not total:
        return None

    return float((in_truth & mask[rows, columns]).sum() / total)


def panel(green, truth, hp, rule, trimmed, kinds):
    """Her outline drawn as its three KINDS, not as one white line.

    Drawing it in a single colour is what made the first version of this picture unreadable: Alice
    pointed at Purp30 and said *"no its field l"* about the structure beyond a border I had described
    as the far border -- and she was right, that stretch is her long straight Field L run, which the
    rule already handles. Only the RED interior stretch is the thing HP is supposed to explain, so it
    has to be visually distinct or every reading of this picture is guesswork.
    """
    grey = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1).astype(np.float32)

    rgb[trimmed] = 0.65 * rgb[trimmed] + 0.35 * np.array([90, 255, 120])
    rgb[hp & ~binary_erosion(hp, iterations=2)] = (255, 170, 0)
    rgb[rule & ~binary_erosion(rule, iterations=2)] = (60, 140, 255)

    # Thickened so a one-pixel boundary survives at grid scale, interior last so nothing covers it.
    for name, colour in (("edge", (150, 150, 150)), ("line", (255, 230, 60)), ("interior", (255, 60, 60))):
        thick = binary_dilation(kinds[name], iterations=2)

        if thick.any():
            rgb[thick] = colour

    return Image.fromarray(rgb.astype(np.uint8))


def main():
    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    images = [str(p) for p in data["images"]]
    scale_um = float(data["scale_um"])

    panels, captions = [], []
    found, excluded = 0, []
    coincidence, rule_scores, trimmed_scores = [], [], []
    rule_cells, trimmed_cells = [], []

    for index, (stem, name) in enumerate(zip(stems, images)):
        truth = data["labels"][index] == 1

        if not truth.any():
            continue

        green = data["inputs"][index, 0].astype(np.float32)
        tissue = data["inputs"][index, 1] > 0.5
        pixel_size = float(data["pixel_sizes"][index])

        hp = hp_on_canvas(name, pixel_size, truth.shape)

        if hp is None:
            continue

        found += 1
        reason = artifact_reason(stem)

        if reason:
            excluded.append(stem)

        # QUESTION 1: geometry. Her far border, by the same definition every earlier measurement used.
        field_l, _ = canvas_field_l(name, pixel_size, scale_um)
        kinds, _ = border_kinds(truth, tissue, field_l)
        far = kinds["interior"]

        to_hp = distance_transform_edt(~(hp & ~binary_erosion(hp, iterations=1))) * CANVAS_UM
        distances = to_hp[far] if far.any() else np.array([np.nan])
        near = float((distances <= CLOSE_UM).mean()) if far.any() else float("nan")

        if reason is None:
            coincidence.append((stem, float(np.median(distances)), near, int(far.sum())))

        # QUESTION 2: worth. An ORACLE -- it uses her own HP outline, so it is the ceiling.
        rule = rule_region(green, tissue, name, pixel_size, scale_um)

        if rule is None:
            print(f"  {stem[:30]:32s} no band found, geometry only")
            continue

        trimmed = largest_blob(binary_fill_holes(rule & ~hp))

        if reason is None:
            rule_scores.append(overlap(rule, truth))
            trimmed_scores.append(overlap(trimmed, truth))

            for store, mask in ((rule_cells, rule), (trimmed_cells, trimmed)):
                kept = cells_kept(mask, truth, name, pixel_size)

                if kept is not None:
                    store.append(kept)

        panels.append(panel(green, truth, hp, rule, trimmed, kinds))
        captions.append((
            stem[:30],
            f"NOT SCORED: {reason}" if reason else f"HP explains {100 * near:.0f}% of the far border",
        ))

    if not found:
        raise SystemExit(
            "No HP polygons found yet.\n"
            "  python draw_regions.py \"<image>\"   then press 5, draw, Ctrl-S\n"
            "Start with OR408_RH_1-2-1 and YW113_RH_1-1-7 -- the rule loses the most of her cells\n"
            "there (77%, 79% kept) and both have normal anatomy. Gre595_RH_3-1-4 loses more still\n"
            "(53%) but its hippocampus has fallen onto NCM, so it is shown and not scored."
        )

    # Only sections that actually HAVE an HP outline, so this never announces the exclusion of a
    # section that was simply never drawn.
    for stem in excluded:
        print(f"  excluded from the scores: {stem[:40]} -- {artifact_reason(stem)}")

    print(f"\n  1. GEOMETRY: is her HP outline where NCM's far border is?  (bar: {CLOSE_UM:.0f} um)\n")
    print(f"  {'section':32s} {'far border px':>13s} {'median dist':>12s} {'within bar':>11s}")

    for stem, median, near, count in coincidence:
        print(f"  {stem[:30]:32s} {count:13d} {median:9.0f} um {100 * near:10.1f}%")

    if rule_scores:
        print(f"\n  2. WORTH, an ORACLE using her own HP outline -- the ceiling, not a result.\n")
        print(f"  {'region':32s} {'IoU':>6s} {'recall':>8s} {'precision':>10s} {'cells kept':>11s}")

        for label, scores, cells in (
            ("the rule alone", rule_scores, rule_cells),
            ("the rule AND NOT HP", trimmed_scores, trimmed_cells),
        ):
            kept = f"{100 * np.mean(cells):10.1f}%" if cells else f"{'no counts':>11s}"
            print(f"  {label:32s} {np.mean([s['iou'] for s in scores]):6.3f} "
                  f"{np.mean([s['recall'] for s in scores]):8.3f} "
                  f"{np.mean([s['precision'] for s in scores]):10.3f} {kept}")

        print(f"\n  n = {len(rule_scores)} sections with an HP outline."
              f"  Compare against the whole-set rule: IoU 0.743, cells kept 79.3%.")

    if panels:
        cell_width = max(p.width for p in panels)
        cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
        grid_rows = int(np.ceil(len(panels) / COLUMNS))

        sheet = Image.new(
            "RGB", (COLUMNS * cell_width, HEADER_HEIGHT + grid_rows * cell_height), "black"
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), "DOES THE HIPPOCAMPUS OUTLINE EXPLAIN NCM'S FAR BORDER?", fill="white")
        draw.text((4, 18), "Her outline by KIND: GREY = tissue edge, YELLOW = Field L, RED = interior far "
                  "border (what HP should explain).  ORANGE = her HP.  BLUE = the rule.  GREEN = rule AND NOT HP.",
                  fill=(140, 255, 160))

        for position, (image, (title, note)) in enumerate(zip(panels, captions)):
            row, column = divmod(position, COLUMNS)
            x, y = column * cell_width, HEADER_HEIGHT + row * cell_height
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + image.height + 2), title, fill="white")
            draw.text((x + 4, y + image.height + 14), note, fill=(140, 255, 160))

        out = Path("grids") / "hippocampus check.png"
        sheet.save(out)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
