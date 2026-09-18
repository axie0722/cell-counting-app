"""Look at the cells the model counted, on the image, in napari -- and correct them.

`count_cells.py --points` writes the position of every cell it kept. This opens the
section with those positions on top, so the count can be inspected rather than taken
on faith -- which matters most when comparing against someone else's manual count: a
number that disagrees says nothing about WHY until you can see where the two differ.

IT USED TO BE READ-ONLY, and that was a real limitation rather than a safety feature.
Alice, 2026-09-12: *"could you change the app so that review cells lets the user add and
delete cells"*. Seeing a false positive and being unable to remove it means the number on
the report stays wrong even though the person who can tell has already looked at it. So the
cell layers are editable now: click to add, select and press Delete to remove, and the count
that comes out is what YOU say is there.

WHICH REGION A NEW CELL BELONGS TO IS DECIDED BY THE OUTLINES, NOT BY THE LAYER YOU
CLICKED IN. The counting rule is centre-inside-region, so a hand-placed cell is looked up in
the same region masks the counter used. Add a cell into the wrong layer and it still lands in
the right region; add one outside every outline and it is refused with a message, because a
cell that belongs to no region has no density to contribute to. The cost is that the masks are
built whether or not the outlines are drawn -- one uint8 array over the section, ~40 MB, where
`--regions` costs a second one for the boundary band.

A HOLLOW RING IS THE MODEL'S CALL, A FILLED DOT IS YOURS. Same colour, because it is the same
region and the same kind of thing; a different fill, because "did a person look at this" is the
question you will have when you come back to it. The `source` column in the counted-cells file
is what makes that survive a reopen -- and it is also what keeps these usable as training data:
confirming a cell the model proposed teaches it nothing it did not already believe, but the ones
you ADD and the ones you DELETE are exactly what it got wrong. See ps6.edits_path.

NOTHING IS WRITTEN UNLESS SOMETHING CHANGED. Saving happens on Ctrl-S and again when the window
closes, and a save with no additions and no deletions touches no file at all -- the same rule as
draw_regions.write_if_changed, and for the same reason: the app decides a count is stale by
comparing modification times, so a look-only visit that rewrote its files would report the count
as out of date when it is not.

When something DID change, three files move together:
  * "<stem> counted cells.csv"  -- the new answer, which is where the app reads its numbers;
  * "<stem> cell edits.csv"     -- append-only, every add and delete with the time;
  * ps6_counts.csv              -- this section's rows only, with the new count and density.
`detected` is left exactly as it was: the detector proposed what it proposed, and editing the
kept cells does not change its history.

Points are split into one layer per region, so a layer can be toggled off, and the
region outlines are drawn too. A disagreement in total count is often a disagreement
about where CMM ends rather than about any individual cell, and that is only visible
with the boundary on screen.

Lowest-confidence calls get their own layer. Those are the ones sitting just above the
threshold, so they are where the model is most likely to be wrong in either direction,
and they are worth looking at before anything else.

These sections are 30-40 Mpx. Open ONE at a time -- two viewers at once ran this
machine out of memory (2026-08-09) and napari misbehaves rather than failing cleanly.
The region overlay is another full-size array on top of the image, so it is OFF by
default and enabled with --regions.

Usage:
  python counting/show_counts.py "OR99_RH_....TIF"
  python counting/show_counts.py "..." --regions      # outline dNCM/vNCM/CMM (costs memory)
  python counting/show_counts.py "..." --clicks       # overlay hand clicks too, if any exist
  python counting/show_counts.py "..." --read-only    # look without being able to change anything
  python counting/show_counts.py --check              # check the editing rules, no window, no image
"""

import sys
from pathlib import Path

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. See app_path.py, and the same block at
# the top of count_cells.py: this runs as its own process, so it gets counting/ on sys.path and none of
# its siblings.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

import napari
import numpy as np
import pandas as pd
from napari.utils.colormaps import DirectLabelColormap

import ps6
from count_cells import OUTPUT_PATH, HISTORY_PATH, merge_run, points_path, read_table, save_csv

# The least confident slice of the kept calls, shown separately. A fraction rather than
# a fixed probability: the score distribution differs between sections, and the useful
# idea is "the bottom of what this section kept", not "everything under 0.995".
MARGINAL_FRACTION = 0.2

REGION_COLORS = {"dNCM": "cyan", "vNCM": "yellow", "CMM": "magenta"}

# Thick enough to see at a zoomed-out view of an 8000 px section, thin enough not to
# swallow a cell sitting on the boundary.
OUTLINE_WIDTH = 10

# Drawn SOLID. It was 0.25 and Alice could not see the boundaries (2026-08-20), which is
# the one thing --regions exists to show. The comment that justified the transparency was
# about not hiding a cell being judged, but that worry does not apply here: this is a
# 10 px band along the edge, not a filled region, and the cell circles are added to the
# viewer AFTER it, so they are painted on top of it either way. Visibility comes from
# contrast, not from area -- a thin solid line reads better at every zoom than a fat faint
# one, and hides less of the image. Drag napari's opacity slider on the layer to see
# underneath a boundary in the rare case that matters.
OUTLINE_OPACITY = 1.0

# Hollow for the model, filled for a person. WHITE rather than a fourth region colour: the
# fill answers "who put this here" and the ring already answers "which region", so reusing a
# region's colour for the fill would collide two meanings in one mark.
MODEL_FILL = "transparent"
HAND_FILL = "white"

# The probability written for a cell you placed yourself. 1.0 because the column means
# "confidence this is a cell" and a person looking at it IS the highest confidence available
# here -- there is no better evidence in this project than Alice's eye. It also keeps hand
# cells out of the least-confident layer, which exists to show where the MODEL is unsure.
# The `source` column, not this number, is what tells the two apart afterwards.
HAND_PROBABILITY = 1.0

# What a counted-cells file holds. `source` is new (2026-09-12) and older files do not have
# it; anything read without one is the model's, which is true by construction because until
# now nothing else could write this file.
CELL_COLUMNS = ["axis-0", "axis-1", "region", "probability", "source"]

# What the edit log holds. `at` first so the file reads as a diary.
EDIT_COLUMNS = ["at", "action", "axis-0", "axis-1", "region", "probability"]


def resolve(argument):
    if Path(argument).exists():
        return argument

    candidate = Path("birds") / argument

    if candidate.exists():
        return str(candidate)

    # A bare stem is what gets typed in practice, since that is what the counts CSV is
    # named after.
    matches = sorted(Path(".").glob(f"*{argument}*.TIF")) + sorted(
        Path("birds").glob(f"*{argument}*.TIF")
    )

    if len(matches) == 1:
        return str(matches[0])

    if len(matches) > 1:
        raise SystemExit(
            f"{argument} matches several images:\n  "
            + "\n  ".join(str(m) for m in matches)
        )

    raise SystemExit(f"No such image: {argument}")


def load_cells(table_path):
    """The counted cells, with a `source` column whether or not the file had one."""
    points = pd.read_csv(table_path)

    if "source" not in points.columns:
        points["source"] = "model"

    return points


def key_of(y, x):
    """One cell's identity: its pixel.

    ROUNDED TO WHOLE PIXELS, which is what makes matching the layer back to the file
    possible at all. The file holds integer candidate centres; napari holds floats and
    hands them back with whatever precision a drag left behind. A cell is a place, and the
    resolution of "place" here is one pixel -- these are 0.65 um, so nothing that matters is
    lost, and two candidates never share a pixel because detection reports centres.

    The side effect is deliberate: DRAGGING a model cell more than half a pixel makes it a
    new hand cell at the new position and the old one a deletion. That is the honest reading
    of what dragging means -- you moved the call, so it is yours now.
    """
    return int(round(float(y))), int(round(float(x)))


def region_labels(image_path, green, want_outlines):
    """(labels, names, outlines) -- which region every pixel is in, for looking cells up.

    ONE uint8 ARRAY INSTEAD OF THE MASKS THEMSELVES. resolve_regions hands back three or four
    full-size boolean masks; keeping them all alive to answer "which region is this pixel in"
    would be 160 MB on a machine that has run out of memory before. A label image answers the
    same question with one index and 40 MB, and the masks are freed as soon as it is built.

    FIRST REGION WINS WHERE TWO OVERLAP, and a warning is printed when any do. The counter
    itself counts such a pixel in BOTH regions -- its rows are computed per mask,
    independently -- so there is no answer here that agrees with it, because a list of cells
    cannot hold one cell in two regions without holding it twice. Overlap between the drawn
    outlines is a drawing question, and the message says where to look.

    `outlines` is the boundary band for display, or None. Built here rather than separately
    because it needs the same masks and this is the only place they exist.
    """
    geometry = ps6.geometry_for(image_path)
    tissue = ps6.compute_tissue(green, geometry)
    regions = ps6.resolve_regions(image_path, green.shape, tissue)

    labels = np.zeros(green.shape, dtype=np.uint8)
    outlines = np.zeros(green.shape, dtype=np.uint8) if want_outlines else None
    names = {}
    colours = {0: "transparent", None: "transparent"}
    claimed = 0

    for index, (name, mask) in enumerate(regions.items(), start=1):
        names[index] = name
        colours[index] = REGION_COLORS.get(name, "white")
        claimed += int(mask.sum())

        # Only where nothing has claimed the pixel yet, so the first region drawn wins and
        # the result does not depend on dictionary order changing under us.
        labels[mask & (labels == 0)] = index

        if outlines is not None:
            # An outline is "inside the mask, but within a few pixels of leaving it",
            # which is what distance_to_edge measures. An outline rather than a filled
            # overlay so it never obscures a cell being judged.
            edge = mask & (ps6.distance_to_edge(mask) <= OUTLINE_WIDTH)
            outlines[edge & (outlines == 0)] = index
            del edge

    shared = claimed - int((labels > 0).sum())

    if shared:
        microns = ps6.pixel_size_um(image_path)
        area = f" ({shared * (microns / 1000.0) ** 2:.3f} mm2)" if microns else ""
        print(
            f"  WARNING: the outlines overlap over {shared:,} pixels{area}. A cell added "
            f"there is assigned to the first region that covers it, and the counter counts "
            f"it in every region that covers it -- so the two will disagree. Redraw the "
            f"outlines so they do not cross."
        )

    del tissue, regions

    return labels, names, outlines, colours


def region_at(labels, names, key):
    """Which region a pixel is in, or None for none of them (or no masks at all)."""
    if labels is None:
        return None

    y, x = key

    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return None

    return names.get(int(labels[y, x]))


def cell_layers(viewer):
    """The layers that hold counted cells, in the order they were added.

    Found by a flag in `metadata` rather than by name or by type. The hand-clicks layer is
    also Points and its name is also readable, and counting somebody else's manual clicks as
    part of this count is the one mistake in here that would be invisible in the result.
    """
    return [layer for layer in viewer.layers if layer.metadata.get("counted")]


def by_pixel(records):
    """Rows grouped by the pixel they sit on: {(y, x): [row, ...]}.

    A LIST PER PIXEL, NOT ONE ROW. Almost always the list has one element, because two
    detected candidates never share a pixel. The exception is a cell inside two overlapping
    outlines: the counter's rows are computed per region independently, so such a cell is
    written twice, once per region. Keyed by pixel alone with one row each, the second would
    quietly replace the first -- and then saving would drop a row nobody deleted and the edit
    log would record a deletion that never happened. Grouping costs a list and makes deleting
    the cell delete it from both regions, which is what deleting a cell means.
    """
    grouped = {}

    for row in records:
        grouped.setdefault(key_of(row["axis-0"], row["axis-1"]), []).append(row)

    return grouped


def gather(layers, original, labels, names):
    """What the layers say the count is now: (rows, added, removed, outside).

    `original` is by_pixel() over the rows loaded from the file. A point that matches a pixel
    keeps those rows verbatim -- their probability, region and source -- so opening a section
    and saving it back cannot quietly rewrite the model's own numbers. Anything unmatched is
    new, and gets its region from the masks.

    A pixel is used once. Two clicks on the same cell, or the same cell showing on two
    layers, is one cell.
    """
    rows = []
    added = []
    outside = []
    seen = set()

    for layer in layers:
        for y, x in np.asarray(layer.data, dtype=float).reshape(-1, 2):
            key = key_of(y, x)

            if key in seen:
                continue

            seen.add(key)
            already = original.get(key)

            if already:
                rows.extend(already)
                continue

            region = region_at(labels, names, key)

            if region is None:
                outside.append(key)
                continue

            fresh = {
                "axis-0": key[0],
                "axis-1": key[1],
                "region": region,
                "probability": HAND_PROBABILITY,
                "source": "hand",
            }

            rows.append(fresh)
            added.append(fresh)

    removed = [
        record
        for key, group in original.items()
        if key not in seen
        for record in group
    ]

    return rows, added, removed, outside


def tally(rows):
    """Cells per region, as a plain dict. Regions with none are absent."""
    counted = {}

    for row in rows:
        counted[row["region"]] = counted.get(row["region"], 0) + 1

    return counted


def log_edits(image_path, added, removed, path=None):
    """Append every add and delete to the section's edit log. See ps6.edits_path."""
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {"at": stamp, "action": action, **{k: row[k] for k in CELL_COLUMNS if k != "source"}}
        for action, group in (("added", added), ("removed", removed))
        for row in group
    ]

    if not rows:
        return None

    path = Path(path if path is not None else ps6.edits_path(image_path))
    fresh = pd.DataFrame(rows, columns=EDIT_COLUMNS)
    older = read_table(path)

    return save_csv(
        pd.concat([older, fresh], ignore_index=True) if older is not None else fresh, path
    )


def update_summary(image_path, rows, added, removed, path=OUTPUT_PATH, history=HISTORY_PATH):
    """Put the edited numbers into the folder's summary table. Returns a message.

    ONLY THIS SECTION'S ROWS, and only the columns that the edit actually changed:
    `classified` becomes the number of cells now on screen and `density_per_mm2` is
    recomputed from it against the area already measured. `detected` and its density are left
    alone -- the detector proposed what it proposed, and rewriting its history to match a
    later human decision would destroy the one measurement that says how much work the
    classifier is doing.

    Going through count_cells.merge_run rather than writing the file directly, so an edit
    inherits everything a count already learned about this file: other sections' rows are
    kept (Alice lost a section's numbers to a plain overwrite on 2026-08-20), the write is
    atomic, and the before-and-after both survive in ps6_counts_history.csv.

    NO RACE WITH A RUNNING COUNT, worth stating because merge_run is read-then-write: the app
    suspends any count while a napari window is open and only resumes it when this process
    exits, so a count cannot write this file between the read and the write here.

    `hand_edits` reads like "+3-1". It is added by this function and never by a count, so a
    later recount silently clears it -- which is right, because a recount replaces the cells
    these edits were made to.
    """
    name = Path(image_path).name
    existing = read_table(path)

    if existing is None or "image" not in existing.columns:
        return f"  {path} has no rows yet, so it was left alone"

    mine = existing[existing["image"] == name]

    if not len(mine):
        return (
            f"  {path} has no rows for this section, so it was left alone. Its cell numbers "
            f"are still correct in the app, which reads the counted-cells file."
        )

    counted = tally(rows)
    gains = tally(added)
    losses = tally(removed)
    updated = []

    for record in mine.to_dict("records"):
        region = record["region"]
        record["classified"] = int(counted.get(region, 0))
        area = record.get("area_mm2")

        if pd.notna(area) and area:
            record["density_per_mm2"] = round(record["classified"] / area, 1)

        record["hand_edits"] = f"+{gains.get(region, 0)}-{losses.get(region, 0)}"
        updated.append(record)

    kept, replaced = merge_run(updated, path=path, history_path=history)

    missing = sorted(set(counted) - set(mine["region"]))
    note = (
        f"\n  NOTE: {', '.join(missing)} has cells now but no row in {path}; recount the "
        f"section to give it one"
        if missing
        else ""
    )

    return (
        f"  {path}: {len(updated)} rows updated, {kept} other rows kept "
        f"(replaced {replaced}){note}"
    )


def save_cells(image_path, layers, original, labels, names, say=print, paths=None):
    """Write the edited count, if it is edited. Returns (added, removed, outside).

    CALLED FROM TWO PLACES -- Ctrl-S and the close of the window -- so it has to be safe to
    call twice with nothing in between, and it is: after a save the file matches the layers,
    so the second call finds nothing added and nothing removed and writes nothing.

    `original` is UPDATED IN PLACE to what was just written, which is what makes that true
    and also makes a second round of edits diff against the right thing.
    """
    paths = paths or {}
    rows, added, removed, outside = gather(layers, original, labels, names)

    if outside:
        say(
            f"{len(outside)} cell(s) are outside every outline and were NOT counted -- "
            f"first at row {outside[0][0]}, column {outside[0][1]}"
        )

    if not added and not removed:
        say("no cells added or deleted, so nothing was written")

        return added, removed, outside

    table = pd.DataFrame(rows, columns=CELL_COLUMNS).sort_values(
        "probability", ascending=False
    )

    counts_at = paths.get("counts", points_path(image_path))
    save_csv(table, counts_at)

    print(f"\n  wrote {counts_at} ({len(table)} cells)")
    print(f"  {log_edits(image_path, added, removed, paths.get('edits'))}: the record of what changed")
    print(update_summary(
        image_path,
        rows,
        added,
        removed,
        path=paths.get("summary", OUTPUT_PATH),
        history=paths.get("history", HISTORY_PATH),
    ))

    per_region = tally(rows)
    say(
        f"saved: +{len(added)} -{len(removed)} -> "
        + "  ".join(f"{name} {number}" for name, number in per_region.items())
    )

    original.clear()
    original.update(by_pixel(rows))

    return added, removed, outside


def main():
    if "--check" in sys.argv[1:]:
        raise SystemExit(check())

    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not arguments:
        raise SystemExit(__doc__)

    image_path = resolve(arguments[0])
    table_path = points_path(image_path)
    editable = "--read-only" not in sys.argv[1:]

    if not Path(table_path).exists():
        raise SystemExit(
            f"No {table_path}. Count the section with positions saved first:\n"
            f'  python counting/count_cells.py "{image_path}" --points'
        )

    points = load_cells(table_path)

    print(f"Loading {Path(image_path).name}...")
    image, green = ps6.load_green(image_path)

    viewer = napari.Viewer(title=f"Counted: {Path(image_path).stem}")

    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 14

    def say(text):
        """On the image AND in the terminal.

        The terminal is not enough on its own: the app runs this in a subprocess whose output
        you never see, so an instruction that only prints is an instruction that does not
        exist. See [[app-messages-must-be-on-screen]].
        """
        viewer.text_overlay.text = text
        print(f"  {text}")

    is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4)

    # Contrast limits set explicitly from percentiles. Left to itself napari scales to
    # the data's full min-max, and these are float32 images in 0-1 whose tissue sits
    # around 0.2 with a thin bright tail -- which renders the whole section BLACK
    # (2026-08-09). The same reason false_positive_gallery.stretch exists.
    low, high = np.percentile(green, [1, 99.9])

    viewer.add_image(
        image,
        name="PS6",
        rgb=is_rgb,
        # A single-channel fluorescence image is not greyscale data -- it is a green
        # stain, and Alice reads the stain. napari shows one channel as grey unless told
        # otherwise; these images arrive single-channel, so it must be told.
        **({} if is_rgb else {"colormap": "green", "contrast_limits": (low, high)}),
    )

    # 30-40 Mpx: every full-size array is ~130 MB as float32. Two viewers at once
    # exhausted memory on this machine (2026-08-09: 62 MB free, napari misbehaving), so
    # anything full-size is released as soon as it has been used.
    del image

    want_outlines = "--regions" in sys.argv[1:]
    labels = names = None

    # THE MASKS ARE BUILT FOR EDITING, NOT FOR DRAWING. A new cell needs a region, and the
    # only honest source for one is the mask the counter used. When neither editing nor
    # --regions is asked for, this is skipped and the window costs what it always did.
    if editable or want_outlines:
        try:
            labels, names, outlines, colours = region_labels(image_path, green, want_outlines)
        except FileNotFoundError as missing:
            # A counted section always had outlines, so this means they were deleted or
            # renamed since. Deleting cells still works without them; adding does not,
            # because there would be nothing to say which region a click landed in.
            labels = names = outlines = None
            print(f"  {missing}")
            print("  no region masks, so cells can be DELETED but not added")

        if outlines is not None:
            # Region boundaries as labels rather than polygons: resolve_regions returns
            # masks (dNCM and vNCM are derived by cutting NCM, so there is no polygon for
            # either), and a mask outline is the honest depiction of what was counted over.
            # Each region's outline gets THE SAME COLOUR AS ITS CELL CIRCLES -- add_labels
            # left to itself picks from napari's own palette, so the CMM boundary came out
            # some arbitrary colour while CMM's cells were magenta.
            layer = viewer.add_labels(
                outlines,
                name="regions",
                opacity=OUTLINE_OPACITY,
                colormap=DirectLabelColormap(color_dict=colours),
            )

            # Not a thing to paint on. Without this, a stray keypress puts a Labels layer in
            # paint mode and the next click edits the boundary picture instead of the count.
            layer.editable = False

            del outlines

    del green

    # The cutoff is taken over the MODEL's calls only. Hand-placed cells sit at 1.0 by
    # definition, and letting them into the quantile would move a threshold whose whole
    # meaning is "the bottom of what the model kept".
    from_model = points[points["source"] == "model"]
    cutoff = from_model["probability"].quantile(MARGINAL_FRACTION) if len(from_model) else 0.0

    marginal = points[(points["source"] == "model") & (points["probability"] <= cutoff)]
    confident = points.drop(marginal.index)

    for name, group in confident.groupby("region"):
        layer = viewer.add_points(
            group[["axis-0", "axis-1"]].to_numpy(),
            # NAME WITHOUT A NUMBER, unlike before. The number went stale the moment a cell
            # was added, and a layer called "CMM (13 cells)" holding 15 is worse than no
            # number at all. The live tally is on the image instead, in refresh().
            name=name,
            size=40,
            face_color=[
                HAND_FILL if source == "hand" else MODEL_FILL for source in group["source"]
            ],
            border_color=REGION_COLORS.get(name, "white"),
            border_width=0.15,
        )
        layer.metadata["counted"] = True
        layer.current_face_color = HAND_FILL
        layer.current_border_color = REGION_COLORS.get(name, "white")

    if len(marginal):
        layer = viewer.add_points(
            marginal[["axis-0", "axis-1"]].to_numpy(),
            name=f"least confident (p<={cutoff:.4f})",
            size=40,
            face_color=MODEL_FILL,
            border_color="red",
            border_width=0.2,
        )
        layer.metadata["counted"] = True
        layer.current_face_color = HAND_FILL
        layer.current_border_color = "red"

    if "--clicks" in sys.argv[1:]:
        annotations = next(
            (
                b.get("annotations")
                for b in ps6.BIRDS
                if Path(b["image"]).stem == Path(image_path).stem
            ),
            None,
        )

        if annotations and Path(annotations).exists():
            clicks = ps6.load_annotations(annotations)
            # Deliberately NOT marked `counted`: these are somebody's manual clicks for
            # comparison, not part of this count. See cell_layers.
            viewer.add_points(
                clicks,
                name=f"hand clicks ({len(clicks)})",
                size=60,
                face_color="transparent",
                border_color="lime",
                border_width=0.12,
            )
            print(f"  overlaid {len(clicks)} hand clicks")
        else:
            print("  no hand clicks registered for this section")

    layers = cell_layers(viewer)
    original = by_pixel(points.to_dict("records"))

    print(
        f"\n{len(points)} cells counted:\n"
        + points.groupby("region")
        .agg(cells=("probability", "size"), lowest_p=("probability", "min"))
        .to_string()
        + f"\n\nRed circles are the least confident {MARGINAL_FRACTION:.0%} "
        f"(p <= {cutoff:.4f}) -- start there.\n"
        f"Region outlines are drawn in their region's own colour: dNCM cyan, vNCM\n"
        f"yellow, CMM magenta -- the same colours as the circles inside them.\n"
        f"Toggle layers in the panel on the left to compare regions.\n"
        f"If the image looks too dark or too flat, drag the contrast slider under the\n"
        f"PS6 layer -- it is set from percentiles, not tuned per section.\n"
    )

    if not editable:
        say("read-only: look, then close the window. Nothing here can change the count.")
        napari.run()

        return

    def summary():
        """The live count, per region, plus what this session has changed."""
        rows, added, removed, outside = gather(layers, original, labels, names)
        counted = tally(rows)
        line = "  ".join(f"{name} {number}" for name, number in sorted(counted.items()))
        change = f"   (+{len(added)} -{len(removed)} unsaved)" if added or removed else ""
        missed = f"   {len(outside)} outside the outlines" if outside else ""

        return f"{line}{change}{missed}"

    def refresh(_event=None):
        viewer.text_overlay.text = (
            f"{summary()}\n"
            "Ctrl-A add cells   Ctrl-E pick, then Delete   Ctrl-S save   saves on close too"
        )

    for layer in layers:
        # `events.data` fires on adding, deleting and dragging, which is exactly the set of
        # things that change the answer. The tally is recomputed rather than adjusted,
        # because an adjustment can drift out of step with the layers and this cannot.
        layer.events.data.connect(refresh)

    def armed():
        """The cells layer a keypress should act on: the selected one, or the first."""
        active = viewer.layers.selection.active

        return active if active in layers else (layers[0] if layers else None)

    @viewer.bind_key("Control-A", overwrite=True)
    def add_cells(_viewer):
        layer = armed()

        if layer is None:
            say("no cell layers to add to")

            return

        viewer.layers.selection.active = layer
        # Selecting a Points layer does not arm it -- without the mode change the first
        # click does nothing and the key looks broken. The same trap as draw_regions' key 4.
        layer.mode = "add"
        refresh()
        say(f"click to add cells to {layer.name}. The region comes from the outlines, not "
            f"from this layer.")

    @viewer.bind_key("Control-E", overwrite=True)
    def pick_cells(_viewer):
        layer = armed()

        if layer is None:
            say("no cell layers to edit")

            return

        viewer.layers.selection.active = layer
        layer.mode = "select"
        refresh()
        say(f"click cells in {layer.name} to select (drag a box for several), then press "
            f"Delete")

    @viewer.bind_key("Control-S", overwrite=True)
    def save_now(_viewer):
        save_cells(image_path, layers, original, labels, names, say)
        refresh()

    print(
        "EDITING IS ON.\n"
        "  Ctrl-A  add cells: click each one you can see and the model missed\n"
        "  Ctrl-E  pick cells: click (or box-drag) the wrong ones, then press Delete\n"
        "  Ctrl-S  save now. It also saves when you close the window, and a visit that\n"
        "          changed nothing writes nothing.\n"
        "A cell you add is FILLED; the model's are hollow rings. Which region it counts in\n"
        "comes from the outlines, so it does not matter which layer you add it to -- but a\n"
        "click outside every outline cannot be counted and will be reported.\n"
        "The app re-reads its numbers when this window closes, so they update themselves.\n"
    )

    refresh()

    try:
        napari.run()
    finally:
        # In `finally:` for the same reason draw_regions saves there -- the work reaches the
        # disk even if the window goes down badly. Nothing here can raise past this point
        # without the edits being written first.
        save_cells(image_path, layers, original, labels, names)


def check():
    """The editing rules, without napari, an image, or any of the real files.

    WHAT THIS CAN AND CANNOT TEST. napari will not run offscreen on this machine, so the
    window is out of reach; everything that decides what gets WRITTEN is not, and that is
    where a mistake costs a number. Fake layers are enough because gather() only ever asks a
    layer for `.data`.
    """
    import tempfile

    class FakeLayer:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=float).reshape(-1, 2)
            self.metadata = {"counted": True}

    # A label image: rows 0-9 are dNCM, rows 10-19 are CMM, the rest is nowhere.
    labels = np.zeros((30, 30), dtype=np.uint8)
    labels[:10] = 1
    labels[10:20] = 2
    names = {1: "dNCM", 2: "CMM"}

    original = by_pixel(
        [
            {"axis-0": 1, "axis-1": 1, "region": "dNCM", "probability": 0.99, "source": "model"},
            {"axis-0": 2, "axis-1": 2, "region": "dNCM", "probability": 0.98, "source": "model"},
            {"axis-0": 12, "axis-1": 3, "region": "CMM", "probability": 0.97, "source": "model"},
        ]
    )

    assert set(original) == {(1, 1), (2, 2), (12, 3)} and len(original[(1, 1)]) == 1

    # Nothing touched: the same three cells, no edits, and the rows come back verbatim.
    kept = dict(original)
    rows, added, removed, outside = gather(
        [FakeLayer([[1, 1], [2, 2]]), FakeLayer([[12, 3]])], kept, labels, names
    )
    assert len(rows) == 3 and not added and not removed and not outside
    assert rows[0]["probability"] == 0.99, "an untouched cell must keep its own score"

    # One deleted, one added inside CMM, one added outside every region.
    rows, added, removed, outside = gather(
        [FakeLayer([[1, 1], [15, 15], [25, 25]])], dict(original), labels, names
    )
    assert len(added) == 1 and added[0]["region"] == "CMM", added
    assert added[0]["source"] == "hand" and added[0]["probability"] == HAND_PROBABILITY
    assert {(r["axis-0"], r["axis-1"]) for r in removed} == {(2, 2), (12, 3)}, removed
    assert outside == [(25, 25)], outside
    assert len(rows) == 2, "the cell outside the outlines must not be counted"

    # Sub-pixel drift is the same cell; half a pixel of drag is a new one.
    rows, added, removed, _ = gather(
        [FakeLayer([[1.2, 0.9], [2, 2], [12, 3]])], dict(original), labels, names
    )
    assert not added and not removed, (added, removed)

    rows, added, removed, _ = gather(
        [FakeLayer([[1, 1], [2, 2], [12, 4.6]])], dict(original), labels, names
    )
    assert len(added) == 1 and len(removed) == 1, (added, removed)

    # The same cell on two layers, and the same pixel clicked twice, is one cell.
    rows, added, _, _ = gather(
        [FakeLayer([[1, 1], [5, 5]]), FakeLayer([[1, 1], [5, 5]])], dict(original), labels, names
    )
    assert len(added) == 1 and sum(1 for r in rows if r["axis-0"] == 1) == 1, rows

    # No masks at all: deletions still work, additions are reported rather than guessed.
    rows, added, removed, outside = gather(
        [FakeLayer([[1, 1], [9, 9]])], dict(original), None, None
    )
    assert not added and len(removed) == 2 and outside == [(9, 9)]

    assert tally(rows) == {"dNCM": 1}, tally(rows)

    # A cell inside two overlapping outlines is written twice by the counter, once per region.
    # Both rows have to survive an untouched save, and deleting the cell has to delete both --
    # see by_pixel for what goes wrong if the pixel holds only one of them.
    twice = by_pixel(
        [
            {"axis-0": 5, "axis-1": 5, "region": "dNCM", "probability": 0.9, "source": "model"},
            {"axis-0": 5, "axis-1": 5, "region": "HP", "probability": 0.9, "source": "model"},
        ]
    )

    rows, added, removed, _ = gather([FakeLayer([[5, 5]])], dict(twice), labels, names)
    assert len(rows) == 2 and not added and not removed, (rows, added, removed)

    rows, added, removed, _ = gather([FakeLayer([])], dict(twice), labels, names)
    assert not rows and len(removed) == 2, (rows, removed)

    print("editing rules: all cases correct")

    # AND THE FILES, in a folder of their own. The summary is the part with a history and
    # other sections' rows to lose, so it is checked against a real merge rather than by eye.
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        summary_at = folder / "ps6_counts.csv"
        history_at = folder / "history.csv"

        pd.DataFrame(
            [
                {"image": "mine.TIF", "region": "dNCM", "detected": 900, "classified": 2,
                 "tissue_px": 100, "threshold": 0.97, "area_mm2": 0.5,
                 "density_per_mm2": 4.0, "detected_density_per_mm2": 1800.0},
                {"image": "mine.TIF", "region": "CMM", "detected": 800, "classified": 1,
                 "tissue_px": 100, "threshold": 0.97, "area_mm2": 0.25,
                 "density_per_mm2": 4.0, "detected_density_per_mm2": 3200.0},
                {"image": "somebody else.TIF", "region": "dNCM", "detected": 700,
                 "classified": 9, "tissue_px": 100, "threshold": 0.97, "area_mm2": 1.0,
                 "density_per_mm2": 9.0, "detected_density_per_mm2": 700.0},
            ]
        ).to_csv(summary_at, index=False)

        rows, added, removed, _ = gather(
            [FakeLayer([[1, 1], [15, 15], [16, 16]])], dict(original), labels, names
        )
        message = update_summary(
            "mine.TIF", rows, added, removed, path=summary_at, history=history_at
        )

        after = pd.read_csv(summary_at)
        mine = after[after["image"] == "mine.TIF"].set_index("region")

        assert len(after) == 3, after
        assert (after["image"] == "somebody else.TIF").sum() == 1, "another section was lost"
        assert after.loc[after["image"] == "somebody else.TIF", "classified"].iloc[0] == 9

        assert mine.loc["dNCM", "classified"] == 1, mine
        assert mine.loc["dNCM", "density_per_mm2"] == 2.0, "0.5 mm2, so 1 cell is 2.0"
        assert mine.loc["CMM", "classified"] == 2, mine
        assert mine.loc["CMM", "density_per_mm2"] == 8.0, "0.25 mm2, so 2 cells is 8.0"
        assert mine.loc["dNCM", "detected"] == 900, "the detector's own number must not move"
        assert mine.loc["dNCM", "hand_edits"] == "+0-1", mine
        assert mine.loc["CMM", "hand_edits"] == "+2-1", mine

        assert len(pd.read_csv(history_at)) == 2, "both edited rows belong in the history"
        assert "2 rows updated" in message and "1 other rows kept" in message, message

        # A section the summary has never heard of: say so, change nothing.
        untouched = pd.read_csv(summary_at)
        message = update_summary(
            "stranger.TIF", rows, added, removed, path=summary_at, history=history_at
        )
        assert "no rows for this section" in message, message
        assert pd.read_csv(summary_at).equals(untouched), "an unknown section changed the file"

        # The edit log: appended to, never rewritten.
        log_at = folder / "edits.csv"
        log_edits("mine.TIF", added, removed, log_at)
        log_edits("mine.TIF", added, [], log_at)
        log = pd.read_csv(log_at)

        assert list(log["action"]) == [
            "added", "added", "removed", "removed", "added", "added",
        ], log
        assert list(log.columns) == EDIT_COLUMNS, log.columns
        assert log["at"].nunique() >= 1

    print("files: the summary keeps other sections, the log only grows")

    # AND THE WHOLE SAVE, twice over. The second call is the one that matters: Ctrl-S followed
    # by closing the window calls this twice with no edits in between, and the promise is that
    # the second one writes nothing at all.
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        paths = {
            "counts": folder / "cells.csv",
            "edits": folder / "edits.csv",
            "summary": folder / "summary.csv",
            "history": folder / "history.csv",
        }

        pd.DataFrame(
            [{"image": "mine.TIF", "region": "CMM", "detected": 800, "classified": 1,
              "tissue_px": 100, "threshold": 0.97, "area_mm2": 0.5,
              "density_per_mm2": 2.0, "detected_density_per_mm2": 1600.0}]
        ).to_csv(paths["summary"], index=False)

        said = []
        state = dict(original)
        layer = FakeLayer([[1, 1], [2, 2], [12, 3], [15, 15]])

        added, removed, outside = save_cells(
            "mine.TIF", [layer], state, labels, names, said.append, paths
        )

        assert len(added) == 1 and not removed and not outside, (added, removed, outside)
        assert paths["counts"].exists()

        written = pd.read_csv(paths["counts"])
        assert len(written) == 4 and list(written.columns) == CELL_COLUMNS, written
        assert (written["source"] == "hand").sum() == 1, written
        assert written["probability"].is_monotonic_decreasing, "kept in score order"
        assert len(state) == 4, "the in-memory copy must match what was written"

        # Nothing changed since: no write, and a sentence saying so.
        stamp = paths["counts"].stat().st_mtime_ns
        again = pd.read_csv(paths["edits"])

        added, removed, _ = save_cells(
            "mine.TIF", [layer], state, labels, names, said.append, paths
        )

        assert not added and not removed
        assert paths["counts"].stat().st_mtime_ns == stamp, "an unchanged save touched the file"
        assert pd.read_csv(paths["edits"]).equals(again), "an unchanged save wrote to the log"
        assert any("nothing was written" in line for line in said), said

        # And a deletion after a save diffs against the SAVED state, not the loaded one.
        layer.data = np.asarray([[1, 1], [15, 15]], dtype=float)
        added, removed, _ = save_cells(
            "mine.TIF", [layer], state, labels, names, said.append, paths
        )

        assert not added and len(removed) == 2, (added, removed)
        assert len(pd.read_csv(paths["counts"])) == 2

    print("save: writes once, says so, and a second save with no edits changes nothing")

    return 0


if __name__ == "__main__":
    main()
