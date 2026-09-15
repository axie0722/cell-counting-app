"""Count PS6 cells per region, and report density.

Density (cells/mm2) is the comparable number: dNCM, vNCM and CMM differ in area,
so raw counts cannot be compared between regions or between birds. Pixel size
comes from the scanner's own metadata, so the conversion is exact.

Counting rule is center-inside-region, with the patch taken from the full image.
Training erodes region edges by 20 px because negatives near a boundary are
untrustworthy; counting must not, or every ROI silently eats the cells along its
rim.

Both counts are reported: `detected` is every candidate the annulus filter
proposed, `classified` is what the CNN kept. The gap between them is the CNN's
contribution, and while its cross-bird precision is around 40% the classified
number is provisional -- treat it as a lower bound pending better training data.

WRITES, per section: "<stem> region areas.csv" (the area cache the app reads, always) and
"<stem> counted cells.csv" (every kept cell, only with --points). Plus ps6_counts.csv for
the whole run -- note the app reads the per-section sidecars, never that one. Every one of
those goes through save_csv, so a count that is stopped part-way leaves the previous file
untouched rather than half of a new one.

A COUNT ADDS; IT NEVER ERASES SOMEBODY ELSE'S NUMBERS. ps6_counts.csv used to be overwritten
with just the sections of the run that finished last, so counting one section through the app
reduced the whole folder's table to three rows. Alice, 2026-08-20: *"i recounted gre595 by
accident and the area and density stats disappeared"*. Now merge_run keeps every other
section's rows, save_previous keeps the cells the recount replaced, and every run is appended
to ps6_counts_history.csv. An accidental recount costs forty minutes and no data.

SURVIVING THE THING THAT LAUNCHED IT. A count is 35-45 minutes and is usually started by
the app, which pipes its output into a window. If that window closes, the pipe breaks, and
the next print would normally kill the run -- see ignore_a_lost_listener, which makes it go
quiet and carry on instead. Being stopped is a decision someone makes, never a side effect
of nobody watching.

QUOTE THE FILENAME. These names contain spaces, so an unquoted one arrives as several
arguments and is refused rather than counted as the wrong section.

Usage:
  python count_cells.py                      # every bird with regions drawn (hours)
  python count_cells.py "Gre595_....TIF"     # one image
  python count_cells.py "..." --points       # also save every kept cell's position
  python count_cells.py "..." --tta          # average 8 orientations (see classify)
  python count_cells.py --check              # check the argument parsing only

  python show_counts.py "..."                # then look at what --points saved
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ps6
from train_cnn import SmallCNN, pick_device, standardise_batch

# The ResNet-18 backbone, not the SmallCNN it replaced. Measured on the blind grid at
# MATCHED RECALL -- the only fair comparison, since the two ship at different thresholds:
#
#   recall   SmallCNN precision   ResNet precision
#     60%          73.9%               78.2%
#     70%          49.1%               68.9%
#     80%          26.4%               56.4%
#
# It wins everywhere and the gap widens as recall rises, where SmallCNN's precision
# collapses. Cross-bird AP 0.726 vs 0.653 agrees, so this is two independent measurements
# rather than one lucky grid.
#
# LICENCE: the backbone starts from ImageNet pretrained weights, which are licensed for
# NON-COMMERCIAL RESEARCH USE ONLY. Alice confirmed 2026-08-11 that this lab work is
# non-commercial research. If that ever changes, retrain from scratch or revert to
# ps6_cnn.pt, which has no such restriction.
#
# BESIDE THIS FILE, not in the working directory. The weights are part of the program, so they
# are found the way the program finds itself: a bare "ps6_cnn_backbone.pt" means "in whatever
# folder this was started from", which is the project only by habit. The app now launches counts
# from anywhere (see app_actions.HERE), and on another computer the habit does not exist at all.
#
# Only the INPUTS are anchored. OUTPUT_PATH and HISTORY_PATH below stay relative on purpose:
# they are results, and results belong where the person running the command is standing. The app
# gives its subprocesses a working directory of the app folder, so for the app nothing moves.
HERE = Path(__file__).resolve().parent

MODEL_PATH = str(HERE / "ps6_cnn_backbone.pt")
FALLBACK_MODEL_PATH = str(HERE / "ps6_cnn.pt")
OUTPUT_PATH = "ps6_counts.csv"

# Append-only: one block of rows per run, with the time it finished. Nothing is ever removed
# from here, which is what makes an accidental recount recoverable -- see merge_run.
HISTORY_PATH = "ps6_counts_history.csv"

# Every flag this script accepts. Checked rather than assumed, and here is why.
#
# THE BUG THIS FIXES. main() used to build its list of image arguments by discarding
# anything starting with "--". An unrecognised flag was therefore discarded too, leaving no
# image argument at all -- and no image argument means "count every bird that has regions
# drawn". So `python count_cells.py --help` did not print help. It started a full-dataset
# run, roughly ten hours of computation, silently. That happened on 2026-08-13.
#
# The general shape of the mistake is worth naming: a filter written as "ignore what I do
# not recognise" turns a typo into a different valid command. Anything not understood has to
# be an error, because the alternative is guessing, and here the guess was the most
# expensive operation the script has.
KNOWN_FLAGS = {"--points", "--tta"}

# Asked for help in any of the usual ways. Handled explicitly so none of them falls through
# to a run: "-h" does not start with "--", so the old filter kept it and it became an image
# name, giving "No such image: -h" instead of the help that was asked for.
HELP_FLAGS = {"--help", "-h", "-help", "help"}


def load_model(device, model_path=MODEL_PATH):
    """Load the trained classifier and the threshold stored with it."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No {model_path}. Run train_backbone.py first, or pass "
            f"{FALLBACK_MODEL_PATH} to use the older SmallCNN."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Build whatever the checkpoint says it is, rather than assuming SmallCNN. Loading
    # ResNet weights into a SmallCNN fails with a wall of shape mismatches that says
    # nothing about the actual problem -- the same trap score_blind_grid.py and
    # sweep_threshold.py both had. Reading the field means a new architecture can be
    # counted with without editing this file.
    architecture = checkpoint.get("architecture", "SmallCNN")

    if architecture == "BackboneCNN":
        from train_backbone import BackboneCNN

        # pretrained=False: the checkpoint's weights replace them immediately, so fetching
        # ImageNet weights here would be a download for nothing.
        model = BackboneCNN(pretrained=False).to(device)
    elif architecture == "SmallCNN":
        model = SmallCNN().to(device)
    else:
        raise SystemExit(f"Unknown architecture {architecture!r} in {model_path}.")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, float(checkpoint["threshold"]), checkpoint


def dihedral(batch, index):
    """One of the 8 flips and quarter-turns of a batch of patches.

    The same symmetry group train_cnn.augment samples from. PS6 cells are
    rotationally symmetric annuli, so every one of these preserves the label.
    """
    if index >= 4:
        batch = torch.flip(batch, dims=[3])

    turns = index % 4

    return torch.rot90(batch, turns, dims=[2, 3]) if turns else batch


def classify(model, green, candidates, device, geometry=None, tta=False):
    """Probability that each candidate is a real cell.

    With `tta`, each patch is scored in all 8 orientations and the probabilities are
    averaged. Training already shows the model all 8, but inference sees only one, so
    the stored orientation gets an arbitrary vote; averaging removes that arbitrariness
    and is the standard way to spend compute for accuracy when the data has a symmetry.

    It costs 8x inference and it CHANGES THE SCORE DISTRIBUTION: a mean over 8 votes is
    pulled down by any orientation that scores lower, so a threshold tuned without TTA
    reads as a recall loss when applied with it. Never compare the two at a fixed
    threshold -- sweep, and read at matched recall.
    """
    geometry = geometry or ps6.Geometry()
    height, width = green.shape
    half = geometry.patch_half_width

    # Candidates too close to the image edge have no full patch. They keep a
    # probability of 0 rather than being dropped, so indices stay aligned.
    has_patch = (
        (candidates[:, 0] >= half)
        & (candidates[:, 0] < height - half)
        & (candidates[:, 1] >= half)
        & (candidates[:, 1] < width - half)
    )

    probabilities = np.zeros(len(candidates), dtype=np.float32)
    usable = np.flatnonzero(has_patch)

    if not len(usable):
        return probabilities

    # Cut at the CNN's own input size: there is nothing to crop at inference, and
    # standardisation happens on the batch, matching the training path exactly.
    patches = np.stack(
        [ps6.cut_patch(green, *candidates[index], geometry) for index in usable]
    ).astype(np.float32)

    views = 8 if tta else 1

    with torch.no_grad():
        for start in range(0, len(patches), 2048):
            chunk = patches[start : start + 2048]
            batch = torch.from_numpy(chunk).unsqueeze(1).to(device)

            # Standardise once, before the transforms. A flip or quarter-turn cannot
            # change a patch's mean or variance, so standardising per view would repeat
            # identical arithmetic 8 times.
            batch = standardise_batch(batch)

            total = None
            for view in range(views):
                scores = torch.sigmoid(model(dihedral(batch, view)))
                total = scores if total is None else total + scores

            probabilities[usable[start : start + 2048]] = (
                (total / views).cpu().numpy()
            )

    return probabilities


def points_path(image_path):
    """Where a section's kept cell coordinates go. See ps6.counts_path."""
    return ps6.counts_path(image_path)


def save_csv(frame, path):
    """Write a table to `path` all at once, so nothing can ever read half of it.

    WHY NOT JUST to_csv. Writing a CSV is not one action -- pandas opens the file, writes a
    header, then writes rows until it runs out. A process stopped in the middle of that
    leaves a file that exists, has the right name, has a header, and ends halfway through a
    row. Nothing downstream can tell that apart from a real result: app_table.py reads it,
    finds fewer cells than were actually counted, and reports a density that is quietly
    wrong. A wrong number that looks right is the worst outcome available here.

    So write a neighbour file first and then rename it over the target. `os.replace` is
    atomic within one filesystem, which means at every single instant the real path holds
    either the complete previous file or the complete new one -- there is no moment at which
    it holds a partial one. The temporary sits beside the target rather than in /tmp because
    a rename ACROSS filesystems is not atomic and os.replace refuses it outright.

    This is what makes stopping a count safe, and it is the reason the app can now offer to
    stop one. It costs a rename.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")

    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

    return path


def save_previous(path):
    """Keep the file about to be overwritten as "<name>.previous.csv". Returns it, or None.

    WHY A RECOUNT NEEDS THIS. Counting a section again replaces its counted-cells file, and a
    recount started by accident is a normal mistake: the section list is a list of names, the
    Count button is one click, and the section that was counted last night is the one already
    selected. Alice, 2026-08-20: *"i recounted gre595 by accident"*. Forty minutes is a cost
    worth paying by mistake; a number you cannot get back is not.

    ONE generation deep, on purpose. A folder of "counted cells.previous.previous.csv" answers
    a question nobody asks, and the run-by-run record already exists in
    ps6_counts_history.csv. This is here for the accident noticed straight away.
    """
    path = Path(path)

    if not path.exists():
        return None

    kept = path.with_name(path.stem + ".previous.csv")
    os.replace(path, kept)

    return kept


def merge_run(rows, path=OUTPUT_PATH, history_path=HISTORY_PATH):
    """Add this run's rows to the folder's table without dropping anyone else's.

    THE BUG THIS FIXES, 2026-08-20. This file used to be written as `save_csv(table, path)`
    where `table` held only the sections THIS run counted. The app counts one section per job,
    so every count truncated the whole folder's table to three rows: Alice recounted one
    Gre595 section and every other section's area and density vanished from it. The rows were
    not corrupted or moved, they were deleted, and nothing warned.

    Merging on `image` rather than appending, because a recount must REPLACE that section's
    rows -- appending would leave two answers for one section and no way to tell which is
    current. Old rows for the sections in this run are dropped; every other row is kept
    verbatim, including sections counted by a version of this script that wrote other columns
    (pandas fills the gaps with blanks, which reads as "not measured" and is true).

    The history file is appended to and never pruned. It is the actual safety net: `image` is
    not unique there, `counted_at` distinguishes the runs, and an accidental recount can be
    undone by reading the row from before it.

    Returns (kept, replaced): how many rows came from other sections, and how many rows this
    run overwrote. Both go in the run's last line, because "wrote 3 rows" and "wrote 3 rows,
    replacing 3, keeping 45" describe very different events.
    """
    table = pd.DataFrame(rows)

    if not len(table):
        return 0, 0

    counted_now = set(table["image"])
    existing = read_table(path)
    replaced = 0
    kept = 0

    if existing is not None and "image" in existing.columns:
        staying = existing[~existing["image"].isin(counted_now)]
        replaced = len(existing) - len(staying)
        kept = len(staying)
        table = pd.concat([staying, table], ignore_index=True)

    save_csv(order_rows(table), path)

    # Stamped once for the whole run, not per section: the point is to identify the RUN, and a
    # per-section time would make two rows of one run look like two runs.
    stamped = pd.DataFrame(rows)
    stamped.insert(0, "counted_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    older = read_table(history_path)
    save_csv(
        pd.concat([older, stamped], ignore_index=True) if older is not None else stamped,
        history_path,
    )

    return kept, replaced


def read_table(path):
    """A CSV as a DataFrame, or None if it is missing, empty or unreadable.

    Unreadable counts as missing rather than as an error. This is called to decide what to
    KEEP; a summary file someone half-edited in Excel must not be able to stop a count that
    took forty minutes from saving its own result.
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        table = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        print(f"  could not read {path}; leaving it out of the merge")

        return None

    return table if len(table) else None


def order_rows(table):
    """Sort by section, then by region in ps6.REGION_NAMES order.

    Alphabetical would put CMM before dNCM and vNCM, which is not the order anyone reads a
    section in. Sorting at all is so that a merged file looks the same every time rather than
    growing whichever section was counted most recently at the bottom.
    """
    order = {name: index for index, name in enumerate(ps6.REGION_NAMES)}
    table = table.copy()
    table["_region_order"] = table["region"].map(lambda name: order.get(name, len(order)))

    return (
        table.sort_values(["image", "_region_order"], kind="stable")
        .drop(columns="_region_order")
        .reset_index(drop=True)
    )


class DeafStream:
    """An output stream that goes quiet instead of raising when nobody is reading.

    THE BUG THIS FIXES, seen 2026-08-20. Alice closed the app during a count of Gre595 and
    the count stopped -- 40 minutes of work lost with nothing written. The app runs the count
    as a subprocess with its output piped into the window, so closing the window breaks the
    pipe; the next write to it raises BrokenPipeError, which nothing catches, so the run dies.
    Whether it happens at all is down to luck: prints to stdout sit in an 8 KB buffer and may
    never reach the pipe, but app_actions.run merges stderr into the same pipe and stderr is
    NOT buffered, so one warning from torch or pandas is enough. "It sometimes survives" is
    worse than either answer, because it cannot be planned around.

    The fix is not to catch the error at each print -- there are dozens, and the next one
    added would forget. It is to make the streams themselves deaf: on the first failure,
    point the real file descriptors 1 and 2 at /dev/null and carry on. dup2 rather than only
    swapping the Python object, because the original buffered writer is flushed again by the
    interpreter at exit, and that flush would raise too.

    A count now ends for exactly one reason: someone decided to end it. Being unobserved is
    not a reason.
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        try:
            return self.stream.write(text)
        except (BrokenPipeError, OSError, ValueError):
            self.go_quiet()

            return len(text)

    def flush(self):
        try:
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.go_quiet()

    def go_quiet(self):
        quiet = open(os.devnull, "w")

        # Both descriptors, because the pipe was shared: stderr was merged into stdout, so
        # if one end is gone the other is too.
        for descriptor in (1, 2):
            try:
                os.dup2(quiet.fileno(), descriptor)
            except OSError:
                pass

        self.stream = quiet

    def __getattr__(self, name):
        """Anything else -- encoding, isatty, fileno -- behaves like the real stream."""
        return getattr(self.stream, name)


def ignore_a_lost_listener():
    """Make this process's output deaf to a broken pipe. See DeafStream."""
    sys.stdout = DeafStream(sys.stdout)
    sys.stderr = DeafStream(sys.stderr)


def write_areas(image_path, area_rows):
    """Cache the region areas this count already measured.

    WHY THE COUNT WRITES THIS. The app reads areas from "<stem> region areas.csv" and
    nowhere else, and until now only `app_table.py --areas` wrote it -- so a freshly
    counted section showed its cell counts with a blank area and a blank DENSITY, which is
    the only comparable number the lab actually wants. The app even said "Areas will be
    computed on the first count", which nothing did. Writing it here makes that true.

    It costs nothing: `tissue_px` is the same tissue-intersected pixel count that
    app_table.compute_areas reopens the TIF to recalculate, so this replaces a second
    full-image pass with a three-line CSV. Because both numbers now come from one
    measurement, the app's density can no longer disagree with the count's own.

    Rows are ordered by ps6.REGION_NAMES so the file reads the same way every time; the
    app parses it into a dict by region, so order is for the human reading it.
    """
    order = {name: index for index, name in enumerate(ps6.REGION_NAMES)}
    ordered = sorted(area_rows, key=lambda row: order.get(row["region"], len(order)))

    path = save_csv(
        pd.DataFrame(ordered, columns=ps6.AREA_COLUMNS), ps6.areas_path(image_path)
    )

    print(f"    wrote {path}")


def count_image(image_path, model, threshold, device, save_points=False, tta=False):
    """Return one row per region for a single section.

    With `save_points`, the coordinates of every kept cell are written next to the
    counts. Counting alone reduces a section to one number per region, which is what
    the lab needs but is impossible to check: a count cannot be inspected, only
    believed. The coordinates make the same run reviewable in napari, and they are a
    by-product of counting rather than a second pass, so the points shown are exactly
    the cells the reported number is made of.
    """
    print(f"\n=== {Path(image_path).name} ===")

    image, green = ps6.load_green(image_path)
    del image

    microns = ps6.pixel_size_um(image_path)
    mm_per_pixel = (microns / 1000.0) if microns else None

    geometry = ps6.Geometry(microns)
    print(f"  {geometry.describe()}")

    tissue = ps6.compute_tissue(green, geometry)
    regions = ps6.resolve_regions(image_path, green.shape, tissue)

    print(f"  regions: {list(regions)}")

    # Detect over the union of regions only, so the candidate budget is spent
    # where it will be counted rather than across the whole slide.
    union = np.zeros(green.shape, dtype=bool)
    for mask in regions.values():
        union |= mask
    union &= tissue

    print(f"  detecting over {union.sum() / 1e6:.1f} Mpx...")
    candidates, _, _ = ps6.detect_candidates(green, union, geometry)

    probabilities = classify(model, green, candidates, device, geometry, tta=tta)
    is_cell = probabilities >= threshold

    rows = []
    points = []
    area_rows = []

    for name, mask in regions.items():
        in_region = mask[candidates[:, 0], candidates[:, 1]]

        detected = int(in_region.sum())
        classified = int((in_region & is_cell).sum())

        if save_points:
            keep = np.flatnonzero(in_region & is_cell)
            points.append(
                pd.DataFrame(
                    {
                        "axis-0": candidates[keep, 0],
                        "axis-1": candidates[keep, 1],
                        "region": name,
                        "probability": probabilities[keep],
                    }
                )
            )

        pixels = int((mask & tissue).sum())

        row = {
            "image": Path(image_path).name,
            "region": name,
            "detected": detected,
            "classified": classified,
            "tissue_px": pixels,
            "threshold": threshold,
        }

        if mm_per_pixel:
            area = pixels * mm_per_pixel**2
            row["area_mm2"] = round(area, 4)
            row["density_per_mm2"] = round(classified / area, 1) if area else None
            row["detected_density_per_mm2"] = round(detected / area, 1) if area else None

            print(
                f"    {name:6s} detected {detected:5d}  classified {classified:5d}  "
                f"{area:6.3f} mm2  {row['density_per_mm2']:8.1f} cells/mm2"
            )
        else:
            print(
                f"    {name:6s} detected {detected:5d}  classified {classified:5d}  "
                f"{pixels:,} px (no pixel size, so no density)"
            )

        rows.append(row)
        area_rows.append(
            {
                "region": name,
                # None rather than 0 when the pixel size is unknown, matching
                # app_table.compute_areas: 0 would divide into an infinite density.
                "area_mm2": row.get("area_mm2"),
                "tissue_px": pixels,
            }
        )

    write_areas(image_path, area_rows)

    if save_points:
        # Sorted by score so the least confident calls are last: those are the ones
        # worth looking at first when checking whether the threshold is right.
        table = pd.concat(points, ignore_index=True).sort_values(
            "probability", ascending=False
        )

        # The cells this count is about to replace, kept where they can be found. See
        # save_previous: a recount is a normal accident and must not be able to lose a number.
        kept = save_previous(points_path(image_path))

        save_csv(table, points_path(image_path))
        print(f"    wrote {points_path(image_path)} ({len(table)} cells)")

        if kept is not None:
            print(f"    the previous count is kept as {kept.name}")

    return rows


def find_images(argument):
    if argument:
        path = argument

        if not Path(path).exists():
            candidate = Path("birds") / path
            if not candidate.exists():
                raise SystemExit(f"No such image: {path}")
            path = str(candidate)

        return [path]

    # Every configured bird that has regions drawn.
    images = []

    for bird in ps6.BIRDS:
        if Path(ps6.region_path(bird["image"])).exists():
            images.append(bird["image"])
        else:
            print(f"skipping {bird['name']}: no regions drawn")

    if not images:
        raise SystemExit(
            "No regions drawn yet. Start with:\n"
            "  python draw_regions.py"
        )

    return images


def parse_arguments(argv):
    """Split the command line into (image, model_path, save_points, tta).

    Refuses anything it does not understand instead of ignoring it -- see KNOWN_FLAGS for
    what that mistake cost. Returns image=None for the deliberate full-dataset run, which is
    now only reachable by passing no arguments at all.

    Separate from main() so it can be checked without loading a model or touching an image;
    the cases are at the bottom of this file.
    """
    if any(a in HELP_FLAGS for a in argv):
        raise SystemExit(__doc__.strip())

    flags = [a for a in argv if a.startswith("-")]
    unknown = [f for f in flags if f not in KNOWN_FLAGS]

    if unknown:
        raise SystemExit(
            f"Unknown option{'s' if len(unknown) > 1 else ''}: {' '.join(unknown)}\n"
            f"Known options: {' '.join(sorted(KNOWN_FLAGS))}\n"
            f"Run with --help for usage."
        )

    # A .pt argument selects the checkpoint, so a retuned threshold can be counted
    # with without editing this file or overwriting the shipped model.
    models = [a for a in argv if a.endswith(".pt")]
    images = [a for a in argv if not a.startswith("-") and not a.endswith(".pt")]

    if len(models) > 1:
        raise SystemExit(f"More than one model given: {' '.join(models)}")

    # Extra positional arguments used to be silently dropped, which means an unquoted
    # filename -- these names are full of spaces -- counted the wrong section without
    # saying so. The most likely cause is worth naming in the message.
    if len(images) > 1:
        raise SystemExit(
            f"More than one image given:\n  " + "\n  ".join(images) + "\n"
            "These filenames contain spaces, so quote them:\n"
            '  python count_cells.py "Gre595_RH_3-1-4_NCM_Slide 1.TIF" --points'
        )

    return (
        images[0] if images else None,
        models[0] if models else MODEL_PATH,
        "--points" in argv,
        "--tta" in argv,
    )


def main():
    image, model_path, save_points, tta = parse_arguments(sys.argv[1:])

    # Before anything expensive starts, and after the arguments are read so a typo can still
    # be reported to whoever is watching.
    ignore_a_lost_listener()

    device = pick_device()

    model, threshold, checkpoint = load_model(device, model_path)

    print(f"{model_path}, threshold {threshold:.3f}")

    if "threshold_note" in checkpoint:
        print(f"  {checkpoint['threshold_note']}")

    if "mean_average_precision" in checkpoint:
        print(
            f"model cross-bird AP {checkpoint['mean_average_precision']:.3f} "
            f"-- classified counts are provisional at this accuracy"
        )

    rows = []

    images = find_images(image)

    # Said out loud before any work starts, because this is the expensive branch: one section
    # takes 35-45 minutes, so the whole set is most of a day. Announcing it means a command
    # meant for one section that ran for all of them is obvious in the first line of output
    # rather than an hour later.
    if image is None:
        print(f"\ncounting ALL {len(images)} sections with regions drawn -- "
              f"roughly {len(images) * 40 // 60} hours. Ctrl-C now if that is not what you meant.\n")

    for image_path in images:
        rows.extend(
            count_image(
                image_path, model, threshold, device, save_points=save_points, tta=tta
            )
        )

    table = pd.DataFrame(rows)
    kept, replaced = merge_run(rows)

    print(f"\nsaved {OUTPUT_PATH}: {len(table)} row(s) from this run, "
          f"{replaced} replaced, {kept} left as they were")
    print(f"every run is also appended to {HISTORY_PATH}")

    if "density_per_mm2" in table.columns:
        print("\n=== density by region (cells/mm2) ===")
        summary = table.pivot_table(
            index="image",
            columns="region",
            values="density_per_mm2",
        )
        print(summary.to_string())


def check_arguments():
    """Every way the command line can be read, checked without loading anything.

    Here because the bug this guards against was invisible: the wrong reading of `--help`
    produced no error, just ten hours of unrequested computation. A wrong parse that looks
    like a valid command can only be caught by stating what each input should mean.

    Run with: python count_cells.py --check
    """
    image, model, points, tta = parse_arguments([])
    assert (image, model, points, tta) == (None, MODEL_PATH, False, False), "bare"

    image, model, points, tta = parse_arguments(["a.TIF", "--points"])
    assert (image, model, points, tta) == ("a.TIF", MODEL_PATH, True, False), "one image"

    image, model, _, tta = parse_arguments(["a.TIF", "other.pt", "--tta"])
    assert (image, model, tta) == ("a.TIF", "other.pt", True), "model and tta"

    # The heart of it: an unknown flag must not read as "count everything".
    for bad in (["--verbose"], ["--point"], ["-p"], ["a.TIF", "--dry-run"]):
        try:
            parse_arguments(bad)
        except SystemExit as error:
            assert "Unknown option" in str(error), f"{bad} gave {error}"
        else:
            raise AssertionError(f"{bad} was accepted; it must be refused")

    for asked in (["--help"], ["-h"], ["a.TIF", "--help"]):
        try:
            parse_arguments(asked)
        except SystemExit as error:
            assert "Usage" in str(error), f"{asked} gave {error}"
        else:
            raise AssertionError(f"{asked} did not show help")

    # An unquoted filename with spaces arrives as several arguments.
    try:
        parse_arguments(["Gre595_RH_3-1-4_NCM_Slide", "1.TIF"])
    except SystemExit as error:
        assert "More than one image" in str(error), str(error)
    else:
        raise AssertionError("unquoted filename was accepted")

    print("argument parsing: all cases correct")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        check_arguments()
    else:
        main()
