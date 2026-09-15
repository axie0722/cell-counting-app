"""What sections exist and how far each one has got. No windows, no buttons.

This is the logic layer of the counting app. Everything here answers questions --
which sections exist, which have outlines, which have been counted -- and returns plain
data. The window (app.py) only calls these functions and displays what comes back.

Keeping the two apart is not tidiness. It means the answers can be checked by running
this file in a terminal and reading the output, which is the only pleasant way to debug:
when a button in a GUI does nothing, you cannot tell whether the button is broken or the
calculation behind it is. It also means the interface is replaceable -- a web version
would rewrite the window and import this file unchanged.

STATE IS DERIVED FROM THE FILESYSTEM, not stored anywhere. A section's progress is read
off which sidecar files exist next to its TIF:

    Gre595_RH_3-1-4_NCM_Slide 1.TIF                    the image
    Gre595_RH_3-1-4_NCM_Slide 1 regions.csv            outlines drawn
    Gre595_RH_3-1-4_NCM_Slide 1 dorsal.csv             dorsal marker clicked
    Gre595_RH_3-1-4_NCM_Slide 1 counted cells.csv      counted

There is deliberately no database and no progress file. A separate record of what has
been done is a record that can disagree with reality -- and this project has already had
one crash mid-run. Because the files ARE the state, that crash cost nothing: whatever had
been written was still true afterwards. Nothing here can go stale.

Nothing in this module opens a TIF. Reading 17 images to build a list would take a minute
and ~2 GB of memory, and the filenames answer the question on their own. Pixel size and
image dimensions need the file, so they are looked up only for a section being counted.

WHICH FOLDER. find_images takes a root and searches it recursively, so the app can be
pointed at any folder on any machine rather than only working from this project directory.
"Beside its TIF" above is now literally true -- see ps6.sidecar_path -- which is what makes
a folder self-contained: copy it somewhere else and the outlines and counts come too.

Usage:
  python app_state.py                # print every section under this folder
  python app_state.py /path/to/images
"""

import sys
from pathlib import Path

import ps6

# Where to look when nobody has said. The project folder itself, scanned recursively --
# which covers the TIFs at the top level and the ones in birds/ without naming either.
DEFAULT_ROOT = Path(".")

# Folder names never scanned. Two kinds, and the distinction matters if you edit this:
#
#   MACHINERY -- .venv holds scikit-image's two sample TIFs, which would otherwise appear
#   as sections. That is not hypothetical; it is why this set exists.
#
#   SNAPSHOTS -- "lambda archive" and "old" hold second copies of real sections. They are
#   excluded rather than deduplicated because a snapshot appearing as a "duplicate note"
#   on every refresh is noise about a decision already made. Both now live inside
#   "archive/", which is listed as well so a future snapshot dropped there is skipped
#   without needing its own entry.
#
# Any folder starting with "." is skipped too, without being listed.
SKIP_FOLDERS = {
    ".venv",
    "__pycache__",
    "runs",
    "wandb",
    "node_modules",
    "archive",
    "lambda archive",
    "old",
}

# Both cases, because a scanner stopped by capitalisation is a scanner that reports "no
# images found" while the person is looking at the folder full of them.
IMAGE_SUFFIXES = {".tif", ".tiff"}

# Status values, in the order a section passes through them. Named constants rather than
# bare strings so a typo is an error the moment it happens instead of a status that
# silently never matches anything.
NEEDS_REGIONS = "needs regions"
NEEDS_DORSAL = "needs dorsal marker"
READY_TO_COUNT = "ready to count"
COUNTED = "counted"


def sidecar_paths(image_path):
    """The three files that may sit beside an image, whether or not they exist.

    Every path comes from the ps6 helper that owns it rather than being rebuilt here.
    ps6.region_path and ps6.dorsal_path are what draw_regions.py writes and what
    count_cells.py reads, so asking them keeps this module agreeing with the rest of the
    project by construction -- if a naming convention ever changes, it changes in one
    place and this follows.
    """
    return {
        "regions": Path(ps6.region_path(image_path)),
        "dorsal": Path(ps6.dorsal_path(image_path)),
        # ps6.counts_path rather than rebuilding the name here. count_cells.points_path()
        # delegates to the same helper, so the writer and this reader cannot drift apart --
        # and asking ps6 costs nothing, where importing count_cells would pull in torch and
        # make a module whose job is listing filenames take seconds to load.
        "counts": Path(ps6.counts_path(image_path)),
    }


def status_for(has_regions, has_dorsal, has_counts):
    """One status word from the three facts.

    Separate from the file-finding on purpose: what is TRUE about a section and how it is
    DESCRIBED to a person are different jobs, and the second changes far more often. It
    also makes the rule testable without any files on disk -- see the checks in main().

    The order matters. Regions are checked before the dorsal marker because drawing the
    outlines is what prompts the dorsal click, and counts are reported last so a section
    that has been counted reads as done even if it is later recounted.
    """
    if not has_regions:
        return NEEDS_REGIONS

    # NCM must be split into dNCM and vNCM to report densities, and ps6.resolve_regions
    # raises without this marker. Worth its own status: it is one click, and a section
    # stuck here looks identical to a finished one in a file listing.
    if not has_dorsal:
        return NEEDS_DORSAL

    if not has_counts:
        return READY_TO_COUNT

    return COUNTED


def skipped(path):
    """True if any part of a path is a folder we never scan."""
    return any(
        part in SKIP_FOLDERS or part.startswith(".") for part in path.parent.parts
    )


def find_images(root=None):
    """Every image under `root`, recursively, at most one per stem.

    Returns (images, duplicates): the paths to use, and (loser, winner) pairs for the
    stems that appeared more than once.

    RECURSIVE, because a person choosing a folder means "my images are in here" and will
    not think to point at the exact subfolder each slide sits in. The cost is that it can
    wander into places that are not data, which is what SKIP_FOLDERS is for.

    SHALLOWEST PATH WINS when a stem repeats. A stem is a section's whole identity here --
    ps6.sidecar_path builds every sidecar name from it, so two images with the same stem
    in different folders would fight over the same outlines and counts. Something has to
    choose, and choosing the shallowest is choosing the copy a person is most likely to
    think of as the real one: files get filed INTO subfolders, so depth means "put aside".

    Ties inside one depth break alphabetically, so the answer is the same on every machine
    rather than depending on the order the filesystem hands back.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT

    if not root.is_dir():
        return [], []

    found = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file() and not skipped(path)
    ]

    images = {}
    duplicates = []

    for path in sorted(found, key=lambda p: (len(p.parts), str(p).lower())):
        if path.stem in images:
            duplicates.append((path, images[path.stem]))
            continue

        images[path.stem] = path

    return [images[stem] for stem in sorted(images)], duplicates


def find_sections(root=None, notes=None):
    """Every section found under `root`, with its progress. Dicts sorted by name.

    Sorted so the list does not reshuffle between runs -- filesystem order is not
    guaranteed, and a table whose rows move around is hard to trust.

    `notes` collects anything a person should know about the scan itself, as opposed to
    about a section -- currently just duplicate stems. Passed in as a list rather than
    printed, because the app has a log pane and print() goes nowhere it can be read. If
    it is None the notes are printed, which is what running this file from a terminal
    wants.
    """
    sections = []

    images, duplicates = find_images(root)

    for loser, winner in duplicates:
        message = f"note: {loser} has the same name as {winner}; using {winner}"

        if notes is None:
            print(f"  {message}")
        else:
            notes.append(message)

    for image_path in images:
        paths = sidecar_paths(str(image_path))
        exists = {name: path.exists() for name, path in paths.items()}

        sections.append(
            {
                "stem": image_path.stem,
                "path": str(image_path),
                "has_regions": exists["regions"],
                "has_dorsal": exists["dorsal"],
                "has_counts": exists["counts"],
                "status": status_for(
                    exists["regions"], exists["dorsal"], exists["counts"]
                ),
            }
        )

    return sorted(sections, key=lambda s: s["stem"])


def summarise(sections):
    """How many sections sit at each status, for a one-line overview."""
    counts = {}

    for section in sections:
        counts[section["status"]] = counts.get(section["status"], 0) + 1

    return counts


def main():
    # The status rule checked against every combination, so it is verified without
    # touching the disk. Three yes/no facts is only eight cases -- small enough to state
    # them all rather than trust that the if-chain reads correctly.
    expected = {
        (False, False, False): NEEDS_REGIONS,
        (False, False, True): NEEDS_REGIONS,
        (False, True, False): NEEDS_REGIONS,
        (False, True, True): NEEDS_REGIONS,
        (True, False, False): NEEDS_DORSAL,
        (True, False, True): NEEDS_DORSAL,
        (True, True, False): READY_TO_COUNT,
        (True, True, True): COUNTED,
    }

    for (regions, dorsal, counts), want in expected.items():
        got = status_for(regions, dorsal, counts)

        if got != want:
            raise AssertionError(
                f"status_for({regions}, {dorsal}, {counts}) gave {got!r}, "
                f"expected {want!r}"
            )

    print(f"status rule: all {len(expected)} cases correct\n")

    root = sys.argv[1] if len(sys.argv) > 1 else None
    sections = find_sections(root)

    if not sections:
        raise SystemExit(
            "No .tif or .tiff files found in this folder or any folder inside it.\n"
            "Run this from the folder holding your images, or pass one:\n"
            "  python app_state.py /path/to/images"
        )

    print(f"{'section':44s} {'status':22s} regions dorsal counts")
    print("-" * 84)

    for section in sections:
        marks = "".join(
            f"   {'yes' if section[key] else ' - ':5s}"
            for key in ("has_regions", "has_dorsal", "has_counts")
        )
        print(f"{section['stem'][:44]:44s} {section['status']:22s}{marks}")

    print()

    for status, count in sorted(summarise(sections).items()):
        print(f"  {count:2d}  {status}")

    print(f"  {len(sections):2d}  total")


if __name__ == "__main__":
    main()
