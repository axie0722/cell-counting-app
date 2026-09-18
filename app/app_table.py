"""Counts, areas and densities per section and region. Still no windows, no buttons.

Step 2 of the app's logic layer, sitting on top of app_state.py: that module answers "what
exists and how far has it got", this one answers "what are the numbers". The window will
call build_table() and display the rows; it will not compute anything itself.

DENSITY IS THE POINT, NOT THE COUNT. Alice: "i told you the cell density could vary a lot
from image because you don't know the context each image was taken in" -- raw counts are
not comparable between sections, because a section that happens to cut through more NCM
has more cells in it without meaning anything. density = count / area is the comparable
figure, so a table that showed counts alone would invite exactly the comparison that is
invalid.

WHY AREAS ARE CACHED TO A SIDECAR. Counts live in "<stem> counted cells.csv" already, but
area does not: computing it needs the tissue mask, which needs the TIF -- about a minute
and ~2 GB to open all 17. A table that recomputed areas on every refresh would freeze the
window for a minute each time it was opened. So areas are written once, to
"<stem> region areas.csv", beside the image like every other piece of state.

count_cells.py writes that file during a count, since it measures the same pixel count
anyway. `--areas` here is now a BACKFILL for sections counted before it did, and for
sections outlined but not yet counted -- the table can show an area before a count exists.

The general habit: when a value is expensive to compute and cheap to store, store it -- but
store it NEXT TO the thing it describes, so it cannot outlive it.

STALENESS IS DETECTED, NOT ASSUMED AWAY. The four sections counted so far were counted
against Alice's older, smaller CMM outlines, so their densities are wrong now. Because both
the counts and the outlines are files, "were these counts made before the outlines
changed?" is answerable from modification times. A stale row says so instead of showing a
confident wrong number.

Usage:
  python app/app_table.py                     # the table, from cached areas
  python app/app_table.py --areas             # compute and cache missing areas (opens TIFs)
  python app/app_table.py --areas --force     # recompute even where a cache exists
"""

import sys
from pathlib import Path

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. See app_path.py, and the same block at
# the top of count_cells.py: the app runs this as its own process for --areas, so it gets app/ on
# sys.path and none of its siblings.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

import numpy as np
import pandas as pd

import ps6
from app_state import COUNTED, find_sections, sidecar_paths

# Column order for the cached area file. Lives in ps6 now that count_cells.py writes this
# file too; re-exported here because that is where readers of this module expect it.
AREA_COLUMNS = ps6.AREA_COLUMNS


def areas_path(image_path):
    """Where a section's cached region areas live, as a Path.

    The name is ps6's now, because count_cells.py writes this file during a count. Kept
    here as a Path-returning wrapper because every caller in the app layer uses .exists().
    """
    return Path(ps6.areas_path(image_path))


def counts_for(image_path):
    """Cells per region from the counted-cells sidecar, or None if not counted.

    Returns a count for every region in ps6.REGION_NAMES, including zeros. A region with
    no cells is a real measurement -- density zero -- and dropping it would make the row
    vanish from the table as though the region did not exist.
    """
    path = sidecar_paths(image_path)["counts"]

    if not path.exists():
        return None

    table = pd.read_csv(path)
    found = table["region"].value_counts().to_dict() if len(table) else {}

    return {name: int(found.get(name, 0)) for name in ps6.REGION_NAMES}


def compute_areas(image_path):
    """Region areas in mm2. Opens the TIF, so callers should cache the result.

    Areas are intersected with the tissue mask, matching count_cells.py exactly: a drawn
    outline can overhang the tissue edge, and counting that overhang as area would inflate
    the denominator and silently deflate every density.
    """
    _, green = ps6.load_green(image_path)
    geometry = ps6.Geometry(ps6.pixel_size_um(image_path))
    tissue = ps6.compute_tissue(green, geometry)
    regions = ps6.resolve_regions(image_path, green.shape, tissue)

    scale_um = ps6.pixel_size_um(image_path)
    mm_per_pixel = scale_um / 1000.0 if scale_um else None

    rows = []

    for name in ps6.REGION_NAMES:
        pixels = int((regions[name] & tissue).sum())

        rows.append(
            {
                "region": name,
                # None rather than 0 when the pixel size is unknown: a missing scale is not
                # an area of zero, and 0 would divide into an infinite density.
                "area_mm2": round(pixels * mm_per_pixel**2, 4) if mm_per_pixel else None,
                "tissue_px": pixels,
            }
        )

    return pd.DataFrame(rows, columns=AREA_COLUMNS)


def cached_areas(image_path):
    """Cached areas as a dict, or None if never computed."""
    path = areas_path(image_path)

    if not path.exists():
        return None

    table = pd.read_csv(path)

    return {
        row["region"]: {
            "area_mm2": None if pd.isna(row["area_mm2"]) else float(row["area_mm2"]),
            "tissue_px": int(row["tissue_px"]),
        }
        for _, row in table.iterrows()
    }


def refresh_areas(sections, force=False):
    """Compute and cache areas for sections that have outlines. Opens TIFs."""
    done = 0

    for section in sections:
        if not (section["has_regions"] and section["has_dorsal"]):
            continue

        path = areas_path(section["path"])

        if path.exists() and not force:
            continue

        print(f"  {section['stem'][:44]:46s} opening TIF...", end="", flush=True)

        try:
            table = compute_areas(section["path"])
        except Exception as error:
            # One unreadable section must not abort the whole sweep: the other sixteen
            # are still worth caching, and the failure is reported rather than swallowed.
            print(f" FAILED: {type(error).__name__}: {error}")
            continue

        table.to_csv(path, index=False)
        done += 1

        shown = "  ".join(
            f"{r.region} {r.area_mm2:.3f}" if pd.notna(r.area_mm2) else f"{r.region} n/a"
            for r in table.itertuples()
        )
        print(f" {shown}")

    print(f"\ncached areas for {done} section(s)")


def is_stale(image_path):
    """Were the counts made before the outlines were last changed?

    Compares modification times of files that are already the source of truth, so no
    separate record has to be kept in step with reality. Alice redrew the CMM outlines
    after four sections had been counted, and those densities are wrong now -- this is what
    stops the table presenting them as current.
    """
    paths = sidecar_paths(image_path)
    counts, regions = paths["counts"], paths["regions"]

    if not (counts.exists() and regions.exists()):
        return False

    return regions.stat().st_mtime > counts.stat().st_mtime


# The columns build_table produces, in order. Named so that a folder with no sections yet -- a
# fresh install before "Choose folder", or a folder with nothing counted -- still yields a table
# with the right shape instead of one with no columns at all. Every reader below and in app.py
# indexes these by name (table["density_per_mm2"], table["stale"], table["stem"]), and on a
# column-less DataFrame each of those is a KeyError, which is a crash on the very first launch.
TABLE_COLUMNS = [
    "stem",
    "bird",
    "region",
    "status",
    "cells",
    "area_mm2",
    "density_per_mm2",
    "stale",
]


def build_table(sections=None):
    """One row per section and region: count, area, density, and whether it is stale."""
    sections = sections if sections is not None else find_sections()
    rows = []

    for section in sections:
        counts = counts_for(section["path"])
        areas = cached_areas(section["path"])
        stale = is_stale(section["path"])

        for name in ps6.REGION_NAMES:
            area = areas.get(name, {}).get("area_mm2") if areas else None
            count = counts.get(name) if counts else None

            density = (
                round(count / area, 1)
                if count is not None and area not in (None, 0)
                else None
            )

            rows.append(
                {
                    "stem": section["stem"],
                    "bird": section["stem"].split("_")[0],
                    "region": name,
                    "status": section["status"],
                    "cells": count,
                    "area_mm2": area,
                    "density_per_mm2": density,
                    "stale": stale,
                }
            )

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def summarise_by_region(table):
    """Mean density per region over sections that have one and are not stale.

    Stale rows are excluded rather than flagged here: this is a number someone would quote,
    and a mean that silently mixes current and outdated areas is worse than no mean. Counts
    are never summed across sections -- see the module docstring.
    """
    usable = table[table["density_per_mm2"].notna() & ~table["stale"]]

    if not len(usable):
        return None

    return usable.groupby("region")["density_per_mm2"].agg(["count", "mean", "std"])


def main():
    sections = find_sections()

    if "--areas" in sys.argv:
        print("computing region areas (this opens the TIFs)\n")
        refresh_areas(sections, force="--force" in sys.argv)
        print()

    table = build_table(sections)

    print(f"{'section':40s} {'region':6s} {'cells':>6s} {'area mm2':>9s} "
          f"{'per mm2':>9s}  note")
    print("-" * 82)

    for stem, group in table.groupby("stem", sort=True):
        for index, (_, row) in enumerate(group.iterrows()):
            # pd.isna, not `is None`: a column holding numbers stores a missing value as
            # NaN, so `is None` silently never matches and every row reads as present.
            note = ""

            if row["stale"]:
                note = "STALE: outlines redrawn after counting"
            elif pd.isna(row["cells"]):
                note = row["status"]
            elif pd.isna(row["area_mm2"]):
                note = "no cached area -- run --areas"

            cells = "-" if pd.isna(row["cells"]) else f"{int(row['cells']):d}"
            area = "-" if pd.isna(row["area_mm2"]) else f"{row['area_mm2']:.3f}"
            density = (
                "-"
                if pd.isna(row["density_per_mm2"])
                else f"{row['density_per_mm2']:.1f}"
            )

            print(
                f"{(stem[:40] if index == 0 else ''):40s} {row['region']:6s} "
                f"{cells:>6s} {area:>9s} {density:>9s}  {note}"
            )

    summary = summarise_by_region(table)

    print()

    if summary is None:
        print(
            "No usable densities yet. Every counted section was counted before its\n"
            "outlines were last redrawn, so all four are stale -- recount them with\n"
            "count_cells.py and the densities become current."
        )
    else:
        print("=== mean density by region (current, non-stale sections only) ===")
        print(summary.round(1).to_string())

    missing = sum(
        1
        for s in sections
        if s["has_regions"] and s["has_dorsal"] and not areas_path(s["path"]).exists()
    )

    if missing:
        print(f"\n{missing} section(s) have outlines but no cached area: run --areas")


if __name__ == "__main__":
    main()
