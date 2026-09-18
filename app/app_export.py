"""Turn the counts table into something a spreadsheet will accept, and nothing more.

Two ways out, both built on the same rows:

    tab-separated on the clipboard   paste straight into an open Google Sheet with cmd-V
    a .csv file next to the images   File > Import in Sheets, or open it in Excel

WHY NOT WRITE TO GOOGLE DIRECTLY. A button that really pushes rows into a Drive spreadsheet
needs a Google Cloud project, the Sheets API enabled, an OAuth client file kept on this
machine, a consent screen the first time, two more packages, and a token that expires -- all
of it to be redone on any other computer the app is installed on. Pasting is one keystroke.
If the API version is ever wanted, it can be added as a third button using these same rows.

TAB-SEPARATED FOR THE CLIPBOARD, NOT COMMA. Google Sheets splits pasted text on tabs into
cells automatically; pasted commas land in one cell and need Data > Split text to columns. The
section stems contain no tabs (they are filenames), so nothing can be mangled by this.

WHAT IS DELIBERATELY NOT IN THE EXPORT. No total row and no mean, because a total count across
sections is meaningless -- sections differ in size and level, so counts are only comparable as
densities. No dates or folder names in the rows either; the filename carries the date, and
provenance in every row is noise in a spreadsheet.

Usage:
  python app/app_export.py          # check the rules, then print the real table as TSV
"""

import datetime
import sys

import pandas as pd

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

import ps6

# The order a person reads: which bird, which section, which region, then the numbers. Renamed
# from the internal column names -- "stem" means nothing outside this codebase, and a
# spreadsheet header is read by people who have never seen the code.
EXPORT_COLUMNS = {
    "bird": "bird",
    "stem": "section",
    "region": "region",
    "cells": "cells",
    "area_mm2": "area mm2",
    "density_per_mm2": "cells per mm2",
    "stale": "stale",
}

# What goes in the stale column. Loud, because a stale row's density is simply wrong -- the
# outlines were redrawn after the count, so the cells were counted inside a different shape.
STALE_MARK = "STALE -- recount"

# The per-bird view. One row per bird per region, which is the level most questions are asked
# at -- a bird is the experimental unit, a section is a sample of it.
BIRD_COLUMNS = [
    "bird",
    "region",
    "sections",
    "mean cells per mm2",
    "sd",
    "mean area mm2",
]


def for_export(table):
    """The counted rows, in reading order, with spreadsheet column names.

    ONLY ROWS WITH A REAL COUNT. A section with no counts yet has nothing to say in a
    spreadsheet: blank cells in a sheet read as an error or as a zero, and neither is true. The
    app's own list is where you look to see what still needs counting.

    Stale rows ARE included, marked. Dropping them would quietly lose data someone spent forty
    minutes producing; exporting them unmarked would put wrong densities in a sheet that
    outlives this app. Marking them is the only honest option.
    """
    counted = table[table["cells"].notna()].copy()

    if not len(counted):
        return pd.DataFrame(columns=list(EXPORT_COLUMNS.values()))

    # Anatomical order, not alphabetical: dNCM, vNCM, CMM is how the regions are always
    # written and how they are read. Sorting by text would give CMM, dNCM, vNCM.
    counted["region"] = pd.Categorical(
        counted["region"], categories=ps6.REGION_NAMES, ordered=True
    )
    counted = counted.sort_values(["bird", "stem", "region"])

    # Counts are whole cells. They arrive as float64 because the column held NaN for uncounted
    # sections, and "13.0" in a spreadsheet cell is wrong in a way people notice.
    counted["cells"] = counted["cells"].astype(int)
    counted["stale"] = counted["stale"].map(lambda flag: STALE_MARK if flag else "")

    return counted[list(EXPORT_COLUMNS)].rename(columns=EXPORT_COLUMNS)


def by_bird(table):
    """One row per bird per region: how many sections, mean density, and the spread.

    MEANS OF DENSITIES, NEVER SUMS OF COUNTS. Sections differ in size and in how deep through
    the brain they were cut, so counts are not comparable between them and adding them up
    produces a number that means nothing -- it mostly measures how many sections happened to be
    cut and counted. Density is the comparable quantity, and averaging it over a bird's sections
    is the whole reason this view exists.

    STALE SECTIONS ARE EXCLUDED. Their outlines were redrawn after counting, so their densities
    are wrong; an average that quietly mixes them in is worse than no average, because it looks
    finished. They stay visible in the per-section view, marked, and the count of what was left
    out is reported by describe_by_bird -- excluded, but not hidden.

    A bird with one usable section gets a blank sd rather than 0. Zero would claim the
    measurement was perfectly repeatable when it has not been repeated at all.
    """
    usable = table[table["density_per_mm2"].notna() & ~table["stale"]].copy()

    if not len(usable):
        return pd.DataFrame(columns=BIRD_COLUMNS)

    usable["region"] = pd.Categorical(
        usable["region"], categories=ps6.REGION_NAMES, ordered=True
    )

    grouped = usable.groupby(["bird", "region"], observed=True).agg(
        sections=("density_per_mm2", "count"),
        mean=("density_per_mm2", "mean"),
        sd=("density_per_mm2", "std"),
        area=("area_mm2", "mean"),
    )

    frame = grouped.reset_index().sort_values(["bird", "region"])
    frame["mean"] = frame["mean"].round(1)
    frame["sd"] = frame["sd"].round(1)
    frame["area"] = frame["area"].round(4)

    frame.columns = BIRD_COLUMNS

    return frame


def describe_by_bird(table, frame):
    """What the per-bird view is built from, as lines for the log. Worst news first."""
    lines = []

    dropped = int((table["cells"].notna() & table["stale"]).sum())

    if dropped:
        lines.append(
            f"  {dropped} stale row(s) are LEFT OUT of these averages -- their outlines were "
            "redrawn after counting. Recount those sections to include them."
        )

    if not len(frame):
        lines.append(
            "No averages yet: no section has a count made after its outlines were last drawn."
        )

        return lines

    sections = int(frame["sections"].max())

    lines.append(
        f"{len(frame)} rows: {frame['bird'].nunique()} birds, "
        f"up to {sections} section(s) averaged per bird and region."
    )

    return lines


def as_text(frame, separator):
    """The frame as text, with blanks for missing numbers rather than the word 'nan'."""
    return frame.to_csv(index=False, sep=separator, na_rep="")


def as_tsv(frame):
    """For the clipboard: Sheets turns tabs into cells."""
    return as_text(frame, "\t")


def as_csv(frame):
    """For a file: what every spreadsheet program opens without being asked twice."""
    return as_text(frame, ",")


def file_name(today=None):
    """'cell counts 2026-08-13.csv'.

    Dated rather than one file overwritten each time, so an export is a record instead of a
    moving target -- someone quoting numbers from a sheet can find the export they came from.
    The date is a parameter so this can be checked without waiting for tomorrow.
    """
    today = today or datetime.date.today()

    return f"cell counts {today.isoformat()}.csv"


def describe(frame):
    """What the export contains, as lines for the log. Worst news first."""
    if not len(frame):
        return ["Nothing to export yet -- no section in this folder has been counted."]

    stale = int((frame["stale"] != "").sum())
    lines = []

    if stale:
        lines.append(
            f"  WARNING: {stale} of {len(frame)} rows are STALE -- their outlines were redrawn "
            "after counting, so those densities are wrong. They are marked in the 'stale' "
            "column; recount those sections and export again."
        )

    lines.append(
        f"{len(frame)} rows: {frame['section'].nunique()} sections, "
        f"{frame['bird'].nunique()} birds."
    )

    return lines


def fake_table():
    """A counts table like app_table.build_table's, without touching any disk."""
    rows = []

    for stem, cells, stale in [
        ("Zebra9_LH_1-1-1", [13, 18, 25], False),
        # Same bird, second section: two usable sections is what gives a real sd.
        ("Zebra9_LH_1-1-2", [17, 14, 31], False),
        ("Amber2_RH_2-1-4", [None, None, None], False),
        ("Amber2_LH_2-1-5", [7, 9, 11], True),
        # One usable section, so its sd must come out blank rather than zero.
        ("Wht17_RH_1-1-3", [10, 20, 30], False),
    ]:
        for name, count in zip(ps6.REGION_NAMES, cells):
            rows.append(
                {
                    "stem": stem,
                    "bird": stem.split("_")[0],
                    "region": name,
                    "status": "counted" if count is not None else "ready to count",
                    "cells": count,
                    "area_mm2": 0.5 if count is not None else None,
                    "density_per_mm2": count / 0.5 if count is not None else None,
                    "stale": stale,
                }
            )

    return pd.DataFrame(rows)


def show(title, frame):
    """Print a frame the way it would paste, with tabs drawn as bars."""
    print(f"\n{title}:\n")

    for line in as_tsv(frame).splitlines():
        print("  " + line.replace("\t", " | "))


def main():
    table = fake_table()
    frame = for_export(table)

    assert list(frame.columns) == list(EXPORT_COLUMNS.values()), list(frame.columns)
    assert len(frame) == 12, "one row per region for each of the four counted sections"
    assert "Amber2_RH_2-1-4" not in set(frame["section"]), "uncounted section leaked in"

    # Anatomical order within a section, and birds together.
    first = frame[frame["section"] == "Amber2_LH_2-1-5"]
    assert list(first["region"]) == ps6.REGION_NAMES, list(first["region"])
    assert list(frame["bird"])[:3] == ["Amber2"] * 3, list(frame["bird"])

    assert list(frame["cells"]) == [7, 9, 11, 10, 20, 30, 13, 18, 25, 17, 14, 31]
    assert "13.0" not in as_csv(frame), "counts must export as whole numbers"

    assert int((frame["stale"] != "").sum()) == 3, "the stale section must be marked"
    assert "\t" in as_tsv(frame) and "," not in as_tsv(frame).splitlines()[0]

    empty = for_export(table.assign(cells=None))
    assert not len(empty) and list(empty.columns) == list(EXPORT_COLUMNS.values())
    assert "Nothing to export" in describe(empty)[0]

    assert file_name(datetime.date(2026, 8, 13)) == "cell counts 2026-08-13.csv"

    # ---- the per-bird view ----
    birds = by_bird(table)

    assert list(birds.columns) == BIRD_COLUMNS, list(birds.columns)
    assert set(birds["bird"]) == {"Wht17", "Zebra9"}, "the stale-only bird must be excluded"
    assert len(birds) == 6, "two birds x three regions"
    assert list(birds["region"])[:3] == ps6.REGION_NAMES, list(birds["region"])

    zebra = birds[(birds["bird"] == "Zebra9") & (birds["region"] == "dNCM")].iloc[0]

    # Two sections at 26.0 and 34.0 per mm2: the mean, and a real spread.
    assert zebra["sections"] == 2, zebra["sections"]
    assert zebra["mean cells per mm2"] == 30.0, zebra["mean cells per mm2"]
    assert zebra["sd"] == 5.7, zebra["sd"]

    lonely = birds[birds["bird"] == "Wht17"].iloc[0]

    assert lonely["sections"] == 1
    assert pd.isna(lonely["sd"]), "one section must give a blank sd, not zero"
    # And it must reach the spreadsheet as an EMPTY cell -- two tabs in a row -- rather than
    # as the word "nan", which sorts and charts as if it were data.
    assert "\t\t" in as_tsv(birds), as_tsv(birds)
    assert "nan" not in as_tsv(birds).lower()

    notes = describe_by_bird(table, birds)
    assert "3 stale row(s) are LEFT OUT" in notes[0], notes

    none_usable = by_bird(table.assign(stale=True))
    assert not len(none_usable) and list(none_usable.columns) == BIRD_COLUMNS
    assert "No averages yet" in describe_by_bird(table.assign(stale=True), none_usable)[-1]

    print("export rules: all cases correct\n")

    for line in describe(frame):
        print(line)

    for line in describe_by_bird(table, birds):
        print(line)

    show("invented data, every section", frame)
    show("invented data, averaged per bird", birds)

    if "--real" in sys.argv:
        from app_state import find_sections
        from app_table import build_table

        real = build_table(find_sections())

        show(f"this folder, every section ({len(for_export(real))} rows)", for_export(real))
        show(f"this folder, per bird ({len(by_bird(real))} rows)", by_bird(real))


if __name__ == "__main__":
    main()
