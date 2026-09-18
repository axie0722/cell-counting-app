"""One sheet showing every section's drawn NCM and CMM, to spot inconsistent outlines.

The regions were drawn one section at a time, months apart, and nothing has ever shown
them side by side. That is how the CMM outlines came to vary 5x in area relative to NCM
(0.13 to 0.70 across 14 sections, measured 2026-08-09) -- an outline looks reasonable in
isolation and only reads as too small next to thirteen others.

This matters more than it sounds. Region area is the denominator of every density the
lab reports, so an undersized CMM inflates its density directly. And a segmentation
model trained on these outlines can only be as consistent as they are: it would learn
the spread and predict an average CMM for every bird.

Each thumbnail is the section at a few percent scale with NCM in cyan and CMM in
magenta, captioned with the CMM/NCM area ratio and the section name. The ratio is
sorted, so the outliers land at the ends rather than being hunted for.

Draws with PIL for the same reason false_positive_gallery.py does -- matplotlib is
deliberately not a dependency of this project.

Usage:
  python review/review_regions.py
  python review/review_regions.py --scale 0.06     # bigger thumbnails, fewer per row

Then redraw whichever ones look wrong; draw_regions.py reloads what is already there:
  python regions/draw_regions.py "Wh175_LH_1-1-7_NCM_Slide 3.TIF"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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

OUTPUT_PATH = "grids/region review.png"

# Small enough that 14 sections fit on one screen, large enough to judge whether an
# outline follows the anatomy. These are 55-100 Mpx images; 4% is ~300 px across.
SCALE = 0.04
COLUMNS = 5
CAPTION_HEIGHT = 26
OUTLINE_WIDTH = 2


def polygon_area(vertices):
    """Shoelace area. Sign-free, since vertex order is whatever the drawing produced."""
    y, x = vertices[:, 0], vertices[:, 1]

    return 0.5 * abs(np.dot(y, np.roll(x, 1)) - np.dot(x, np.roll(y, 1)))


def find_image(stem):
    for candidate in (Path(f"{stem}.TIF"), Path("birds") / f"{stem}.TIF"):
        if candidate.exists():
            return str(candidate)

    return None


def region_csvs():
    """Every regions CSV, in BOTH folders, one per section stem.

    `birds/` HAS TO BE SEARCHED. It did not when this was written -- every CSV sat in the top folder then --
    but 11 of the 15 outlines have since been moved into `birds/`, so a top-level-only glob now finds 4. That
    would not raise: `prepare_regions` would happily rebuild `ps6_region_maps.npz` from 4 sections, and every
    held-out number in the project would silently change meaning while still printing a mean. See
    [[archive-copies-win-name-keyed-surveys]] for the same class of bug found in a name-keyed survey.

    A stem in both folders is reported rather than resolved quietly, and the copy next to its own TIF wins --
    guessing here writes one section's polygons under another section's name.
    """
    by_stem = {}

    for folder in (Path("."), Path("birds")):
        for path in sorted(folder.glob("*regions.csv")):
            stem = path.name[: -len(" regions.csv")]
            seen = by_stem.get(stem)

            if seen is None:
                by_stem[stem] = path
                continue

            # Prefer whichever copy sits beside the image `find_image` will resolve.
            image = find_image(stem)
            keep = path if image is not None and Path(image).parent == path.parent else seen
            print(f"  {stem} has a regions.csv in both folders; using {keep}")
            by_stem[stem] = keep

    return [by_stem[stem] for stem in sorted(by_stem)]


def sections():
    """Every section with regions drawn, and the polygons, ordered by CMM/NCM area."""
    found = []

    for path in region_csvs():
        stem = path.name[: -len(" regions.csv")]
        image_path = find_image(stem)

        if image_path is None:
            print(f"  no TIF for {stem}; skipping")
            continue

        table = pd.read_csv(path)

        # HAND-DRAWN OUTLINES ONLY. `draw_regions.py` now opens with the rule's own prediction already in the
        # layer, and writes `source = model` for any polygon still exactly as proposed. This is the corpus
        # every fit and every grade is built from, so letting a proposal in would train the next model on its
        # own output -- worth about +0.2 of fake score the last time that happened
        # ([[model-proposed-labels-are-circular]]). Older files have no `source` column and are all hand-drawn.
        if "source" in table.columns:
            dropped = sorted(set(table.loc[table["source"] == "model", "region"].astype(str)))
            table = table[table["source"] != "model"]

            if dropped:
                print(f"  {stem}: {', '.join(dropped)} is the model's own proposal, not a label; skipping it")

        polygons = {
            str(name): group[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)
            for name, group in table.groupby("region", sort=False)
        }

        areas = {name: polygon_area(v) for name, v in polygons.items() if len(v) >= 3}
        ratio = (
            areas["CMM"] / areas["NCM"]
            if areas.get("NCM") and areas.get("CMM")
            else float("nan")
        )

        found.append(
            {
                "stem": stem,
                "image": image_path,
                "polygons": polygons,
                "ratio": ratio,
                "areas": areas,
            }
        )

    # Sorted by ratio so the extremes sit at the ends of the sheet. NaN last.
    return sorted(found, key=lambda s: (np.isnan(s["ratio"]), s["ratio"]))


def thumbnail(section, scale):
    """The section at `scale`, with its polygons drawn on."""
    _, green = ps6.load_green(section["image"])

    # Percentile contrast, for the reason in false_positive_gallery.stretch: these are
    # float images whose tissue sits well below the bright tail, and min-max scaling
    # renders the whole section nearly black.
    low, high = np.percentile(green, [1, 99.5])
    scaled = (np.clip(green, low, high) - low) / max(1e-6, high - low)

    height, width = green.shape
    del green

    image = Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
    del scaled

    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(size, Image.BILINEAR)

    draw = ImageDraw.Draw(image)

    for name, colour in (("NCM", "cyan"), ("CMM", "magenta")):
        vertices = section["polygons"].get(name)

        if vertices is None or len(vertices) < 3:
            continue

        points = [(v[1] * scale, v[0] * scale) for v in vertices]
        draw.polygon(points, outline=colour, width=OUTLINE_WIDTH)

    return image


def main():
    scale = next(
        (
            float(sys.argv[i + 1])
            for i, a in enumerate(sys.argv)
            if a == "--scale" and i + 1 < len(sys.argv)
        ),
        SCALE,
    )

    found = sections()

    if not found:
        raise SystemExit("No *regions.csv files found.")

    print(f"{len(found)} sections with regions drawn, at scale {scale}\n")

    thumbnails = []

    for section in found:
        print(f"  {section['stem'][:44]:46s} CMM/NCM {section['ratio']:.2f}", flush=True)
        thumbnails.append(thumbnail(section, scale))

    cell_width = max(t.width for t in thumbnails)
    cell_height = max(t.height for t in thumbnails) + CAPTION_HEIGHT
    rows = int(np.ceil(len(thumbnails) / COLUMNS))

    header = 20
    sheet = Image.new(
        "RGB", (COLUMNS * cell_width, header + rows * cell_height), "black"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 5),
        "Drawn regions -- NCM cyan, CMM magenta. Sorted by CMM/NCM area: "
        "smallest CMM first.",
        fill="white",
    )

    for index, (section, image) in enumerate(zip(found, thumbnails)):
        row, column = divmod(index, COLUMNS)
        x = column * cell_width
        y = header + row * cell_height

        sheet.paste(image, (x, y))

        draw.text(
            (x + 2, y + image.height + 2),
            f"CMM/NCM {section['ratio']:.2f}",
            fill="yellow",
        )
        draw.text(
            (x + 2, y + image.height + 13),
            section["stem"][:38],
            fill="white",
        )

    sheet.save(OUTPUT_PATH)
    print(f"\nwrote {OUTPUT_PATH}")

    ratios = np.array([s["ratio"] for s in found])
    usable = ratios[~np.isnan(ratios)]

    print(
        f"\nCMM/NCM area ratio across {len(usable)} sections: "
        f"median {np.median(usable):.2f}, range {usable.min():.2f}-{usable.max():.2f}\n"
    )

    # Flagged relative to the median rather than against a fixed number: what counts as
    # a small CMM is not knowable in the abstract, only against how these were usually
    # drawn. This flags candidates for review; it does not decide anything.
    median = np.median(usable)

    for section in found:
        if np.isnan(section["ratio"]):
            continue

        if section["ratio"] < 0.6 * median:
            print(
                f"  SMALL  {section['ratio']:.2f}  {section['stem'][:44]}\n"
                f'         python regions/draw_regions.py "{section["image"]}"'
            )

    print(
        "\nThe ratio is a prompt to look, not a verdict -- a genuinely small CMM is\n"
        "possible. Judge from the outline against the anatomy in the thumbnail."
    )


if __name__ == "__main__":
    main()
