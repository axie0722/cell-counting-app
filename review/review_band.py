"""Draw BandNet's PREDICTED Field L borders against Alice's outlines.

The band equivalent of review_boxes.py. Two failures look identical in an accuracy number
and completely different on a sheet:

  * the borders are roughly right but the regions run too far along the tissue
    -- expected, since a band has no end cuts yet, and it is an AREA problem
  * a border landed on the wrong side entirely
    -- CMM came out 0% on Wh175_LH_1-1-7 and YW113_RH_1-1-2, which is a sign error, not a
       near miss

CROSS-BIRD BY DEFAULT, and cached. train_band.py saves only the final all-data model, so the
honest sheet has to refit the 7 folds -- about 90 minutes. The outputs are written to
CACHE_PATH so that cost is paid once. A sheet that quietly fell back to the final model would
be presenting memorised borders as held-out ones, which is the mistake review_predictions.py
has to warn about in its own header; here the honest version is the default and --final is
opt-in and labelled on the image.

WHAT TO LOOK FOR. The predicted band is drawn as two lines with the Field L strip between them
shaded, so a crossed pair is immediately visible as an inverted strip. Alice's own outlines are
drawn solid on top. Since the regions have no end cuts, expect the fills to overrun her
outlines along the Field L axis -- that is the known area problem, not the thing this sheet is
for. Judge the LINES, not the extent.

Usage:
  python review/review_band.py             # cross-bird, honest; refits 7 folds unless cached
  python review/review_band.py --final     # the all-data model, on sections it trained on
  python review/review_band.py --recache   # refit and overwrite the cache
"""

import sys
from pathlib import Path

import numpy as np
import torch
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

from ceiling_lines import DATA_PATH
from field_l_band import signed_distance, split_tissue
from review_predictions import COLUMNS, DRAWN_COLOURS, PREDICTED_COLOURS, outline
from train_band import MODEL_PATH, OUTPUTS, bands_from_outputs, train
from train_boxes import BoxNet

# 8 um per canvas pixel, the same scale train_band.py's TEMPERATURE is expressed in. Only
# used to print band widths in microns, so they can be compared against the measured
# 400-937 um range rather than read as bare pixels.
UM_PER_PIXEL = 8.0

OUTPUT_PATH = "grids/band predictions.png"
CACHE_PATH = "grids/band crossbird outputs.npy"

CAPTION_HEIGHT = 34
HEADER_HEIGHT = 48

# The Field L strip between the two predicted borders. Shaded rather than outlined so a
# CROSSED pair -- ncm_offset below cmm_offset -- shows up as a strip in the wrong place
# instead of silently rendering as nothing.
BAND_COLOUR = (250, 250, 120)


def band_strip(band, tissue):
    """The tissue between the two predicted borders: predicted Field L.

    Uses each border's own normal, because the two are not parallel -- field_l_band.py
    measured that forcing them parallel costs 5 points of NCM cells.
    """
    ncm_signed = signed_distance(
        np.array([band["normal_y"], band["normal_x"]]), tissue.shape
    )
    cmm_signed = signed_distance(
        np.array([band["cmm_normal_y"], band["cmm_normal_x"]]), tissue.shape
    )

    return tissue & (ncm_signed <= band["ncm_offset"]) & (cmm_signed >= band["cmm_offset"])


def panel(green, tissue, drawn, predicted, strip):
    """Predicted regions filled, predicted Field L shaded, Alice's outlines solid on top.

    Same convention as review_boxes.py: fills show what the model claims, outlines show what
    Alice drew, so disagreement reads as fill spilling past an outline.
    """
    base = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    rgb[outline(tissue)] = (70, 70, 70)

    for name in ("NCM", "CMM"):
        if predicted[name].any():
            colour = np.array(PREDICTED_COLOURS[name])
            rgb[predicted[name]] = (
                0.55 * rgb[predicted[name]] + 0.45 * colour
            ).astype(np.uint8)

    if strip.any():
        rgb[strip] = (0.5 * rgb[strip] + 0.5 * np.array(BAND_COLOUR)).astype(np.uint8)

    for name in ("NCM", "CMM"):
        rgb[outline(drawn[name])] = DRAWN_COLOURS[name]

    return Image.fromarray(rgb)


def cross_bird_outputs(data, device, recache=False):
    """Predict every section from a model that never saw its bird. Cached to CACHE_PATH.

    The cache is keyed on nothing but the file existing, which is deliberate but worth
    knowing: change train_band.py and the cache is stale. --recache is the way out, and the
    shape check below catches the one stale case that would otherwise crash confusingly.
    """
    if not recache and Path(CACHE_PATH).exists():
        cached = np.load(CACHE_PATH)

        if cached.shape == (len(data["stems"]), OUTPUTS):
            print(f"using cached cross-bird outputs from {CACHE_PATH}")

            return cached

        print(f"{CACHE_PATH} has the wrong shape {cached.shape}; refitting")

    stems = [str(s) for s in data["stems"]]
    birds = np.array([str(b) for b in data["birds"]])

    inputs = torch.from_numpy(data["inputs"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).long().to(device)

    outputs = np.zeros((len(stems), OUTPUTS), dtype=np.float64)

    for held_out in sorted(set(birds)):
        test = birds == held_out
        keep = np.nonzero(~test)[0]

        print(f"  refitting without {held_out}...", flush=True)

        model = train(inputs[keep], labels[keep], device, quiet=True)
        model.eval()

        with torch.no_grad():
            indices = np.nonzero(test)[0]
            outputs[indices] = model(inputs[indices]).cpu().numpy()

    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_PATH, outputs)
    print(f"cached to {CACHE_PATH}")

    return outputs


def final_outputs(data, device):
    saved = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = BoxNet(
        in_channels=saved["in_channels"], outputs=saved.get("outputs", OUTPUTS)
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    inputs = torch.from_numpy(data["inputs"]).float().to(device)

    with torch.no_grad():
        return model(inputs).cpu().numpy()


def main():
    final = "--final" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]

    if final:
        if not Path(MODEL_PATH).exists():
            raise SystemExit(f"{MODEL_PATH} not found; run train_band.py first.")

        print("--final: the all-data model, on sections it TRAINED ON. Best case.\n")
        outputs = final_outputs(data, device)
    else:
        print("cross-bird: each section drawn by the fold that never saw its bird.\n")
        outputs = cross_bird_outputs(data, device, recache="--recache" in sys.argv)

    panels, captions = [], []
    crossed = []
    scores = {"NCM": [], "CMM": []}

    for index, stem in enumerate(stems):
        tissue = data["inputs"][index, 1] > 0.5
        green = data["inputs"][index, 0]
        labels = data["labels"][index]

        drawn = {"NCM": labels == 1, "CMM": labels == 2}
        band = bands_from_outputs(outputs[index])

        ncm_side, cmm_side = split_tissue(band, tissue)
        predicted = {"NCM": ncm_side, "CMM": cmm_side}

        width = band["ncm_offset"] - band["cmm_offset"]

        # A negative width means the borders crossed, so CMM's side has been placed past
        # NCM's. Recorded rather than hidden: it is the failure mode two sections showed.
        if width < 0:
            crossed.append(stem)

        parts = []

        for name in ("NCM", "CMM"):
            total = max(int(drawn[name].sum()), 1)
            # "right" = fraction of the pixels Alice drew for this region that the model put
            # on this region's side. The same figure train_band.evaluate reports, so the
            # sheet and the training log cannot disagree.
            right = (predicted[name] & drawn[name]).sum() / total
            area = predicted[name].sum() / total

            scores[name].append((right, area))
            parts.append(f"{name} {right:.0%} right x{area:.1f} area")

        panels.append(panel(green, tissue, drawn, predicted, band_strip(band, tissue)))
        captions.append(
            (stem[:30], f"{'   '.join(parts)}   band {width * UM_PER_PIXEL:+.0f}um")
        )

        print(
            f"  {stem[:32]:34s} {'   '.join(parts)}"
            f"   band {width * UM_PER_PIXEL:+5.0f}um"
            + ("   <-- CROSSED" if width < 0 else "")
        )

    cell_width = max(p.width for p in panels)
    cell_height = max(p.height for p in panels) + CAPTION_HEIGHT
    rows = int(np.ceil(len(panels) / COLUMNS))

    sheet = Image.new(
        "RGB", (COLUMNS * cell_width, HEADER_HEIGHT + rows * cell_height), "black"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, 4),
        "SOLID LINES = Alice's outlines (NCM cyan, CMM magenta).  "
        "FILLED = BandNet prediction (NCM blue, CMM orange).  "
        "PALE YELLOW = predicted Field L, which belongs to neither.",
        fill="white",
    )
    draw.text(
        (4, 18),
        "Each region = a half-plane (a directed normal + an offset) intersected with the "
        "tissue. NO END CUTS YET, so overrun along the Field L axis is expected -- "
        "judge the LINES, not the extent.",
        fill=(180, 220, 255),
    )
    draw.text(
        (4, 32),
        "--final: TRAINED ON these sections, best case, NOT an accuracy measurement"
        if final
        else "cross-bird: every section predicted by a model that never saw its bird",
        fill=(255, 180, 180) if final else (180, 255, 180),
    )

    for index, (image, (stem, numbers)) in enumerate(zip(panels, captions)):
        row, column = divmod(index, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height

        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), stem, fill="white")
        draw.text((x + 2, y + image.height + 15), numbers, fill="yellow")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_PATH)

    print()
    for name in ("NCM", "CMM"):
        right = np.array([r for r, _ in scores[name]])
        area = np.array([a for _, a in scores[name]])
        print(
            f"{name}: median {np.median(right):.0%} of drawn pixels on the right side, "
            f"median area x{np.median(area):.2f}, "
            f"{int((right < 0.5).sum())}/{len(right)} sections below half"
        )

    print(f"\nwrote {OUTPUT_PATH}")

    if crossed:
        print(
            f"\n{len(crossed)} section(s) have CROSSED borders -- CMM's line placed past\n"
            "NCM's, so the predicted Field L strip is inverted and a region comes out\n"
            "empty. This is a sign error, not a near miss:\n  " + "\n  ".join(crossed)
        )


if __name__ == "__main__":
    main()
