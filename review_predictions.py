"""Draw what the region model predicts against what Alice drew, on one sheet.

Alice, 2026-08-10: "can you show me what the region model outlines so i can see". The
cross-bird numbers say it over-predicts (median area error 50% on NCM, 109% on CMM) but a
number cannot say WHERE the boundary went wrong -- whether it is a slightly loose outline
everywhere, or the right shape in the wrong place, or region spilling into tissue that is
obviously not NCM. Those have different fixes, and only a picture distinguishes them.

WHAT THIS SHEET IS NOT. The saved model (ps6_region_unet.pt) is the FINAL model, trained
on all 14 sections, so every prediction here is on a section it has already seen. These
are best-case predictions and they flatter the model. The honest numbers are the
leave-one-bird-out ones in `grids/region run.log`; the fold models themselves were not
kept. This sheet is for seeing the FAILURE MODE, not for measuring: if it over-predicts
even on sections it trained on, that is worse news than the summary, not better.

Drawn outlines are solid, predictions are dotted, and each panel is captioned with the
area error so the picture and the number sit together.

Also measures how much predicted region falls OUTSIDE the tissue mask, because that is a
constraint the model is never told about and a cheap thing to fix if it is being violated.

Usage:
  python review_predictions.py
  python review_predictions.py --scale 0.10     # bigger panels
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import ps6
from train_regions import CLASS_NAMES, UNet

DATA_PATH = "ps6_region_maps.npz"
MODEL_PATH = "ps6_region_unet.pt"
OUTPUT_PATH = "grids/region predictions.png"

COLUMNS = 4
CAPTION_HEIGHT = 34
HEADER_HEIGHT = 34

# Panels are drawn from the 480x480 canvas the model works on, upscaled for legibility.
# Nothing is re-read from the TIFs: the point is to see what the MODEL saw and what it
# did with it, and the cached canvas is exactly that.
PANEL_SCALE = 1.0

DRAWN_COLOURS = {"NCM": (0, 255, 255), "CMM": (255, 0, 255)}
PREDICTED_COLOURS = {"NCM": (0, 140, 255), "CMM": (255, 140, 0)}


def outline(mask):
    """The boundary pixels of a mask, as a boolean array.

    ps6.distance_to_edge gives distance to the nearest non-mask pixel, so a distance of
    exactly 1 is the outermost ring of the shape -- a one-pixel outline without needing
    an erosion helper, which this project does not have.
    """
    if not mask.any():
        return mask

    return mask & (ps6.distance_to_edge(mask) <= 1)


def dotted(mask, period=3):
    """Every `period`-th pixel of an outline, so predictions read as dashed.

    Dashed rather than a second solid colour: the two outlines frequently sit a few
    pixels apart, and two solid lines at that distance are hard to tell apart. A dashed
    line stays legible where it overlaps the drawn one.
    """
    keeper = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    keep = ((ys + xs) % period) == 0
    keeper[ys[keep], xs[keep]] = True

    return keeper


def panel(green, tissue, drawn, predicted, scale):
    """One section: predictions filled, drawn outlines on top.

    Predictions are FILLED rather than outlined. The first version drew them as dotted
    outlines, which was unreadable for the wrong reason: the model emits hundreds of tiny
    fragments, and the outline of a fragment is the fragment, so everything became
    confetti whether it covered 5% of the tissue or 70%. Filling shows AREA, which is the
    quantity that matters -- region area is the denominator of every density.

    A general lesson about plotting: outlines communicate a boundary's position, fills
    communicate extent. Choosing the wrong one hides the failure you are looking for.
    """
    base = (np.clip(green, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)

    # Tissue edge in dim grey, so it is possible to see whether a prediction has spilled
    # off the tissue entirely -- which no anatomical region can do.
    rgb[outline(tissue)] = (70, 70, 70)

    for name in ("NCM", "CMM"):
        value = CLASS_NAMES.index(name)
        mask = predicted == value

        if mask.any():
            colour = np.array(PREDICTED_COLOURS[name])
            rgb[mask] = (0.55 * rgb[mask] + 0.45 * colour).astype(np.uint8)

    for name in ("NCM", "CMM"):
        rgb[outline(drawn == CLASS_NAMES.index(name))] = DRAWN_COLOURS[name]

    image = Image.fromarray(rgb)

    if scale != 1.0:
        size = (int(image.width * scale), int(image.height * scale))
        # NEAREST so a one-pixel outline stays a crisp line instead of being blurred into
        # the background by interpolation.
        image = image.resize(size, Image.NEAREST)

    return image


def area_mm2(mask, scale_um):
    return float(mask.sum()) * scale_um**2 / 1e6


def main():
    scale = next(
        (
            float(sys.argv[i + 1])
            for i, a in enumerate(sys.argv)
            if a == "--scale" and i + 1 < len(sys.argv)
        ),
        PANEL_SCALE,
    )

    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"{MODEL_PATH} not found; run train_regions.py first.")

    data = np.load(DATA_PATH)
    inputs = torch.from_numpy(data["inputs"])
    labels = data["labels"]
    stems = [str(s) for s in data["stems"]]
    scale_um = float(data["scale_um"])

    saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = UNet(in_channels=saved["in_channels"], width=saved["width"])
    model.load_state_dict(saved["state_dict"])
    model.eval()

    print(
        f"{MODEL_PATH}: trained on all {len(saved['sections'])} sections, "
        f"{saved['epochs']} epochs\n"
        f"NOTE these are predictions on sections the model TRAINED ON, so they are\n"
        f"     best-case. The honest numbers are in grids/region run.log.\n"
    )

    panels, captions, leaks = [], [], []

    with torch.no_grad():
        for index, stem in enumerate(stems):
            predicted = model(inputs[index : index + 1]).argmax(dim=1)[0].numpy()
            drawn = labels[index]
            tissue = inputs[index, 1].numpy() > 0.5

            errors = {}

            for name in ("NCM", "CMM"):
                value = CLASS_NAMES.index(name)
                p = area_mm2(predicted == value, scale_um)
                t = area_mm2(drawn == value, scale_um)
                errors[name] = p / t - 1 if t else float("nan")

            # How much predicted region sits off the tissue altogether. The model is never
            # told regions must lie within tissue, and if it is breaking that rule then
            # simply masking the output is a free correction.
            region = predicted > 0
            outside = float((region & ~tissue).sum() / max(1, region.sum()))
            leaks.append(outside)

            panels.append(panel(inputs[index, 0].numpy(), tissue, drawn, predicted, scale))
            captions.append(
                (
                    stem[:30],
                    f"NCM {errors['NCM']:+.0%}  CMM {errors['CMM']:+.0%}  "
                    f"off-tissue {outside:.0%}",
                )
            )

            print(
                f"  {stem[:34]:36s} NCM {errors['NCM']:+7.0%}  CMM {errors['CMM']:+7.0%}"
                f"  off-tissue {outside:5.1%}"
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
        "FILLED = model prediction (NCM blue, CMM orange).  grey = tissue edge",
        fill="white",
    )
    draw.text(
        (4, 18),
        "Predictions are on sections the model TRAINED ON -- best case, not an "
        "honest test.",
        fill=(255, 180, 180),
    )

    for index, (image, (stem, numbers)) in enumerate(zip(panels, captions)):
        row, column = divmod(index, COLUMNS)
        x, y = column * cell_width, HEADER_HEIGHT + row * cell_height

        sheet.paste(image, (x, y))
        draw.text((x + 2, y + image.height + 2), stem, fill="white")
        draw.text((x + 2, y + image.height + 15), numbers, fill="yellow")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_PATH)

    print(f"\nwrote {OUTPUT_PATH}")
    print(
        f"\npredicted region falling outside tissue: "
        f"median {np.median(leaks):.1%}, worst {np.max(leaks):.1%}"
    )

    if np.median(leaks) > 0.02:
        print(
            "  That is a constraint worth enforcing: no anatomical region lies off the\n"
            "  tissue, and masking the prediction by tissue is free."
        )
    else:
        print("  The model already keeps regions on the tissue; nothing to gain there.")


if __name__ == "__main__":
    main()
