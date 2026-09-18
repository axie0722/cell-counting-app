"""Predict the Field L band: two borders, supervised on the SPLIT rather than the numbers.

Replaces train_boxes.py's approach. Three measured findings drive every design choice here.

1. THE BORDERS ARE WHAT MATTER. score_rungs_cells.py: the Field L split alone puts 99% of
   NCM cells and 100% of CMM cells on the correct side, with 0% side errors in all 14
   sections. Adding the box's three end cuts kept FEWER cells (96%/98%) -- they cost
   assignment accuracy and bought only area. So the borders are the whole game for counting,
   and area is a separate, later problem.

2. TWO BORDERS, NOT ONE. Alice, 2026-08-11: "ncm and cmm should not have a shared line".
   Field L is a stripe with width -- measured median 544 um, range 400-937, present in every
   section. NCM borders one edge, CMM the other, and the strip between belongs to neither.

3. EACH BORDER KEEPS ITS OWN ANGLE. field_l_band.py measured the alternative: forcing one
   shared normal drops NCM 99% -> 94% and CMM 100% -> 98%, and on OR99_LH (borders 3.7 apart)
   it throws a fifth of the cells to the wrong side. Alice: "the straight lines of ncm and cmm
   regions dont have to be exactly parallel". So 6 numbers, 4 degrees of freedom -- still well
   under the box model's 10.

WHY THE LOSS IS ON THE SPLIT. train_boxes.py regressed each of its 10 numbers against truth
independently, and the parameter probe showed why that fails: centre and angle errors
INTERACT. Substituting true angle alone bought +0.02 IoU, true centre alone +0.11, but both
together +0.26. A per-parameter loss cannot see a pairing -- it has no term that gets better
when two numbers become consistent with each other.

So each border is RENDERED as a soft mask, sigmoid((signed distance - offset) / TEMPERATURE)
rather than a hard >, and scored on the thing we actually measure: is it high over Alice's
pixels for that region and low over the other region's. The sigmoid is differentiable, so one
gradient reaches the normal and the offset together and a border that is correctly angled but
misplaced is penalised as one mistake. Hard comparison at inference, soft only for training --
the standard way to make a threshold learnable.

UNLABELLED TISSUE IS EXCLUDED FROM THE LOSS. Only pixels Alice assigned to NCM or CMM carry
truth about which side a border belongs on. Where she drew nothing there is no answer, and
supervising it would train the model on my guess about where her regions would have stopped.
This is also consistent with what she has said twice -- a cell outside the drawn regions is
still a cell, so absence of a label is not a negative.

A WEAK PARAMETER ANCHOR IS KEPT. The split loss alone is indifferent to where inside the
544 um Field L gap a border sits: any line in the stripe separates the labelled pixels
equally well, so the offsets could drift across the whole band and the reported area would
wander with them. A small L1 term on the parameters pins them to Alice's actual borders
without being big enough to reintroduce the interaction blindness.

Usage:
  python training/train_band.py                 # cross-bird by bird, then fit on all 14 and save
  python training/train_band.py --quick         # one fold, few epochs, nothing saved
  python training/train_band.py --no-augment    # ablation
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from field_l_band import BAND_FIELDS, band_from_labels
from train_boxes import BoxNet, CANVAS, TRUNK_DOWNSAMPLE
from train_regions import augment

MODEL_PATH = "ps6_region_band.pt"
PROGRESS_PATH = "grids/band folds.json"

SEED = 0
EPOCHS = 400
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4

# Softness of the rendered border, in canvas pixels (8 um each). Too sharp and the gradient
# vanishes everywhere except within a pixel of the border, so a badly misplaced line gets no
# signal at all and never moves. Too soft and the "border" is a wide grey smear whose
# position the loss barely cares about. 8 px = 64 um, comfortably narrower than the 400 um
# minimum Field L width, so the two borders stay distinguishable.
TEMPERATURE = 8.0

# How much the parameter anchor counts against the split loss. Small on purpose -- it exists
# only to stop the offsets drifting inside the Field L gap.
ANCHOR_WEIGHT = 0.1

# Predicted normals are free 2-vectors that get normalised, so their length is unconstrained
# and could collapse towards zero -- at which point the direction is numerically arbitrary.
# This keeps the raw vector near unit length.
LENGTH_WEIGHT = 0.01

OUTPUTS = 6


def coordinates(shape, device):
    """Per-pixel (y, x) offsets from the canvas centre, in canvas pixels.

    Built once per shape and reused: this is the grid every border is rendered against, and
    rebuilding it per sample per epoch is pure waste.
    """
    ys = torch.arange(shape[0], device=device, dtype=torch.float32) - shape[0] / 2.0
    xs = torch.arange(shape[1], device=device, dtype=torch.float32) - shape[1] / 2.0

    return ys.view(-1, 1), xs.view(1, -1)


def unpack(outputs):
    """Raw network outputs to (normals, offsets), both per region.

    Normals are normalised here rather than in the loss so that training and inference use
    the identical parameterisation -- a normal whose length drifted would otherwise rescale
    the offset and turn a small direction error into a large translation.
    """
    normals = {}
    offsets = {}

    for position, name in ((0, "NCM"), (3, "CMM")):
        vector = outputs[:, position : position + 2]
        length = vector.norm(dim=1, keepdim=True).clamp_min(1e-6)

        normals[name] = vector / length
        # Offsets are predicted scaled by the canvas, like every other length in this
        # project, so the network's outputs all live on a similar numeric range.
        offsets[name] = outputs[:, position + 2] * CANVAS

    return normals, offsets, outputs[:, [0, 1, 3, 4]]


def soft_sides(normals, offsets, shape, device):
    """Each region's soft mask: how strongly each pixel is on that region's side.

    NCM's normal points towards NCM, so NCM is where the signed distance EXCEEDS its offset.
    CMM's normal also points towards NCM (one sign convention throughout, fixed in
    field_l_band.py), so CMM is where the signed distance falls BELOW its offset -- hence the
    negated argument rather than a second convention to keep straight.
    """
    ys, xs = coordinates(shape, device)
    sides = {}

    for name in ("NCM", "CMM"):
        normal = normals[name]
        offset = offsets[name]

        # (batch, height, width) via broadcasting: normal_y * y + normal_x * x.
        signed = normal[:, 0].view(-1, 1, 1) * ys + normal[:, 1].view(-1, 1, 1) * xs
        margin = signed - offset.view(-1, 1, 1)

        sides[name] = torch.sigmoid((margin if name == "NCM" else -margin) / TEMPERATURE)

    return sides


def split_loss(outputs, labels):
    """How well the predicted borders assign Alice's labelled pixels. Lower is better.

    Each region's soft mask should be 1 over that region's pixels and 0 over the other's.
    Both are checked for both regions, which is what makes this a loss on the SPLIT: a
    border that keeps all of NCM by swallowing CMM as well scores badly, where a loss on
    NCM's pixels alone would call it perfect.

    Regions are weighted equally per section, not per pixel: NCM is usually the larger
    region, and per-pixel averaging would let it quietly outvote CMM.
    """
    normals, offsets, raw = unpack(outputs)
    sides = soft_sides(normals, offsets, labels.shape[-2:], outputs.device)

    masks = {"NCM": (labels == 1).float(), "CMM": (labels == 2).float()}

    total = 0.0

    for name, other in (("NCM", "CMM"), ("CMM", "NCM")):
        side = sides[name]

        # Mean over the region's own pixels only. clamp_min on the pixel count guards a
        # section whose region was rotated off the canvas by augmentation.
        own_area = masks[name].sum(dim=(1, 2)).clamp_min(1.0)
        other_area = masks[other].sum(dim=(1, 2)).clamp_min(1.0)

        inside = (side * masks[name]).sum(dim=(1, 2)) / own_area
        leaked = (side * masks[other]).sum(dim=(1, 2)) / other_area

        # Want inside -> 1 and leaked -> 0. Written as (1 - inside) + leaked so both terms
        # are costs on the same scale and neither can be traded off cheaply against the
        # other.
        total = total + ((1.0 - inside) + leaked).mean()

    # Unit-length nudge on the raw normals, so the direction stays numerically meaningful.
    length_penalty = ((raw[:, :2].norm(dim=1) - 1.0) ** 2).mean() + (
        (raw[:, 2:].norm(dim=1) - 1.0) ** 2
    ).mean()

    return total / 2.0 + LENGTH_WEIGHT * length_penalty


def targets_for(labels, tissue):
    """Alice's 6 numbers for one section, scaled. Order fixed by BAND_FIELDS x 2.

    Returns None when a region has been rotated off the canvas by augmentation, so the
    caller can skip the sample rather than fit a line to an empty mask.
    """
    band = band_from_labels(labels, labels.shape, tilt=True)

    if band is None:
        return None

    return [
        band["normal_y"],
        band["normal_x"],
        band["ncm_offset"] / CANVAS,
        band["cmm_normal_y"],
        band["cmm_normal_x"],
        band["cmm_offset"] / CANVAS,
    ]


def bands_from_outputs(row):
    """Raw outputs back to a band dict field_l_band.split_tissue can render.

    Normals are normalised on the way out for the same reason unpack() does it: an
    un-normalised normal silently rescales its offset.
    """
    ncm = np.array([float(row[0]), float(row[1])])
    cmm = np.array([float(row[3]), float(row[4])])

    ncm = ncm / max(np.linalg.norm(ncm), 1e-9)
    cmm = cmm / max(np.linalg.norm(cmm), 1e-9)

    return {
        "normal_y": float(ncm[0]),
        "normal_x": float(ncm[1]),
        "ncm_offset": float(row[2]) * CANVAS,
        "cmm_normal_y": float(cmm[0]),
        "cmm_normal_x": float(cmm[1]),
        "cmm_offset": float(row[5]) * CANVAS,
    }


def evaluate(model, inputs, labels, indices):
    """Per section: what fraction of each region's PIXELS land on the right side.

    Pixels rather than cells, because only 8 of the 14 sections have cells and every fold
    has to be scoreable. score_region_cells.py does the cells version on the saved model;
    this is the training-time proxy, and it is a proxy for the right thing rather than for
    area.
    """
    from field_l_band import split_tissue

    model.eval()
    rows = []

    with torch.no_grad():
        outputs = model(inputs[indices]).cpu().numpy()

    for position, index in enumerate(indices):
        tissue = inputs[index, 1].cpu().numpy() > 0.5
        drawn_labels = labels[index].cpu().numpy()

        ncm_side, cmm_side = split_tissue(bands_from_outputs(outputs[position]), tissue)

        row = {}

        for name, value, predicted in (
            ("NCM", 1, ncm_side),
            ("CMM", 2, cmm_side),
        ):
            drawn = drawn_labels == value
            total = max(int(drawn.sum()), 1)

            row[f"{name}_correct"] = float((drawn & predicted).sum() / total)
            row[f"{name}_error"] = float(predicted.sum() / total - 1)
            row[f"{name}_iou"] = float(
                (predicted & drawn).sum() / max((predicted | drawn).sum(), 1)
            )

        rows.append(row)

    return rows


def train(inputs, labels, device, epochs=EPOCHS, augment_data=True, quiet=False):
    generator = torch.Generator(device=device).manual_seed(SEED)
    torch.manual_seed(SEED)

    model = BoxNet(in_channels=inputs.shape[1], outputs=OUTPUTS).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    count = inputs.shape[0]

    # The soft masks are rendered at full canvas resolution, which is where the loss gets
    # its spatial precision -- only the TRUNK is downsampled, for speed.
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(count, generator=generator, device=device)
        total = 0.0
        seen = 0

        for start in range(0, count, BATCH_SIZE):
            chunk = order[start : start + BATCH_SIZE]
            batch_inputs, batch_labels = inputs[chunk], labels[chunk]

            if augment_data:
                batch_inputs, batch_labels = augment(
                    batch_inputs, batch_labels, generator
                )

            # Anchors are recomputed from the augmented labels rather than transformed
            # algebraically -- working out how a rotation composes with a fitted normal is
            # exactly where a sign error hides, and band_from_labels already gets it right
            # from pixels.
            anchors = []
            usable = []

            for position in range(batch_labels.shape[0]):
                mask = batch_labels[position].cpu().numpy()
                values = targets_for(mask, None)

                if values is None:
                    continue

                anchors.append(values)
                usable.append(position)

            if not usable:
                continue

            keep = torch.tensor(usable, dtype=torch.long, device=device)
            outputs = model(batch_inputs[keep])

            loss = split_loss(outputs, batch_labels[keep])

            if ANCHOR_WEIGHT:
                target = torch.tensor(anchors, dtype=torch.float32, device=device)
                # Smooth L1 rather than MSE: with 13 training sections one badly-fitted
                # section would otherwise dominate through its squared error.
                loss = loss + ANCHOR_WEIGHT * F.smooth_l1_loss(
                    outputs, target, beta=0.05
                )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total += loss.item() * len(usable)
            seen += len(usable)

        schedule.step()

        if not quiet and ((epoch + 1) % 40 == 0 or epoch == 0):
            print(f"    epoch {epoch + 1:4d}/{epochs}  loss {total / max(seen, 1):.5f}")

    return model


def summarise(rows):
    out = {}

    for name in ("NCM", "CMM"):
        out[name] = {
            "mean_correct": float(np.mean([r[f"{name}_correct"] for r in rows])),
            "median_correct": float(np.median([r[f"{name}_correct"] for r in rows])),
            "median_abs_error": float(
                np.median(np.abs([r[f"{name}_error"] for r in rows]))
            ),
            "median_iou": float(np.median([r[f"{name}_iou"] for r in rows])),
        }

    return out


def main():
    quick = "--quick" in sys.argv
    augment_data = "--no-augment" not in sys.argv
    epochs = 60 if quick else EPOCHS

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    birds = np.array([str(b) for b in data["birds"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.from_numpy(data["inputs"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).long().to(device)

    parameters = sum(
        p.numel() for p in BoxNet(outputs=OUTPUTS).parameters()
    )

    print(
        f"BandNet: {parameters:,} parameters, predicting {OUTPUTS} numbers "
        f"(4 degrees of freedom)\n"
        f"device {device}, {epochs} epochs, augment {augment_data}, "
        f"temperature {TEMPERATURE:.0f} px, anchor {ANCHOR_WEIGHT}\n"
    )

    bird_names = sorted(set(birds))
    results = []

    for held_out in bird_names if not quick else bird_names[:1]:
        test = birds == held_out
        train_indices = np.nonzero(~test)[0]
        test_indices = np.nonzero(test)[0]

        print(f"=== hold out {held_out} ({test.sum()} sections) ===")
        started = time.time()

        model = train(
            inputs[train_indices],
            labels[train_indices],
            device,
            epochs=epochs,
            augment_data=augment_data,
        )

        rows = evaluate(model, inputs, labels, test_indices)

        for position, index in enumerate(test_indices):
            row = rows[position]
            print(
                f"  {stems[index][:32]:34s} NCM {row['NCM_correct']:4.0%} "
                f"IoU {row['NCM_iou']:.2f}   CMM {row['CMM_correct']:4.0%} "
                f"IoU {row['CMM_iou']:.2f}"
            )

        results += rows
        print(f"  ({time.time() - started:.0f}s)\n")

    overall = summarise(results)

    print("=== cross-bird, held out by bird ===")
    print(f"{'region':8s} {'pixels right':>13} {'median |err|':>13} {'median IoU':>11}")

    for name in ("NCM", "CMM"):
        stats = overall[name]
        print(
            f"{name:8s} {stats['mean_correct']:12.0%} "
            f"{stats['median_abs_error']:12.0%} {stats['median_iou']:11.2f}"
        )

    print(
        "\nreference points:\n"
        "  BoxNet cross-bird:                 NCM IoU 0.55, CMM IoU 0.31\n"
        "  ceiling from Alice's OWN borders:   99% of NCM cells, 100% of CMM cells\n"
        "  (cells, not pixels: run score_region_cells.py on the saved model)"
    )

    if quick:
        print("\n--quick: one fold only, no model saved.")
        return

    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRESS_PATH).write_text(
        json.dumps(
            {"epochs": epochs, "augmented": augment_data, "cross_bird": overall},
            indent=1,
        )
    )

    model = train(inputs, labels, device, epochs=epochs, augment_data=augment_data)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "BandNet",
            "in_channels": inputs.shape[1],
            "outputs": OUTPUTS,
            "canvas": CANVAS,
            "fields": BAND_FIELDS,
            "temperature": TEMPERATURE,
            "epochs": epochs,
            "augmented": augment_data,
            "sections": stems,
            "cross_bird": overall,
        },
        MODEL_PATH,
    )

    print(f"\nwrote {MODEL_PATH} and {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
