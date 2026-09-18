"""Predict each region as 5 numbers instead of 230,400 pixel decisions.

WHY THIS REPLACES THE U-NET. train_regions.py asks a per-pixel classifier to decide every
pixel independently, so it has no way to express "this boundary is straight" or "this edge
follows the tissue" -- those are facts about the whole shape, and its output was
correspondingly blobby (50% median area error cross-bird). region_boxes.py showed that a
rotated rectangle intersected with the tissue mask reproduces Alice's regions to IoU 0.93
(NCM) / 0.77 (CMM). So the shape is already known; only its placement has to be learned.

Straightness and tissue-following now hold BY CONSTRUCTION. The network cannot emit a
blobby region because it does not emit regions at all -- it emits ten numbers, and the mask
is built from them.

WHAT IT OUTPUTS: 5 parameters per region x 2 regions = 10, plus the angle carried as
(sin 2t, cos 2t) so 12 raw outputs. See region_boxes.BOX_FIELDS.

GRADED THE SAME WAY AS THE U-NET -- area error and IoU on rebuilt masks, held out by bird,
7 folds. Not on parameter error: a 3-pixel centre error and a 3-pixel width error are the
same size in parameter space and completely different in area, and area is the denominator
of every density the lab reports. Comparing to the U-Net requires the same metric on the
same folds.

Usage:
  python training/train_boxes.py                 # 7-fold leave-one-bird-out, then final model
  python training/train_boxes.py --quick         # 1 fold, fewer epochs
  python training/train_boxes.py --no-augment    # ablation
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
from region_boxes import BOX_FIELDS, box_from_mask, render_box
from train_regions import CLASS_NAMES, augment

MODEL_PATH = "ps6_region_boxes.pt"
PROGRESS_PATH = "grids/box folds.json"

SEED = 0
EPOCHS = 400
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4

WIDTH = 16

# The trunk sees a downsampled canvas. Profiling said the network -- not the target
# recomputation I assumed -- was the cost: 922 ms per batch of 4 at 480x480 against 59 ms
# to derive a section's targets. Halving each side quarters the convolution work.
#
# Downsampling is safe here in a way it would never be for the cell detector: a PS6 cell is
# 1-2 px and would simply vanish, but a region box is a few hundred pixels across and its
# placement is a whole-section judgement. Losing fine texture costs nothing. Targets and
# the rebuilt masks stay at full resolution -- only the trunk's input shrinks.
TRUNK_DOWNSAMPLE = 2

# Outputs are scaled into roughly [-1, 1] before the loss sees them, so no single
# parameter dominates by virtue of its units. A centre lives in pixels (0-480) and
# sin2t lives in [-1, 1]; without scaling the centre's gradient would be ~500x larger and
# the angle would never be learned.
CANVAS = 480.0


class BoxNet(nn.Module):
    """Convolutional trunk down to a global summary, then a linear head to 12 numbers.

    The global average pool is the important part. Region placement is a whole-section
    judgement -- "NCM is the ventromedial block on the far side of Field L" -- so the head
    must see the entire canvas at once. A per-pixel decoder would reintroduce exactly the
    locality that made the U-Net blobby.
    """

    def __init__(self, in_channels=2, outputs=12, width=WIDTH):
        super().__init__()

        # GroupNorm for the same reason as train_regions.UNet: batch statistics over 4
        # images of 13 are noisier than the signal they normalise.
        def block(inside, outside):
            return nn.Sequential(
                nn.Conv2d(inside, outside, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, outside), outside),
                nn.ReLU(inplace=True),
                nn.Conv2d(outside, outside, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, outside), outside),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.trunk = nn.Sequential(
            block(in_channels, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 8),
        )
        self.head = nn.Linear(width * 8, outputs)

    def forward(self, x):
        if TRUNK_DOWNSAMPLE != 1:
            x = F.avg_pool2d(x, TRUNK_DOWNSAMPLE)

        features = self.trunk(x)

        # Mean over space: 12 numbers describing the section, not a map.
        pooled = features.mean(dim=(2, 3))

        return self.head(pooled)


def targets_for(labels, tissue):
    """The 12 numbers, scaled, for one section. Order fixed by BOX_FIELDS."""
    drawn = {"NCM": labels == 1, "CMM": labels == 2}
    values = []

    for name, other in (("NCM", "CMM"), ("CMM", "NCM")):
        box = box_from_mask(drawn[name], drawn[other], tissue)

        values += [
            box["centre_y"] / CANVAS,
            box["centre_x"] / CANVAS,
            box["half_along"] / CANVAS,
            box["half_across"] / CANVAS,
            box["sin2t"],
            box["cos2t"],
        ]

    return values


def boxes_from_outputs(row):
    """Undo the scaling: 12 raw outputs back to two box dictionaries."""
    boxes = {}

    for offset, name in ((0, "NCM"), (6, "CMM")):
        values = row[offset : offset + 6]

        boxes[name] = {
            "centre_y": float(values[0]) * CANVAS,
            "centre_x": float(values[1]) * CANVAS,
            # Half-lengths are widths, so a negative one is meaningless; abs() rather than
            # a clamp so a sign error stays a size rather than collapsing to zero area and
            # producing an empty mask that scores 0 IoU for the wrong reason.
            "half_along": abs(float(values[2])) * CANVAS,
            "half_across": abs(float(values[3])) * CANVAS,
            "sin2t": float(values[4]),
            "cos2t": float(values[5]),
        }

    return boxes


def evaluate(model, inputs, labels, indices):
    """Area error and IoU per section, from rebuilt masks -- the U-Net's metric."""
    model.eval()
    rows = []

    with torch.no_grad():
        outputs = model(inputs[indices]).cpu().numpy()

    for position, index in enumerate(indices):
        tissue = inputs[index, 1].cpu().numpy() > 0.5
        drawn_labels = labels[index].cpu().numpy()
        boxes = boxes_from_outputs(outputs[position])

        row = {}

        for name, value in (("NCM", 1), ("CMM", 2)):
            drawn = drawn_labels == value
            rebuilt = render_box(boxes[name], tissue)

            row[f"{name}_error"] = float(rebuilt.sum() / max(drawn.sum(), 1) - 1)
            row[f"{name}_iou"] = float(
                (rebuilt & drawn).sum() / max((rebuilt | drawn).sum(), 1)
            )

        rows.append(row)

    return rows


def train(inputs, targets, labels, device, epochs=EPOCHS, augment_data=True, quiet=False):
    generator = torch.Generator(device=device).manual_seed(SEED)
    torch.manual_seed(SEED)

    model = BoxNet(in_channels=inputs.shape[1]).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    count = inputs.shape[0]

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(count, generator=generator, device=device)
        total = 0.0

        for start in range(0, count, BATCH_SIZE):
            chunk = order[start : start + BATCH_SIZE]
            batch_inputs, batch_labels = inputs[chunk], labels[chunk]

            if augment_data:
                batch_inputs, batch_labels = augment(
                    batch_inputs, batch_labels, generator
                )

            # Targets are RECOMPUTED from the augmented labels rather than transformed
            # algebraically. Working out how a rotation composes with a fitted Field L
            # direction and a doubled angle is exactly the kind of derivation that hides a
            # sign error, and box_from_mask already does it correctly from pixels.
            moved = []
            usable = []

            for position in range(batch_labels.shape[0]):
                mask = batch_labels[position].cpu().numpy()
                tissue = batch_inputs[position, 1].cpu().numpy() > 0.5

                # A hard rotation can push a region off the canvas entirely; fitting a line
                # to an empty mask would raise rather than train.
                if not ((mask == 1).any() and (mask == 2).any()):
                    continue

                moved.append(targets_for(mask, tissue))
                usable.append(position)

            if not usable:
                continue

            batch_targets = torch.tensor(
                moved, dtype=torch.float32, device=device
            )
            keep = torch.tensor(usable, dtype=torch.long, device=device)

            # Smooth L1, not plain MSE: one badly-fitted section would otherwise dominate
            # the gradient through its squared error, and with 13 training sections there
            # is no averaging to absorb that.
            loss = F.smooth_l1_loss(
                model(batch_inputs[keep]), batch_targets, beta=0.05
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total += loss.item() * len(usable)

        schedule.step()

        if not quiet and ((epoch + 1) % 40 == 0 or epoch == 0):
            print(f"    epoch {epoch + 1:4d}/{epochs}  loss {total / max(count, 1):.5f}")

    return model


def summarise(rows):
    out = {}

    for name in ("NCM", "CMM"):
        errors = np.array([r[f"{name}_error"] for r in rows])
        ious = np.array([r[f"{name}_iou"] for r in rows])

        out[name] = {
            "median_abs_error": float(np.median(np.abs(errors))),
            "bias": float(errors.mean()),
            "median_iou": float(np.median(ious)),
        }

    return out


def main():
    quick = "--quick" in sys.argv
    augment_data = "--no-augment" not in sys.argv
    epochs = 60 if quick else EPOCHS

    data = np.load(DATA_PATH)
    stems = [str(s) for s in data["stems"]]
    birds = np.array([str(b) for b in data["birds"]])

    # Same choice train_regions.py makes: CPU is fine here. BoxNet is small and the
    # dataset is 14 images -- the bottleneck is recomputing targets from augmented labels
    # on the CPU each step, which a GPU would not touch.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.from_numpy(data["inputs"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).long().to(device)

    targets = torch.tensor(
        [
            targets_for(data["labels"][i], data["inputs"][i, 1] > 0.5)
            for i in range(len(stems))
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"BoxNet: {sum(p.numel() for p in BoxNet().parameters()):,} parameters, "
        f"predicting {len(BOX_FIELDS) * 2} numbers per section\n"
        f"device {device}, {epochs} epochs, augment {augment_data}\n"
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
            targets[train_indices],
            labels[train_indices],
            device,
            epochs=epochs,
            augment_data=augment_data,
        )

        rows = evaluate(model, inputs, labels, test_indices)

        for position, index in enumerate(test_indices):
            row = rows[position]
            print(
                f"  {stems[index][:32]:34s} NCM {row['NCM_error']:+6.0%} "
                f"IoU {row['NCM_iou']:.2f}   CMM {row['CMM_error']:+6.0%} "
                f"IoU {row['CMM_iou']:.2f}"
            )

        results += rows
        print(f"  ({time.time() - started:.0f}s)\n")

    overall = summarise(results)

    print("=== cross-bird, held out by bird ===")
    print(f"{'region':8s} {'median |err|':>13} {'bias':>7} {'median IoU':>11}")

    for name in ("NCM", "CMM"):
        stats = overall[name]
        print(
            f"{name:8s} {stats['median_abs_error']:12.0%} "
            f"{stats['bias']:+7.0%} {stats['median_iou']:11.2f}"
        )

    print(
        "\nreference points:\n"
        "  per-pixel U-Net, cross-bird:      NCM/CMM about 50% median area error\n"
        "  ceiling with Alice's OWN lines:   NCM 3% IoU 0.93, CMM 23% IoU 0.77"
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

    model = train(inputs, targets, labels, device, epochs=epochs, augment_data=augment_data)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "BoxNet",
            "in_channels": inputs.shape[1],
            "width": WIDTH,
            "canvas": CANVAS,
            "fields": BOX_FIELDS,
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
