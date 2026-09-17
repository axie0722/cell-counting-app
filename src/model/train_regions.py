"""Train a U-Net to predict the NCM and CMM outlines, held out by bird.

Predicts three classes per pixel -- background, NCM, CMM -- from the cached maps
prepare_regions.py writes. dNCM and vNCM are NOT predicted: ps6.resolve_regions derives
them by cutting NCM at the dorsal marker along the Field L axis, so two predictions yield
all three countable regions and the subdivision keeps coming from a stated rule.

GRADED ON AREA ERROR, not Dice. Dice is the conventional segmentation metric and would be
the wrong headline here: region area is the denominator of every density the lab reports,
so a 15% area error IS a 15% density error, while Dice mixes area error together with
boundary placement and hides which one moved. IoU is reported alongside as the check that
the shape is in the right place at all -- a prediction can have exactly the right area in
entirely the wrong location, and only IoU catches that.

HELD OUT BY BIRD, 7 folds, the same discipline train_cnn.py uses. 14 sections is not 14
independent examples: YW113 contributes 4 and Wh175 3, and sections of one brain share an
animal, a staining batch and a scan session. Splitting them would measure memorisation.

AUGMENTATION IS VERTICAL-FLIP ONLY, and that asymmetry is measured, not assumed. Across
all 14 sections CMM sits on the +x side of NCM without exception, while the vertical
offset flips sign (8 sections CMM above NCM, 6 below) and the flip does not track
hemisphere -- the two Wh175 LH sections disagree with each other. So slide placement is
arbitrary vertically and fixed horizontally. A horizontal flip would manufacture a
left-handed arrangement that never occurs in the data and teaches the model a
configuration it will never be asked about. Small rotations and shifts are added because
those vary continuously in how a slide is laid down.

Usage:
  python train_regions.py                 # 7-fold leave-one-bird-out, then final model
  python train_regions.py --quick         # 1 fold, fewer epochs, for a sanity check
  python train_regions.py --no-augment    # ablation
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "ps6_region_maps.npz"
MODEL_PATH = "ps6_region_unet.pt"
PROGRESS_PATH = "grids/region folds.json"

SEED = 0
EPOCHS = 120
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4

# Minibatches, not the full 13-section batch. The first attempt trained full-batch and
# spent 906s of SYSTEM time against 632s of user time -- that ratio is swap thrashing, not
# arithmetic. Backpropagating through 13 images of 480x480 stores gigabytes of activations
# on a machine that had ~57 MB free. Four at a time also gives 4 gradient steps per pass
# instead of 1, so it learns more per epoch as well as fitting in memory.
BATCH_SIZE = 4

# Base channel width. Small on purpose: 14 training images cannot support a wide network,
# and the target is one smooth blob rather than fine texture.
WIDTH = 16

# Background dominates the canvas, so an unweighted loss is minimised well by predicting
# background everywhere. Weights are the inverse of each class's pixel share, computed
# from the training fold only -- computing them over all 14 would leak the held-out
# section's region sizes into training.
CLASS_NAMES = ["background", "NCM", "CMM"]

AUGMENT_ROTATION_DEGREES = 12.0
AUGMENT_SHIFT_FRACTION = 0.06
AUGMENT_SCALE_RANGE = 0.10


class UNet(nn.Module):
    """Three-level U-Net. Small enough to train on CPU in minutes on 14 images.

    Skip connections matter more than depth here: the boundary the model must place is
    visible as a local intensity change, while WHICH side of the brain it is on is a
    whole-section judgement. The downsampling path supplies the second, the skips
    preserve the first.
    """

    def __init__(self, in_channels=2, classes=3, width=WIDTH):
        super().__init__()

        # GroupNorm, not BatchNorm. BatchNorm's statistics come from the batch, so with 4
        # images per step they are noisier than the signal, and the running averages it
        # keeps for evaluation would be estimated from 13 images total. GroupNorm
        # normalises within each image and is batch-size independent, which is what makes
        # minibatching safe here rather than a memory-driven compromise -- and on 13
        # training images it is the better choice regardless.
        def block(inside, outside):
            groups = min(8, outside)

            return nn.Sequential(
                nn.Conv2d(inside, outside, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outside),
                nn.ReLU(inplace=True),
                nn.Conv2d(outside, outside, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outside),
                nn.ReLU(inplace=True),
            )

        self.down1 = block(in_channels, width)
        self.down2 = block(width, width * 2)
        self.down3 = block(width * 2, width * 4)
        self.middle = block(width * 4, width * 8)

        self.up3 = block(width * 8 + width * 4, width * 4)
        self.up2 = block(width * 4 + width * 2, width * 2)
        self.up1 = block(width * 2 + width, width)

        self.head = nn.Conv2d(width, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        one = self.down1(x)
        two = self.down2(self.pool(one))
        three = self.down3(self.pool(two))
        deep = self.middle(self.pool(three))

        def up(feature, skip, block):
            feature = F.interpolate(
                feature, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

            return block(torch.cat([feature, skip], dim=1))

        x = up(deep, three, self.up3)
        x = up(x, two, self.up2)
        x = up(x, one, self.up1)

        return self.head(x)


def augment(inputs, labels, generator):
    """Random small rotation, shift and scale, plus a vertical flip.

    One affine grid applied to both the image and the labels, so they cannot drift out of
    register. Labels are sampled NEAREST -- interpolating a class index would invent a
    label 1.4 between NCM and CMM.
    """
    batch = inputs.shape[0]
    device = inputs.device

    def uniform(low, high):
        return torch.rand(batch, generator=generator, device=device) * (high - low) + low

    angle = uniform(-AUGMENT_ROTATION_DEGREES, AUGMENT_ROTATION_DEGREES) * np.pi / 180.0
    scale = uniform(1 - AUGMENT_SCALE_RANGE, 1 + AUGMENT_SCALE_RANGE)
    shift_y = uniform(-AUGMENT_SHIFT_FRACTION, AUGMENT_SHIFT_FRACTION) * 2
    shift_x = uniform(-AUGMENT_SHIFT_FRACTION, AUGMENT_SHIFT_FRACTION) * 2

    # Vertical flip only. See the module docstring: the up/down arrangement of CMM
    # relative to NCM genuinely varies in the data, the left/right arrangement never does.
    flip = torch.where(
        torch.rand(batch, generator=generator, device=device) < 0.5, -1.0, 1.0
    )

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale

    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin * flip
    theta[:, 0, 2] = shift_x
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos * flip
    theta[:, 1, 2] = shift_y

    grid = F.affine_grid(theta, list(inputs.shape), align_corners=False)

    moved_inputs = F.grid_sample(
        inputs, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # +1 then -1 around the sample so that anything rotated in from outside the canvas
    # becomes background (0) rather than a spurious NCM label.
    moved_labels = F.grid_sample(
        (labels.float() + 1.0).unsqueeze(1),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    moved_labels = (moved_labels.squeeze(1) - 1.0).clamp(min=0).long()

    return moved_inputs, moved_labels


def class_weights(labels):
    """Inverse-frequency weights from the TRAINING fold only, so nothing leaks."""
    counts = torch.bincount(labels.reshape(-1), minlength=len(CLASS_NAMES)).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)

    # Capped because NCM and CMM are ~2-6% of the canvas each, and an uncapped inverse
    # weight lets a handful of boundary pixels dominate the gradient and produce a model
    # that floods the section with region.
    return weights.clamp(max=20.0)


def train(inputs, labels, device, epochs=EPOCHS, augment_data=True, quiet=False):
    torch.manual_seed(SEED)
    generator = torch.Generator(device=device).manual_seed(SEED)

    model = UNet(in_channels=inputs.shape[1]).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    weights = class_weights(labels).to(device)

    inputs, labels = inputs.to(device), labels.to(device)
    model.train()

    for epoch in range(epochs):
        # A fresh shuffle each epoch, so the minibatches are not the same 4 sections
        # every time -- with 13 images a fixed grouping would let the model fit each
        # group's peculiarities rather than the task.
        order = torch.randperm(len(inputs), generator=generator, device=device)
        total = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            chunk = order[start : start + BATCH_SIZE]
            batch_inputs, batch_labels = inputs[chunk], labels[chunk]

            if augment_data:
                batch_inputs, batch_labels = augment(
                    batch_inputs, batch_labels, generator
                )

            optimiser.zero_grad()
            loss = F.cross_entropy(model(batch_inputs), batch_labels, weight=weights)
            loss.backward()
            optimiser.step()

            total += loss.item() * len(chunk)

        schedule.step()

        if not quiet and (epoch + 1) % 20 == 0:
            print(
                f"    epoch {epoch + 1:4d}/{epochs}  loss {total / len(order):.4f}",
                flush=True,
            )

    model.eval()

    return model


def predict(model, one_input, device):
    with torch.no_grad():
        logits = model(one_input.unsqueeze(0).to(device))

        return logits.argmax(dim=1)[0].cpu().numpy()


def score(predicted, truth, scale_um):
    """Per-region area error and IoU. Area error is the headline; see the docstring."""
    out = {}

    for value, name in enumerate(CLASS_NAMES):
        if name == "background":
            continue

        p, t = predicted == value, truth == value
        area_scale = scale_um**2 / 1e6

        predicted_mm2 = p.sum() * area_scale
        truth_mm2 = t.sum() * area_scale
        union = (p | t).sum()

        out[name] = {
            "predicted_mm2": float(predicted_mm2),
            "truth_mm2": float(truth_mm2),
            "area_error": float(predicted_mm2 / truth_mm2 - 1) if truth_mm2 else float("nan"),
            "iou": float((p & t).sum() / union) if union else float("nan"),
        }

    return out


def load_progress(epochs, augment_data):
    """Folds already finished, or [] if the file is missing or from a different setting.

    Both the epoch count and the augmentation flag are checked, because resuming across
    either would average folds trained differently and report the mean as one model --
    which would quietly turn the --no-augment ablation into a meaningless mixture.
    """
    if not Path(PROGRESS_PATH).exists():
        return []

    saved = json.loads(Path(PROGRESS_PATH).read_text())

    if saved.get("epochs") != epochs or saved.get("augmented") != augment_data:
        print(f"  {PROGRESS_PATH} is from a different setting; starting over")

        return []

    results = saved.get("results", [])

    if results:
        print(f"  resuming: {len(results)} sections already done")

    return results


def save_progress(results, epochs, augment_data):
    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRESS_PATH).write_text(
        json.dumps(
            {"epochs": epochs, "augmented": augment_data, "results": results}, indent=1
        )
    )


def main():
    quick = "--quick" in sys.argv
    augment_data = "--no-augment" not in sys.argv
    epochs = 40 if quick else EPOCHS

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(DATA_PATH, allow_pickle=False)
    inputs = torch.from_numpy(data["inputs"])
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    stems = [str(s) for s in data["stems"]]
    birds = np.array([str(b) for b in data["birds"]])
    scale_um = float(data["scale_um"])

    print(f"device: {device}   {len(stems)} sections at {scale_um:g} um/px")
    print(f"augmentation: {'vertical flip + affine' if augment_data else 'OFF'}")

    share = torch.bincount(labels.reshape(-1), minlength=3).float()
    share = share / share.sum()
    print(
        "canvas class share: "
        + ", ".join(f"{n} {s:.1%}" for n, s in zip(CLASS_NAMES, share.tolist()))
    )

    bird_names = sorted(set(birds))

    if quick:
        bird_names = bird_names[:1]

    print(f"\nleave-one-bird-out over {bird_names}\n")

    # Folds are saved as they finish and skipped on restart, for the reason spelled out in
    # train_backbone.PROGRESS_PATH: Alice's machine crashed mid-run on 2026-08-09 and an
    # 8 GB machine under memory pressure will do it again. --quick never resumes, since a
    # sanity check should always run the current code.
    results = [] if quick else load_progress(epochs, augment_data)
    done = {r["bird"] for r in results}

    for held_out in bird_names:
        if held_out in done:
            print(f"=== hold out {held_out}: already done, skipping ===", flush=True)
            continue

        test = birds == held_out
        train_mask = ~test

        started = time.time()
        print(
            f"=== hold out {held_out} "
            f"({train_mask.sum()} train sections, {test.sum()} test) ===",
            flush=True,
        )

        model = train(
            inputs[train_mask], labels[train_mask], device,
            epochs=epochs, augment_data=augment_data,
        )

        for index in np.flatnonzero(test):
            predicted = predict(model, inputs[index], device)
            measured = score(predicted, labels[index].numpy(), scale_um)

            results.append({"stem": stems[index], "bird": held_out, **measured})

            print(
                "  "
                + "  ".join(
                    f"{name} area {measured[name]['area_error']:+6.1%} "
                    f"IoU {measured[name]['iou']:.2f}"
                    for name in ("NCM", "CMM")
                )
                + f"   {stems[index][:30]}"
            )

        del model

        if not quick:
            save_progress(results, epochs, augment_data)

        print(f"  ({(time.time() - started) / 60:.1f} min, saved)", flush=True)

    print("\n=== cross-bird summary (each section predicted by a model that never saw "
          "its bird) ===")

    for name in ("NCM", "CMM"):
        errors = np.array([r[name]["area_error"] for r in results])
        ious = np.array([r[name]["iou"] for r in results])

        print(
            f"  {name}: median |area error| {np.median(np.abs(errors)):.1%}, "
            f"worst {np.max(np.abs(errors)):.1%}, "
            f"bias {np.mean(errors):+.1%}   median IoU {np.median(ious):.2f}"
        )

    # The ratio is the number review_regions.py tracks and the one the redraw was about,
    # so it is worth checking the model reproduces it rather than only getting each region
    # separately right.
    predicted_ratio = np.array(
        [r["CMM"]["predicted_mm2"] / r["NCM"]["predicted_mm2"] for r in results]
    )
    truth_ratio = np.array(
        [r["CMM"]["truth_mm2"] / r["NCM"]["truth_mm2"] for r in results]
    )
    print(
        f"  CMM/NCM ratio: predicted median {np.median(predicted_ratio):.2f} "
        f"(range {predicted_ratio.min():.2f}-{predicted_ratio.max():.2f}) "
        f"vs drawn median {np.median(truth_ratio):.2f} "
        f"(range {truth_ratio.min():.2f}-{truth_ratio.max():.2f})"
    )

    if quick:
        print("\n--quick: one fold only, no model saved.")
        return

    print("\n=== final model, all sections ===", flush=True)
    model = train(inputs, labels, device, epochs=epochs, augment_data=augment_data)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "UNet",
            "width": WIDTH,
            "in_channels": inputs.shape[1],
            "classes": CLASS_NAMES,
            "scale_um": scale_um,
            "canvas": int(inputs.shape[-1]),
            "epochs": epochs,
            "augmented": augment_data,
            "sections": stems,
        },
        MODEL_PATH,
    )
    print(f"saved {MODEL_PATH}")
    print(
        "\nThe honest numbers are the cross-bird ones above; this final model has seen\n"
        "every section, so its fit to them says nothing about a new bird."
    )


if __name__ == "__main__":
    main()
