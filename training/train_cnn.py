"""Train a small CNN to separate real PS6 cells from detector false positives.

Evaluation is leave-one-bird-out, not a random split. Patches from one section
share background texture and staining intensity, so a random split leaks and
reports a number far better than the counter will reach on a new bird. Holding
out whole sections measures the thing that matters: does this transfer?

Uses CUDA when present, CPU otherwise.

Three archives can be trained on, and the choice matters more than any
hyperparameter here:

  ps6_patches.npz            97891 patches, 420 positive. Negatives are GUESSES:
                             any candidate without a nearby click was called
                             negative, and 38% of the ones sitting on a real cell
                             are mislabelled.
  ps6_window_patches.npz     22170 patches, 101 positive, negatives VERIFIED
    (--windows)              inside exhaustively-clicked windows. Too few
                             positives on its own -- measured 2026-08-05, blind
                             recall fell from 71.4% to 57.1%.
  ps6_combined_patches.npz   22573 patches, 504 positive. Every positive from both
    (--combined)             archives plus the verified negatives only. This is
                             the one to use: the poisoned negatives were the
                             harmful part, and a click marks a real cell whichever
                             pass found it. Built by combine_patches.py.
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

PATCH_PATH = "ps6_patches.npz"
WINDOW_PATCH_PATH = "ps6_window_patches.npz"
COMBINED_PATCH_PATH = "ps6_combined_patches.npz"
MODEL_PATH = "ps6_cnn.pt"

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Negatives outnumber positives about 250 to 1. Sampling a fixed number per
# positive keeps each epoch balanced enough to learn from, while resampling
# across epochs still exposes the model to most of the negative set.
NEGATIVES_PER_POSITIVE = 20

# Hard-negative mining, OFF by default. Every result on record -- SmallCNN's 0.653 mean
# AP, the backbone's blind grid at 69.6% recall -- was measured with uniform negative
# sampling, so turning this on by default would make those numbers unreproducible
# without anyone noticing which change did it.
#
# Why it is worth trying: uniform sampling spends nearly the whole negative budget on
# empty background the model already scores at zero. On the combined archive that is
# 10460 of 46213 negatives per epoch drawn flat, and the objects that actually cost
# precision -- vessel walls, fibre bundles, out-of-focus haze, the things
# false_positive_gallery.py was written to look at -- are a small fraction of the pool
# getting the same 23% chance of being shown as blank tissue.
#
# Only HALF the budget is mined, for two reasons, and the first is specific to this
# project:
#   * the highest-scoring negatives are disproportionately MISLABELLED CELLS. Even the
#     exhaustive windows missed real cells -- 8 of 15 blind-grid false positives turned
#     out to be real on review (2026-08-08) -- and a real cell sitting in the negative
#     pool is exactly what a good model scores highest. Mining only the top of the
#     ranking would train the model to reject cells, which is the opposite of the goal.
#   * a fixed hard set stops being a sample of the negative distribution. The model
#     drifts onto it and forgets the easy background it currently handles for free.
HARD_NEGATIVE_FRACTION = 0.5

# The hard half is drawn from a pool this many times its own size rather than being the
# exact top of the ranking, so the same few thousand crops are not re-shown every epoch.
HARD_NEGATIVE_POOL_FACTOR = 2

# Epochs of uniform sampling before mining starts. An untrained model's ranking of
# negatives is noise, and mining from epoch 0 would lock onto an arbitrary subset and
# then spend the whole run reinforcing it.
HARD_NEGATIVE_WARMUP_EPOCHS = 3

# Epochs between rescoring the pool. One forward pass over every negative: seconds on
# the H100, a few minutes on the Mac. This interval is a CPU concession, not a
# modelling choice -- rescoring every epoch would be strictly more current.
HARD_NEGATIVE_REFRESH_EPOCHS = 3

# Negatives scored per call when ranking them. The stored patches are float32 80x80, so
# the whole negative pool is 1.2 GB and fancy-indexing it in one go is what would push
# an 8 GB Mac into swap.
HARD_NEGATIVE_SCORING_CHUNK = 8192

# Stain augmentation, aimed at the cross-bird gap. Ranges are deliberately modest:
# these are meant to span the difference between staining batches of the same
# protocol, not to invent appearances the scanner never produces.
#
# TESTED 2026-08-05 AND IT DOES NOT HELP -- left off, and left here so it is not
# re-tried as a fresh idea. Leave-one-bird-out mean AP moved 0.653 -> 0.670, which
# on test sets of 8-34 cells is noise, and the two unseen birds it was aimed at did
# not improve (LBlu59 0.470 -> 0.457, Wh175 0.437 -> 0.479) while familiar birds
# stayed at 0.73-0.90. On the blind grid it was strictly worse at matched recall:
# at 69.6% recall, precision 46.8% against the baseline's 57.8%, and best F1 0.624
# against 0.634. It mostly made the model less confident, which slides the operating
# point along the curve without moving the curve.
#
# The reasoning behind trying it was that normalise() already removes per-patch
# brightness, leaving contrast and intensity-distribution shape as the untested
# variables. That reasoning was wrong: whatever distinguishes a new bird is not
# contrast. Sweep kept at "grids/blind grid threshold sweep.stain.csv".
STAIN_JITTER = False
GAMMA_RANGE = (0.7, 1.4)
CONTRAST_RANGE = (0.75, 1.3)

SEED = 0


class SmallCNN(nn.Module):
    """Four conv blocks over 64x64 single-channel patches.

    Deliberately small. With 190 positives a heavier network memorises the
    training birds instead of learning the annulus shape, and the held-out bird
    is what we care about. The H100 could run something far larger; capacity is
    not the constraint here, labels are.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def normalise(patches):
    """Per-patch standardisation.

    Sections differ in mean brightness by more than a factor of two, so absolute
    intensity identifies the bird rather than the cell. Removing each patch's own
    mean and scale forces the model onto shape.

    Must run after any cropping: the mean and std of a stored 80x80 patch differ
    from those of the 64x64 crop the model actually sees.
    """
    flat = patches.reshape(len(patches), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-6

    return ((flat - mean) / std).reshape(patches.shape)


def scale_jitter(batch, generator):
    """Present the batch at a random apparent scale.

    Center crop to a random width, then resize back to 64. A narrower crop
    magnifies the cell, a wider one shrinks it, so the model meets a real cell at
    every apparent size the scanners produce -- and meets bilinear interpolation
    often enough to treat it as ordinary.

    Without this the model was sharply scale-specific: a 9% resample dropped
    confident cells from 52% to 1%. One width per batch rather than per patch,
    which is cheaper and still varies plenty across an epoch.
    """
    stored = batch.shape[-1]
    output = 2 * ps6.PATCH_HALF_WIDTH

    width = int(
        torch.randint(
            ps6.MIN_TRAIN_CROP,
            min(ps6.MAX_TRAIN_CROP, stored) + 1,
            (1,),
            generator=generator,
        ).item()
    )

    start = (stored - width) // 2
    cropped = batch[..., start : start + width, start : start + width]

    if width == output:
        return cropped

    # align_corners=False and bilinear match skimage's resize closely enough that
    # inference does not meet a different interpolation than training.
    return F.interpolate(
        cropped, size=(output, output), mode="bilinear", align_corners=False
    )


def standardise_batch(batch):
    """Per-patch standardisation on a tensor, after cropping.

    The numpy normalise() runs before training on stored patches; scale jitter
    changes each patch's content, so the statistics must be recomputed here.
    """
    flat = batch.reshape(batch.shape[0], -1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    # unbiased=False to match numpy's default, so training and evaluation
    # standardise identically.
    std = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1) + 1e-6

    return (batch - mean) / std


def stain_jitter(batch, generator):
    """Present the batch as if it were stained a little differently.

    Cross-bird generalization is the measured failure: trained without Wh175 the
    model scores AP 0.437 on it, against 0.873 on Purp30 (2026-08-05). What differs
    between birds is the staining batch and the scan session.

    normalise() and standardise_batch() already remove each patch's mean and spread,
    so absolute brightness cannot be the cue. What survives standardisation is the
    SHAPE of the intensity distribution, and that is exactly what a stronger or
    weaker stain changes. So jitter the shape:

      gamma     bends the intensity curve, moving mid-greys relative to the
                extremes -- a weakly stained cell is dimmer in its middle tones
                without its peak changing much.
      contrast  stretches or compresses the spread about the mean, which survives
                standardisation as a change in how far the annulus stands out from
                its background relative to the noise.

    Applied to raw stored values, before standardisation, because gamma needs
    non-negative input and standardisation must be the last step so the model always
    receives zero-mean unit-variance patches. One draw per batch, matching
    scale_jitter: cheaper, and still plenty of variety across an epoch.
    """
    # Shift to non-negative before the power. Patches come from the extractor as raw
    # green values, but a negative minimum would make a fractional power produce nan.
    low = batch.amin(dim=(1, 2, 3), keepdim=True)
    high = batch.amax(dim=(1, 2, 3), keepdim=True)
    spread = (high - low).clamp(min=1e-6)
    unit = (batch - low) / spread

    gamma = float(torch.empty(1).uniform_(*GAMMA_RANGE, generator=generator).item())
    unit = unit.clamp(min=0).pow(gamma)

    contrast = float(
        torch.empty(1).uniform_(*CONTRAST_RANGE, generator=generator).item()
    )
    mean = unit.mean(dim=(1, 2, 3), keepdim=True)

    return mean + (unit - mean) * contrast


def augment(batch, generator):
    """Random dihedral transform, apparent scale, and apparent stain.

    PS6 cells are rotationally symmetric annuli, so all 8 flips and quarter
    turns preserve the label. This is the cheapest source of extra positives.
    """
    if torch.rand(1, generator=generator).item() < 0.5:
        batch = torch.flip(batch, dims=[3])

    turns = int(torch.randint(0, 4, (1,), generator=generator).item())

    if turns:
        batch = torch.rot90(batch, turns, dims=[2, 3])

    batch = scale_jitter(batch, generator)

    if STAIN_JITTER:
        batch = stain_jitter(batch, generator)

    # Standardisation is always last: whatever the augmentation did to the intensity
    # distribution, the model must meet a zero-mean unit-variance patch, exactly as
    # predict() produces at inference.
    return standardise_batch(batch)


def predict(model, patches, device):
    """Probabilities for stored patches, cropped to the CNN's input size.

    Takes the raw stored patch and applies the same crop-then-standardise the
    training path ends on, so evaluation measures the deployed behaviour.
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(patches), 2048):
            chunk = ps6.to_model_input(patches[start : start + 2048])
            batch = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(1)
            outputs.append(
                torch.sigmoid(model(standardise_batch(batch.to(device)))).cpu().numpy()
            )

    return np.concatenate(outputs)


def summarise(probabilities, labels, name):
    """Report average precision, plus precision/recall at the best-F1 threshold.

    Average precision is the headline number: it summarises the whole
    precision-recall tradeoff without committing to a threshold, which matters
    because the right operating point depends on whether you would rather
    hand-remove false positives or hand-add missed cells.
    """
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]

    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, len(sorted_labels) + 1)
    recall = true_positives / max(1, sorted_labels.sum())
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    best = int(np.argmax(f1))
    average_precision = float(np.sum(np.diff(np.r_[0, recall]) * precision))

    print(
        f"  {name:8s} AP={average_precision:.3f}  F1={f1[best]:.3f} "
        f"at p>={probabilities[order][best]:.3f}  "
        f"precision={precision[best]:.1%} recall={recall[best]:.1%}  "
        f"({best + 1} predicted, {int(sorted_labels.sum())} real)"
    )

    return {
        "average_precision": average_precision,
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(probabilities[order][best]),
    }


def hard_negative_pool(model, patches, negative_index, device, size):
    """The `size` negatives this model currently scores highest.

    Ranked through predict(), so the scores come from the same crop-and-standardise path
    used at inference, with no augmentation: the question being asked is "which of these
    would the deployed model call a cell", and jittered scale would add noise to that.

    Scored in chunks because `patches[negative_index]` in one expression would copy the
    whole 1.2 GB negative pool. predict() leaves the model in eval mode; train()'s loop
    calls model.train() at the top of every epoch, after this returns.
    """
    scores = np.concatenate(
        [
            predict(
                model,
                patches[negative_index[start : start + HARD_NEGATIVE_SCORING_CHUNK]],
                device,
            )
            for start in range(0, len(negative_index), HARD_NEGATIVE_SCORING_CHUNK)
        ]
    )

    return negative_index[np.argsort(-scores)[:size]]


def train(
    patches,
    labels,
    device,
    epochs=EPOCHS,
    quiet=False,
    make_model=None,
    make_optimiser=None,
    mine_hard_negatives=False,
):
    """Train a fresh model. Seeded, so runs are comparable across splits.

    `make_model` and `make_optimiser` exist so an alternative architecture can be
    measured through this exact harness -- same seed, same negative resampling, same
    augmentation, same leave-one-bird-out loop. Any comparison run through a separate
    copy of this function would differ in ways nobody could enumerate, and the whole
    point of testing a new architecture is that the training procedure is the thing
    held constant. Both default to the shipped SmallCNN behaviour.

    `mine_hard_negatives` concentrates the negative budget on the crops this model
    currently scores highest instead of drawing it flat; see HARD_NEGATIVE_FRACTION. It
    defaults to False, and with it off this function is line-for-line what produced every
    number on record.
    """
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    positive_index = np.flatnonzero(labels == 1)
    negative_index = np.flatnonzero(labels == 0)

    model = (make_model or SmallCNN)().to(device)
    optimiser = (
        make_optimiser(model)
        if make_optimiser
        else torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

    sample_size = min(len(negative_index), NEGATIVES_PER_POSITIVE * len(positive_index))

    # Whatever imbalance survives the sampling, correct for in the loss.
    positive_weight = torch.tensor(
        sample_size / max(1, len(positive_index)),
        device=device,
    )

    # Mining ranks the pool and then draws the rest of the budget from what is left, so
    # it needs a negative set bigger than the pool. Below that, every negative is
    # already shown every epoch and there is nothing to concentrate.
    hard_count = int(round(sample_size * HARD_NEGATIVE_FRACTION))
    pool_size = hard_count * HARD_NEGATIVE_POOL_FACTOR
    mining = mine_hard_negatives and pool_size < len(negative_index)

    if mine_hard_negatives and not mining:
        print(
            f"    mining requested but skipped: {len(negative_index)} negatives is not "
            f"more than the {pool_size}-patch pool it would rank"
        )

    hard_pool = None
    easier = None

    for epoch in range(epochs):
        due = (epoch - HARD_NEGATIVE_WARMUP_EPOCHS) % HARD_NEGATIVE_REFRESH_EPOCHS == 0

        if mining and epoch >= HARD_NEGATIVE_WARMUP_EPOCHS and due:
            hard_pool = hard_negative_pool(
                model, patches, negative_index, device, pool_size
            )

            # The uniform half is drawn from everything OUTSIDE the pool. Drawing it from
            # the whole set instead would put pool members in both halves of the same
            # epoch, double-weighting them by accident on top of the weighting that is
            # the point of this.
            easier = np.setdiff1d(negative_index, hard_pool, assume_unique=True)

            if not quiet:
                print(
                    f"    epoch {epoch + 1:2d}/{epochs}  mined the top {pool_size} "
                    f"negatives; {hard_count} of {sample_size} per epoch now come "
                    f"from there"
                )

        if hard_pool is None:
            # Resample negatives each epoch, so over training the model sees most of
            # them without any single epoch being wildly imbalanced.
            sampled = rng.choice(negative_index, size=sample_size, replace=False)
        else:
            sampled = np.concatenate(
                [
                    rng.choice(hard_pool, size=hard_count, replace=False),
                    rng.choice(easier, size=sample_size - hard_count, replace=False),
                ]
            )

        epoch_index = rng.permutation(np.concatenate([positive_index, sampled]))

        model.train()
        total_loss = 0.0

        for start in range(0, len(epoch_index), BATCH_SIZE):
            chunk = epoch_index[start : start + BATCH_SIZE]

            batch = torch.from_numpy(patches[chunk]).unsqueeze(1).to(device)
            target = torch.from_numpy(labels[chunk]).float().to(device)

            loss = F.binary_cross_entropy_with_logits(
                model(augment(batch, generator)),
                target,
                pos_weight=positive_weight,
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * len(chunk)

        scheduler.step()

        if not quiet and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:2d}/{epochs}  loss {total_loss / len(epoch_index):.4f}")

    return model


def pick_device(announce=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if announce:
        name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"device: {device} ({name})")

    return device


def load_data(path=PATCH_PATH):
    """Return (stored patches, stored patches, labels, bird names).

    Patches are returned raw and unnormalised: standardisation now happens after
    cropping, inside the training and prediction paths, because the scale
    augmentation changes each patch's own statistics.

    The first two values are the same array, kept as a pair so callers written
    against the earlier (raw, normalised) signature still work.
    """
    data = np.load(path)

    raw = data["patches"]

    if raw.shape[-1] != 2 * ps6.PATCH_CONTEXT_HALF_WIDTH:
        raise ValueError(
            f"{path} holds {raw.shape[-1]} px patches, expected "
            f"{2 * ps6.PATCH_CONTEXT_HALF_WIDTH}. Re-run the extractor."
        )

    return raw, raw, data["labels"].astype(np.int64), data["birds"]


def scoreable_mask(coordinates, labels, birds):
    """Which patches may be scored: everything except accepted model proposals.

    See ps6.model_proposed. Training uses every label; scoring cannot use the ones
    the previous model suggested without measuring itself. Negatives are always
    scoreable, so precision stays honest -- only positives are ever excluded.
    """
    first_pass = {}

    for bird in ps6.annotated_birds():
        backup = ps6.first_pass_path(bird["annotations"])
        first_pass[bird["name"]] = ps6.load_annotations(backup) if backup else None
        print(
            f"  {bird['name']:8s} unassisted labels: "
            f"{'NONE (assisted from the start)' if backup is None else len(first_pass[bird['name']])}"
        )

    return ~ps6.model_proposed(coordinates, labels, birds, first_pass)


def resolve_scoreable(path, data, labels, birds, coordinates, verified):
    """Which patches may be graded on, and the reasoning printed alongside.

    Split out of main() so an alternative architecture can be measured against exactly
    the same test set. This rule is the most easily got-wrong thing in the project --
    it is what separates an honest AP from one that grades the model on its own earlier
    proposals -- and a second copy of it would eventually drift from this one.
    """
    if verified:
        if "from_test_grid" in data.files and data["from_test_grid"].any():
            raise SystemExit(
                f"{path} contains test-grid patches. Re-run "
                f"`python training/extract_windows.py --train-only` -- training on the blind "
                f"grid would destroy the only independent measurement in the project."
            )

        if "source" in data.files:
            # The combined archive is only partly verified. Its window and blank-canvas
            # patches were clicked with no model output on screen, but its "archive"
            # positives came from whole-section passes that WERE model-assisted, so
            # scoring on them measures the previous model's decisions (see
            # ps6.model_proposed). Training uses all of them; scoring uses the blind
            # ones only.
            #
            # "blank" counts as blind: those sections were swept at proposal threshold
            # 1.01, so no proposal was ever displayed, let alone accepted. Omitting
            # them was a real measurement bug -- it left Gre595 and OR408 graded on a
            # single cell each, both reporting a meaningless AP=0.111 that dragged the
            # 7-bird mean to 0.471 while the blind grid showed the model IMPROVING.
            blind = np.isin(data["source"], ["window", "blank"])
            scoreable = blind.copy()

            # Negatives are always safe to score against: no model proposed them as
            # cells, so keeping them cannot flatter precision.
            scoreable |= labels == 0

            print(
                f"{(~blind & (labels == 1)).sum()} archive positives excluded "
                f"from scoring as possibly model-proposed;\n"
                f"{(blind & (labels == 1)).sum()} blind-clicked positives remain "
                f"scoreable. All {(labels == 1).sum()} are still used for training."
            )
        else:
            # Window-only archive: every label is a direct blind hand click.
            scoreable = np.ones(len(labels), bool)
            print("all labels scoreable: hand-clicked, no model proposals involved")
    else:
        print("\n=== separating scoreable labels from model-proposed ones ===")
        scoreable = scoreable_mask(coordinates, labels, birds)

        excluded = (labels == 1) & ~scoreable
        print(
            f"\n{excluded.sum()} positives excluded from scoring as model-proposed; "
            f"{((labels == 1) & scoreable).sum()} scoreable positives remain.\n"
            f"All {(labels == 1).sum()} are still used for training."
        )

    return scoreable


def archive_for(argv):
    """(patch archive, model path, verified) for the --windows / --combined flags."""
    windows = "--windows" in argv
    combined = "--combined" in argv

    if windows and combined:
        raise SystemExit("Pick one of --windows or --combined.")

    path = (
        COMBINED_PATCH_PATH if combined else WINDOW_PATCH_PATH if windows else PATCH_PATH
    )
    model_path = (
        "ps6_cnn_combined.pt"
        if combined
        else "ps6_cnn_windows.pt"
        if windows
        else MODEL_PATH
    )

    # Both of these archives carry only verified negatives, so neither needs the
    # model-proposed exclusion the whole-section archive does.
    return path, model_path, windows or combined


def main():
    path, model_path, verified = archive_for(sys.argv)

    _, patches, labels, birds = load_data(path)
    data = np.load(path)
    coordinates = data["coordinates"]
    device = pick_device()

    print(f"{path}: {len(patches)} patches, {(labels == 1).sum()} positive")

    scoreable = resolve_scoreable(path, data, labels, birds, coordinates, verified)

    bird_names = sorted(set(birds.tolist()))
    print(f"\nleave-one-bird-out over {bird_names}\n")

    results = {}
    all_label_results = {}

    for held_out in bird_names:
        is_test = birds == held_out

        print(
            f"=== hold out {held_out} "
            f"({(~is_test).sum()} train patches, "
            f"{(labels[~is_test] == 1).sum()} train cells / "
            f"{(labels[is_test] == 1).sum()} test cells, "
            f"{((labels[is_test] == 1) & scoreable[is_test]).sum()} scoreable) ==="
        )

        model = train(patches[~is_test], labels[~is_test], device)
        probabilities = predict(model, patches[is_test], device)

        # Headline: model-proposed positives dropped entirely from the test set.
        keep = scoreable[is_test]

        if (labels[is_test][keep] == 1).sum() == 0:
            # Nothing to grade against. AP over an empty positive set is undefined,
            # not zero -- averaging in a 0.000 would understate the model rather than
            # report a limitation. The two archives reach this state for opposite
            # reasons, so say which: the archive because every positive was a model
            # proposal, the windows because the bird genuinely had no cells clicked.
            reason = (
                "no cells clicked in its windows"
                if verified
                else "no unassisted labels"
            )
            print(f"  {held_out:8s} not scoreable ({reason})")
        else:
            results[held_out] = summarise(
                probabilities[keep], labels[is_test][keep], held_out
            )

        # Reference only, for comparison against earlier runs. Circular.
        all_label_results[held_out] = summarise(
            probabilities, labels[is_test], f"{held_out}*"
        )

    print("\n=== cross-bird summary (scoreable labels only -- the honest number) ===")
    for held_out, result in results.items():
        print(
            f"  {held_out:8s} AP={result['average_precision']:.3f} "
            f"F1={result['f1']:.3f} "
            f"precision={result['precision']:.1%} recall={result['recall']:.1%}"
        )

    mean_average_precision = float(
        np.mean([r["average_precision"] for r in results.values()])
    )
    print(
        f"\nmean AP across {len(results)} scoreable birds "
        f"({', '.join(results)}): {mean_average_precision:.3f}"
    )

    ungraded = [b for b in bird_names if b not in results]

    if ungraded:
        why = (
            "no cells were clicked in their windows"
            if verified
            else "their labels are entirely model-proposed"
        )
        print(
            f"  {', '.join(ungraded)} contributed training data but could not be "
            f"graded:\n  {why}, so they have no honest test set."
        )

    circular_mean = float(
        np.mean([r["average_precision"] for r in all_label_results.values()])
    )

    # With --windows there is nothing circular to contrast against: the only
    # difference is the ungraded birds averaged in as 0.000, which understates
    # rather than inflates. Printing it as "inflated" would be simply false.
    if verified:
        print(
            f"(counting the {len(ungraded)} ungraded bird(s) as 0.000 it would read "
            f"{circular_mean:.3f} -- understated, not comparable)"
        )
    else:
        print(
            f"(over all {len(all_label_results)} birds including model-proposed "
            f"positives it would read {circular_mean:.3f} -- inflated, do not cite)"
        )

    # Final model trained on every bird, for counting new sections. Its threshold
    # is the mean of the held-out thresholds; one tuned on training data would be
    # optimistic, since the model has already seen those cells. Averaged over the
    # scoreable birds only, since a threshold read off model-proposed labels would
    # be tuned to reproduce the previous model's decisions.
    print("\n=== final model, all birds ===")
    model = train(patches, labels, device)
    threshold = float(np.mean([r["threshold"] for r in results.values()]))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "cross_bird_results": results,
            "mean_average_precision": mean_average_precision,
            # Kept so a later run cannot mistake the inflated figure for the headline.
            "mean_average_precision_all_labels": circular_mean,
            "scoreable_positives": int(((labels == 1) & scoreable).sum()),
            "total_positives": int((labels == 1).sum()),
            "patch_archive": path,
        },
        model_path,
    )

    print(f"saved {model_path}, suggested threshold {threshold:.3f}")


if __name__ == "__main__":
    main()
