"""Does a pretrained backbone recall more cells than the small CNN trained from scratch?

The recall gap is the open problem: 67.9% on trained tissue, 55.6% on an unseen
hemisphere, and the cells being missed are not a fixable blind spot -- Alice reviewed
the four the classifier was most confident about (2026-08-09) and they are all blurry,
ambiguous objects. So the remaining route is a model that is simply better, not a patch
to one failure mode.

SmallCNN learns everything from ~420 positives starting from random weights, which
means those labels have to teach it what a blob is before they can teach it which blob
is a PS6 cell. A backbone pretrained on ImageNet arrives already knowing edges, blobs
and textures, so the labels only pay for the second half. That is the standard reason
transfer learning wins on small datasets, and this dataset is small.

It might well lose. 11M parameters on 420 positives is exactly the overfitting risk
the SmallCNN docstring warns about, and ImageNet is natural photographs, not
single-channel fluorescence. Both outcomes are worth having on record.

Three adaptations, none of them optional:

  1 channel     the pretrained first conv expects RGB. Its three input weights are
                SUMMED into one channel rather than tiling the patch three times --
                identical arithmetic for a grey image, a third of the work.
  64x64 input   ResNet's stem (stride-2 7x7 conv, then maxpool) is built for 224x224
                and would leave a 2x2 feature map. Replaced with a stride-1 3x3 conv,
                the standard small-image fix, keeping every pretrained residual block.
  learning rate the pretrained layers get 10x less than the fresh head, so 420
                positives cannot wash out the features being borrowed.

Everything else is imported from train_cnn deliberately -- the same seed, the same
negative resampling, the same augmentation, the same leave-one-bird-out split, and the
same scoreable-label rule. The architecture is the only variable, which is the only way
the comparison means anything.

Usage:
  python train_backbone.py --combined          # the archive the shipped model uses
  python train_backbone.py --combined --epochs 15
  python train_backbone.py --combined --mine   # half the negatives mined, to
                                               # ps6_cnn_backbone_mined.pt
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

import ps6
from train_cnn import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    archive_for,
    load_data,
    pick_device,
    predict,
    resolve_scoreable,
    summarise,
    train,
)

MODEL_PATH = "ps6_cnn_backbone.pt"

# --mine writes here instead. A separate path on purpose: ps6_cnn_backbone.pt is what
# count_cells.py loads (see its MODEL_PATH), so an experiment that overwrote it would
# change every count produced afterwards before anyone had compared the two on the blind
# grid. Nothing reads these until score_blind_grid.py is pointed at them by hand.
MINED_MODEL_PATH = "ps6_cnn_backbone_mined.pt"

# Each fold's result is written here the moment that fold finishes, and folds already in
# the file are skipped on restart. Not a nicety: this run takes ~4 hours PER FOLD on this
# CPU, and Alice's machine crashed 4 hours into fold 1 of 8 on 2026-08-09, losing all of
# it because nothing was saved until every fold was done. A long job with no intermediate
# state cannot survive an interruption, and on an 8-fold ResNet run interruptions are the
# expected case, not the exception.
PROGRESS_PATH = "grids/backbone folds.json"
MINED_PROGRESS_PATH = "grids/backbone folds mined.json"

# PyTorch's default thread count left the first run at ~118% CPU on an 8-core machine.
# 6 leaves cores for the region training and for Alice's own work; threading changes only
# how the same arithmetic is scheduled, not the seed, the negative resampling or the
# result.
THREADS = 6

# Fewer epochs than the 40 SmallCNN uses. A pretrained network starts near a good
# solution and mostly needs to be pointed at this task; running it as long as a
# from-scratch model mainly gives it time to memorise 420 positives.
EPOCHS = 15

# The pretrained layers move at a tenth of the head's rate. Without this the first few
# batches of a randomly-initialised head send large gradients back through the features
# and destroy the pretraining before it can be used -- the most common way transfer
# learning quietly turns into from-scratch training with extra steps.
BACKBONE_LEARNING_RATE_FACTOR = 0.1


class BackboneCNN(nn.Module):
    """ResNet-18, pretrained, adapted to 64x64 patches.

    `in_channels` defaults to 1, which is the cell detector's single greyscale patch. Passing 2 is
    used by train_edge for a two-scale patch (fine detail plus a downsampled wide surround); the
    seeding below splits the pretrained weights across the channels so the total response magnitude
    is unchanged, for the same batch-norm reason as folding RGB.
    """

    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        # Fold RGB into one channel. Summing (not averaging) keeps the response
        # magnitude a grey image would have produced by passing through all three
        # planes, so the pretrained batch-norm statistics downstream stay valid.
        first = net.conv1.weight.detach()
        grey = first.sum(dim=1, keepdim=True)

        # The stem is replaced, so the pretrained 7x7 kernel cannot be reused directly.
        # Its channel-summed weights still seed the new 3x3 by centre-cropping, which
        # starts the stem as a smaller version of the edge detectors ImageNet learned
        # rather than from noise.
        stem = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)

        if pretrained:
            with torch.no_grad():
                seed = grey[:, :, 2:5, 2:5] / in_channels
                stem.weight.copy_(seed.repeat(1, in_channels, 1, 1))

        net.conv1 = stem

        # Drop the maxpool too. Stem and maxpool together downsample by 4 before the
        # residual blocks even start, which on a 64 px patch throws away the fine
        # structure that distinguishes a PS6 annulus from a bright smudge -- the exact
        # distinction the blurry rejected cells turn on.
        net.maxpool = nn.Identity()

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(net.fc.in_features, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

    def parameter_groups(self):
        """Pretrained features slow, fresh head at full rate."""
        return [
            {
                "params": self.features.parameters(),
                "lr": LEARNING_RATE * BACKBONE_LEARNING_RATE_FACTOR,
            },
            {"params": self.classifier.parameters(), "lr": LEARNING_RATE},
        ]


def make_optimiser(model):
    return torch.optim.AdamW(model.parameter_groups(), weight_decay=WEIGHT_DECAY)


def load_progress(epochs, path):
    """Folds already finished, or {} if the file is absent or from a different run.

    The epoch count is stored alongside, because resuming a 15-epoch run into a 40-epoch
    one would silently average folds trained for different lengths and report the mean as
    if it were one model.
    """
    if not Path(path).exists():
        return {}

    saved = json.loads(Path(path).read_text())

    if saved.get("epochs") != epochs:
        print(
            f"  {path} is from a {saved.get('epochs')}-epoch run, not {epochs}; "
            f"starting over"
        )

        return {}

    done = saved.get("results", {})

    if done:
        print(f"  resuming: {len(done)} folds already done ({', '.join(done)})")

    return done


def save_progress(results, epochs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"epochs": epochs, "results": results}, indent=1))


def main():
    epochs = next(
        (
            int(sys.argv[i + 1])
            for i, a in enumerate(sys.argv)
            if a == "--epochs" and i + 1 < len(sys.argv)
        ),
        EPOCHS,
    )

    mine = "--mine" in sys.argv
    model_path = MINED_MODEL_PATH if mine else MODEL_PATH
    progress_path = MINED_PROGRESS_PATH if mine else PROGRESS_PATH

    torch.set_num_threads(THREADS)

    path, _, verified = archive_for(sys.argv)

    _, patches, labels, birds = load_data(path)
    data = np.load(path)
    device = pick_device()

    print(f"{path}: {len(patches)} patches, {(labels == 1).sum()} positive")

    scoreable = resolve_scoreable(
        path, data, labels, birds, data["coordinates"], verified
    )

    reference = BackboneCNN(pretrained=False)
    print(
        f"\nResNet-18 backbone: "
        f"{sum(p.numel() for p in reference.parameters()):,} parameters "
        f"(SmallCNN has 68,209), {epochs} epochs"
    )
    del reference

    print(
        f"negatives: {'HALF MINED (--mine)' if mine else 'uniform, the shipped setting'}"
        f"\nwriting to: {model_path}, folds to {progress_path}"
    )

    bird_names = sorted(set(birds.tolist()))
    print(f"\nleave-one-bird-out over {bird_names}\n")

    results = load_progress(epochs, progress_path)

    for held_out in bird_names:
        if held_out in results:
            print(f"=== hold out {held_out}: already done, skipping ===", flush=True)
            continue

        is_test = birds == held_out
        keep = scoreable[is_test]

        started = time.time()
        print(
            f"=== hold out {held_out} "
            f"({(~is_test).sum()} train patches, "
            f"{(labels[~is_test] == 1).sum()} train cells / "
            f"{(labels[is_test] == 1).sum()} test cells, "
            f"{((labels[is_test] == 1) & keep).sum()} scoreable) ===",
            flush=True,
        )

        model = train(
            patches[~is_test],
            labels[~is_test],
            device,
            epochs=epochs,
            make_model=BackboneCNN,
            make_optimiser=make_optimiser,
            mine_hard_negatives=mine,
        )
        probabilities = predict(model, patches[is_test], device)
        del model

        if (labels[is_test][keep] == 1).sum() == 0:
            print(f"  {held_out:8s} not scoreable (no blind-clicked cells)")
            continue

        results[held_out] = summarise(
            probabilities[keep], labels[is_test][keep], held_out
        )

        # Written now, not at the end. See PROGRESS_PATH.
        save_progress(results, epochs, progress_path)
        print(f"  ({(time.time() - started) / 60:.0f} min, saved)", flush=True)

    print("\n=== cross-bird summary (scoreable labels only) ===")
    for held_out, result in results.items():
        print(
            f"  {held_out:8s} AP={result['average_precision']:.3f} "
            f"F1={result['f1']:.3f} "
            f"precision={result['precision']:.1%} recall={result['recall']:.1%}"
        )

    mean_average_precision = float(
        np.mean([r["average_precision"] for r in results.values()])
    )

    # The number to beat. Anything inside a few hundredths of it is noise on test sets
    # of 8-34 cells, and the honest conclusion in that case is "no difference", not a
    # winner -- stain augmentation moved this figure 0.653 -> 0.670 and still lost on
    # the blind grid.
    print(
        f"\nmean AP across {len(results)} scoreable birds "
        f"({', '.join(results)}): {mean_average_precision:.3f}\n"
        f"  SmallCNN on this archive: 0.653\n"
        f"  a difference under ~0.03 here is noise; the blind grid decides."
    )

    print("\n=== final model, all birds ===", flush=True)
    model = train(
        patches,
        labels,
        device,
        epochs=epochs,
        make_model=BackboneCNN,
        make_optimiser=make_optimiser,
        mine_hard_negatives=mine,
    )
    threshold = float(np.mean([r["threshold"] for r in results.values()]))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "architecture": "BackboneCNN",
            "cross_bird_results": results,
            "mean_average_precision": mean_average_precision,
            "epochs": epochs,
            "patch_archive": path,
            "hard_negative_mining": mine,
        },
        model_path,
    )

    print(f"saved {model_path}, suggested threshold {threshold:.3f}")
    print(
        "\nThis is NOT comparable to the shipped model yet. Cross-bird AP is measured\n"
        "on model-assisted labels and small test sets; the blind grid is the only\n"
        "honest comparison, and it must be read at MATCHED RECALL:\n"
        f"  python score_blind_grid.py {model_path}"
    )


if __name__ == "__main__":
    main()
