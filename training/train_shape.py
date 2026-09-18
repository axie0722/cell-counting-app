"""ShapeNet: find_band supplies the POSE, a small net supplies the SHAPE.

Alice, 2026-08-11: "is there a way to combine find_band with a model? i think the main problem is
the region outlines i feel like the general area its landing in is pretty good". Then: "finding
the outline of the ncm (like the border of the tissue) shouldn't be that hard for you to do
without a model".

Both were right, and both were measured before this file was written:

  * which_error.py replaced find_band's five ingredients with the truth one at a time. A PERFECT
    box -- true angle, border, depth, centre and length -- keeps only 74% of her NCM cells against
    the real finder's 83%. Placement is nearly solved; the BOX SHAPE is the binding constraint.
  * Of the NCM outline she draws, 63% runs within 120 um of the TISSUE EDGE and 30% along the band
    border, leaving only 11% genuinely free. For CMM it is 32% / 32% / 37% free. So NCM is almost
    entirely determined by two things I already have exactly, and CMM is not.
  * The NCM depth limit is therefore nearly redundant: removing it (depth 9999 um) gives
    84% x1.04 / 80% density against the current 1300 um's 83% x1.04 / 81%. The tissue edge was
    doing that job. And the end cuts trade 1:1 -- 1900->2200 um buys 8 points of cells and 11%
    more area, so density stays pinned near 80%. A straight end cut cannot do better.

WHAT THIS NET IS ALLOWED TO DO, and why that is small enough for 14 sections:

  1. It works in band_warp.py's frame, so there is NO ROTATION LEFT TO LEARN. That is what killed
     train_band.py, which collapsed to one constant vertical line for all 14 sections.
  2. It predicts a CORRECTION TO THE ATLAS, not a mask. The prior's log-odds are added to its
     output, so an untrained net IS the atlas (86% of NCM cells at x1.05 area) and can only earn
     credit by beating it.
  3. It is given the coordinates. A convolution cannot tell row 60 from row 200, and "how far
     across the band" is the most informative fact available, so across and along go in as
     channels.

HELD OUT BY BIRD, and the atlas prior is rebuilt per fold from the training birds only. Sections
of one bird are not independent (one-name-per-bird-not-per-section), and an atlas that has seen
the held-out section is the same circularity that inflated an AP by 0.2 in this project before.

Usage:
  python training/train_shape.py             # leave-one-bird-out, 7 folds, scored on CELLS
  python training/train_shape.py --epochs 40 # quicker
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
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

from band_warp import ACROSS_HIGH_UM, ALONG_HALF_UM, COLUMNS, FRAME_UM, ROWS, unwarp, warped_sections
from score_region_cells import cells_for
from score_rungs_cells import canvas_indices

# A section's bird. Sections of one bird share tissue, staining and mounting, so they must be
# held out together or the fold is measuring memorisation.
def bird(stem):
    return stem.split("_")[0]


# Wide enough to see a whole region (a 1300 um NCM is 65 px here), small enough that 13 training
# images cannot be memorised outright. Two downsamples give a 4x receptive-field reach.
CHANNELS = (16, 32, 64)

# Clamp the prior's log-odds so a bin no training bird ever called NCM is still recoverable. An
# infinite prior would be an unlearnable veto.
PRIOR_CLAMP = 3.0

# Train at half the warp's resolution: 40 um/px instead of 20. Not a compromise -- the band
# border can only be placed to about 92 um in the first place, so 40 um/px is already four times
# finer than the input it is built on, and it makes a 7-fold run 22 minutes instead of 1.5 hours
# on this laptop (no MPS, and the lambda box is unreachable). Regions are 1000+ um objects.
STRIDE = 2

# Only tissue pixels are scored -- the black surround is free marks and would swamp the loss.
CLASS_WEIGHTS = (1.0, 2.0, 3.0)   # background, NCM, CMM; CMM is the smallest and the weakest


class ShapeNet(nn.Module):
    """A small U-Net. Output is a CORRECTION added to the atlas's log-odds, not a mask."""

    def __init__(self, in_channels):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            )

        c1, c2, c3 = CHANNELS

        self.down1 = block(in_channels, c1)
        self.down2 = block(c1, c2)
        self.middle = block(c2, c3)
        self.up2 = block(c3 + c2, c2)
        self.up1 = block(c2 + c1, c1)

        # Start at ZERO so an untrained net is exactly the atlas. A random head would throw the
        # prior away on step one, which is the whole advantage being spent for nothing.
        self.head = nn.Conv2d(c1, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, prior):
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        m = self.middle(F.max_pool2d(d2, 2))

        u2 = self.up2(torch.cat([F.interpolate(m, d2.shape[-2:], mode="nearest"), d2], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, d1.shape[-2:], mode="nearest"), d1], 1))

        return self.head(u1) + prior


def shrink(array):
    """Drop to the training resolution. One place, so nothing can disagree about the grid."""
    return array[::STRIDE, ::STRIDE]


def shrink_max(array):
    """Same grid, but by MAXIMUM over each block instead of picking one pixel.

    For the dark-lamina channel only. Subsampling a thin line throws away every pixel it does not
    happen to land on, which is most of them -- the standard aliasing trap. Taking the block
    maximum keeps "there was a line somewhere in here", which is the whole content of the channel.
    Regions are 1000 um objects, so the 40 um of positional slop this costs is irrelevant.
    """
    rows = array.shape[0] // STRIDE * STRIDE
    columns = array.shape[1] // STRIDE * STRIDE
    blocks = array[:rows, :columns].reshape(
        rows // STRIDE, STRIDE, columns // STRIDE, STRIDE
    )
    out = blocks.max(axis=(1, 3))

    return out[: shrink(array).shape[0], : shrink(array).shape[1]]


def channel_count(section):
    """green, tissue, across, along -- plus the lamina channel if this run is using it.

    Read off the section dict rather than passed around, so the net's input width and the tensors
    actually built can never disagree.
    """
    return 4 + int("small_valley" in section)


def grow(array):
    """Back to the warp's resolution, for unwarping. Nearest-neighbour, as everywhere here."""
    out = np.repeat(np.repeat(array, STRIDE, axis=0), STRIDE, axis=1)

    return out[:ROWS, :COLUMNS]


def coordinate_channels():
    """across and along in microns, scaled to roughly +-1. The net's sense of place."""
    across = (ACROSS_HIGH_UM - (np.arange(ROWS) + 0.5) * FRAME_UM) / 2000.0
    along = (-ALONG_HALF_UM + (np.arange(COLUMNS) + 0.5) * FRAME_UM) / 2000.0

    return (
        shrink(np.broadcast_to(across[:, None], (ROWS, COLUMNS))),
        shrink(np.broadcast_to(along[None, :], (ROWS, COLUMNS))),
    )


def prior_from(sections):
    """The atlas, rebuilt in the warped frame from these sections only.

    Same idea as band_atlas.atlas_from, but at frame resolution so it lines up with the net's
    output pixel for pixel. Fraction of sections WITH TISSUE at a pixel that called it each class.
    """
    shape = shrink(np.zeros((ROWS, COLUMNS))).shape
    total = np.zeros(shape)
    hits = np.zeros((3,) + shape)

    for section in sections:
        tissue = section["small_tissue"]
        total += tissue

        for value in (0, 1, 2):
            hits[value] += tissue & (section["small_labels"] == value)

    share = np.divide(hits, total, out=np.full_like(hits, 1 / 3), where=total > 0)
    share = np.clip(share, 1e-3, 1 - 1e-3)

    return np.clip(np.log(share), -PRIOR_CLAMP, PRIOR_CLAMP)


def tensors(section, across, along, device):
    tissue = section["small_tissue"]

    layers = [
        section["small_green"],
        tissue.astype(np.float32),
        across * tissue,
        along * tissue,
    ]

    if "small_valley" in section:
        layers.append(section["small_valley"] * tissue)

    stack = np.stack(layers).astype(np.float32)

    return (
        torch.from_numpy(stack)[None].to(device),
        torch.from_numpy(section["small_labels"].astype(np.int64))[None].to(device),
        torch.from_numpy(tissue)[None].to(device),
    )


def jitter(x, y, tissue, generator):
    """Mild augmentation only. The pose is already canonical, so big transforms would teach the
    net to undo an alignment it will never be given at test time. Shifts are +-100 um, rotation
    +-8 deg -- roughly the measured band-angle error, so the net sees the noise it will face."""
    degrees = (torch.rand(1, generator=generator).item() * 2 - 1) * 8.0
    shift_y = (torch.rand(1, generator=generator).item() * 2 - 1) * 0.03
    shift_x = (torch.rand(1, generator=generator).item() * 2 - 1) * 0.03
    scale = 1.0 + (torch.rand(1, generator=generator).item() * 2 - 1) * 0.08

    radians = np.radians(degrees)
    cos, sin = np.cos(radians) / scale, np.sin(radians) / scale

    theta = torch.tensor(
        [[[cos, -sin, shift_x], [sin, cos, shift_y]]], dtype=torch.float32, device=x.device
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)

    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    labels = F.grid_sample(
        y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False
    )[:, 0].long()

    keep = F.grid_sample(
        tissue[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False
    )[:, 0] > 0.5

    # Brightness jitter: stain intensity varies between birds and is not a shape cue.
    x[:, 0] = x[:, 0] * (0.85 + 0.3 * torch.rand(1, generator=generator).item())

    return x, labels, keep


def train_fold(train, prior, across, along, epochs, device, seed):
    """One net. Called once per ensemble member; see train_ensemble."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    net = ShapeNet(channel_count(train[0])).to(device)
    prior_tensor = torch.from_numpy(prior.astype(np.float32))[None].to(device)
    weights = torch.tensor(CLASS_WEIGHTS, device=device)

    optimiser = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = [tensors(s, across, along, device) for s in train]

    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(batch), generator=generator)

        for index in order:
            x, y, tissue = batch[index]
            x, y, tissue = jitter(x, y, tissue, generator)

            logits = net(x, prior_tensor)
            loss = F.cross_entropy(logits, y, weight=weights, reduction="none")
            loss = (loss * tissue).sum() / tissue.sum().clamp(min=1)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    return net, prior_tensor


def train_ensemble(train, prior, across, along, epochs, device, seed, members):
    """Several nets per fold, differing only in seed.

    WHY: a single net's CMM density ranged 68-83% across three seeds against the box's 73%, so one
    net's number is not a measurement. Averaging is the standard fix for a high-variance model on
    small data, and it also gives an uncertainty signal for free -- how much the members DISAGREE
    on a section, which is the honest way to decide when to fall back rather than a hand-picked
    geometry threshold.
    """
    return [
        train_fold(train, prior, across, along, epochs, device, seed + member)
        for member in range(members)
    ]


def masks_from(logits, section):
    from scipy.ndimage import binary_fill_holes

    from ceiling_lines import largest_blob

    classes = logits.argmax(1)[0].cpu().numpy()
    out = {}

    for name, value in (("NCM", 1), ("CMM", 2)):
        frame_mask = grow(classes == value) & section["tissue"]
        canvas = unwarp(frame_mask, section["band"], section["shape"], section["mirror"])
        out[name] = largest_blob(binary_fill_holes(canvas & section["canvas_tissue"]))

    return out


def predict_ensemble(members, section, across, along, device):
    """Average the members' PROBABILITIES, and report how much they disagreed.

    Averaging probabilities rather than masks means a pixel two members call NCM weakly and one
    calls CMM strongly can still go to CMM -- majority voting on masks throws that away.
    """
    x, _, _ = tensors(section, across, along, device)

    probabilities = None
    individual = []

    for net, prior_tensor in members:
        net.eval()
        with torch.no_grad():
            logits = net(x, prior_tensor)

        probability = F.softmax(logits, dim=1)
        probabilities = probability if probabilities is None else probabilities + probability
        individual.append(masks_from(logits, section))

    found = masks_from(torch.log(probabilities / len(members)), section)

    agreement = {}

    for name in ("NCM", "CMM"):
        scores = []
        for a in range(len(individual)):
            for b in range(a + 1, len(individual)):
                first, second = individual[a][name], individual[b][name]
                scores.append(
                    (first & second).sum() / max((first | second).sum(), 1)
                )
        agreement[name] = float(np.mean(scores)) if scores else 1.0

    return found, agreement


def predict(net, section, prior_tensor, across, along, device):
    """The net's regions, unwarped back onto the section, cleaned like find_band's are."""
    from scipy.ndimage import binary_fill_holes

    from ceiling_lines import largest_blob

    net.eval()
    x, _, _ = tensors(section, across, along, device)

    with torch.no_grad():
        classes = net(x, prior_tensor).argmax(1)[0].cpu().numpy()

    out = {}

    for name, value in (("NCM", 1), ("CMM", 2)):
        frame_mask = grow(classes == value) & section["tissue"]
        canvas = unwarp(frame_mask, section["band"], section["shape"], section["mirror"])
        out[name] = largest_blob(binary_fill_holes(canvas & section["canvas_tissue"]))

    return out


def atlas_only(section, prior, across, along):
    """The prior's own answer, unwarped -- the number the net has to beat."""
    from scipy.ndimage import binary_fill_holes

    from ceiling_lines import largest_blob

    classes = prior.argmax(0)
    out = {}

    for name, value in (("NCM", 1), ("CMM", 2)):
        frame_mask = grow(classes == value) & section["tissue"]
        canvas = unwarp(frame_mask, section["band"], section["shape"], section["mirror"])
        out[name] = largest_blob(binary_fill_holes(canvas & section["canvas_tissue"]))

    return out


def tally_into(tally, section, found, cells):
    ys, xs = cells

    for name, value in (("NCM", 1), ("CMM", 2)):
        drawn = section["canvas_labels"] == value

        tally[name][2] += int(found[name].sum())
        tally[name][3] += int(drawn.sum())

        if len(ys):
            inside = drawn[ys, xs]
            tally[name][0] += int(inside.sum())
            tally[name][1] += int((inside & found[name][ys, xs]).sum())


def line(label, tally):
    parts = []

    for name in ("NCM", "CMM"):
        total, kept, area, drawn = tally[name]
        fraction = kept / max(total, 1)
        ratio = area / max(drawn, 1)
        parts.append(f"{fraction:5.0%} x{ratio:4.2f} {fraction / max(ratio, 1e-9):5.0%}")

    return f"  {label:26s}" + "    ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="npz of held-out masks, for review_shape.py")
    parser.add_argument("--members", type=int, default=3, help="nets per fold")
    parser.add_argument(
        "--valley",
        action="store_true",
        help="add the dark-lamina channel (band_warp.valley_map)",
    )
    args = parser.parse_args()

    device = "cpu"
    sections = warped_sections()

    for section in sections:
        section["small_green"] = shrink(section["green"])
        section["small_tissue"] = shrink(section["tissue"])
        section["small_labels"] = shrink(section["labels"])

        if args.valley:
            section["small_valley"] = shrink_max(section["valley"])

    across, along = coordinate_channels()

    birds = sorted({bird(s["stem"]) for s in sections})
    print(f"{len(sections)} sections, {len(birds)} birds: {', '.join(birds)}")
    print(f"leave-one-bird-out, {args.epochs} epochs per fold")
    print(f"{channel_count(sections[0])} input channels"
          f"{' (green, tissue, across, along, lamina)' if args.valley else ''}\n")

    cells = {}
    for section in sections:
        points, _ = cells_for(section["image"])
        cells[section["stem"]] = (
            canvas_indices(points, section["pixel_size"], section["shape"])
            if len(points)
            else (np.array([], int), np.array([], int))
        )

    saved = {}
    agreements = {}
    net_tally = {"NCM": [0, 0, 0, 0], "CMM": [0, 0, 0, 0]}
    atlas_tally = {"NCM": [0, 0, 0, 0], "CMM": [0, 0, 0, 0]}

    for held in birds:
        train = [s for s in sections if bird(s["stem"]) != held]
        test = [s for s in sections if bird(s["stem"]) == held]

        prior = prior_from(train)
        members = train_ensemble(
            train, prior, across, along, args.epochs, device, args.seed * 100, args.members
        )

        for section in test:
            found, agreement = predict_ensemble(members, section, across, along, device)
            agreements[section["stem"]] = agreement
            tally_into(net_tally, section, found, cells[section["stem"]])
            tally_into(atlas_tally, section, atlas_only(section, prior, across, along), cells[section["stem"]])

            if args.save:
                saved[section["stem"] + "|NCM"] = found["NCM"]
                saved[section["stem"] + "|CMM"] = found["CMM"]

        print(f"  fold {held:8s} trained on {len(train)}, tested on {len(test)}")

    print("\n" + " " * 28 + "NCM  kept  area  dens        CMM  kept  area  dens")
    print(line("ShapeNet (held out)", net_tally))
    print(line("atlas alone (held out)", atlas_tally))
    print("  find_band boxes             83% x1.03   81%       73% x1.00   73%")

    # DOES DISAGREEMENT FLAG THE COLLAPSES? A single net kept only 10% of Gre595_RH's NCM cells
    # (box: 55%) and predicted almost nothing on YW113_RH_1-1-2. If those two show the lowest
    # member agreement, agreement is a usable fall-back trigger and no geometry threshold is
    # needed. If they do not, the collapses are confident and a different guard is required.
    print("\nmember agreement per section (low = the nets disagree = suspect):")
    for stem in sorted(agreements, key=lambda s: min(agreements[s].values())):
        scores = agreements[stem]
        print(f"  {stem[:34]:36s} NCM {scores['NCM']:.2f}   CMM {scores['CMM']:.2f}")

    if args.save:
        np.savez_compressed(args.save, **saved)
        print(f"\nwrote {args.save} -- held-out masks, one per section, for review_shape.py")


if __name__ == "__main__":
    main()
