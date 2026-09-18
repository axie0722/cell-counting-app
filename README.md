# Cell counting

Counts PS6-labelled cells in NCM and CMM on zebra finch brain sections, and reports a density for
each region. You outline the two regions on a section — the app proposes the outlines and you correct
them — and it does the counting.

**→ [INSTALL.md](INSTALL.md) is the instruction manual.** Read that first. It covers installing,
running, what the file names have to say, and what the numbers mean.

## Licence — read before you use this

`ps6_cnn_backbone.pt` is the trained detector, and it is built on **ImageNet pretrained weights,
which are licensed for NON-COMMERCIAL RESEARCH USE ONLY.** That restriction travels with the file: it
applies to this copy, to any copy you make, and to anything you use it for. Research and teaching are
fine. Anything commercial is not permitted by that licence. A commercially usable version would need
the backbone retrained from scratch.

No licence has been chosen for the rest of the code yet, which by default means all rights reserved.
Ask before redistributing it.

## Quick start

**One command sets everything up.** You need **one** of these on the machine first — either is
enough, and the launcher does the rest:

* **Python 3.11, 3.12 or 3.13** — not the newest version, because PyTorch has no downloads ready for
  a brand new Python for some months after it appears; or
* **[uv](https://docs.astral.sh/uv/)** — if you have uv but no suitable Python, the launcher uses it
  to fetch a private Python of its own. This is the zero-Python path, and it needs no admin.

Everything the app installs — a private Python if uv fetches one, and about 1 GB of packages — lands
**inside this folder** (`.venv`) or in uv's own cache in your home folder. Nothing is installed onto
the system, and deleting the folder removes it all.

1. Download this repository (green **Code** button → **Download ZIP**) and unzip it, or clone it.
2. Start the launcher — that single step installs and runs everything:
   * **macOS:** double-click **`Count cells.command`** (or, in a terminal in the folder,
     `./"Count cells.command"`).
   * **Windows:** double-click **`Count cells.bat`**.
   * **Linux:** `bash count-cells.sh`.
3. The first start builds the app its own Python environment and downloads about 1 GB, most of it
   PyTorch. Measured on a fast connection with nothing cached: two minutes, plus half a minute for
   the window to appear. On a slow connection, budget 15–20 minutes for the same gigabyte. **It
   happens once** — after that, starting takes about a second.
4. In the app, press **Choose folder** and point it at your sections.

Your operating system will warn you that this is from an unidentified developer — that is about the
app not being signed with a paid developer account, not about anything being wrong with it.
INSTALL.md says what to click.

If you downloaded the ZIP and double-clicking `Count cells.command` does nothing, the file lost its
executable flag on the way. Open a Terminal in the folder and run `bash count-cells.sh` instead.

## What the numbers are, in one paragraph

It under-counts, and it is measured: on tissue from a hemisphere it was trained on it finds about
**66%** of cells, and about **86%** of what it reports is really a cell; on a hemisphere it has never
seen, about **57%** and **90%**. It misses cells rather than inventing them. So **compare densities
within a study; do not quote a raw count as truth**, and look at your sections in Review cells before
using the numbers. The full version of this, including what cannot be fixed by any setting, is
section 5 of INSTALL.md.

## What is in here

The program and the trained model. **No images, no counts, no outlines, no research data.**

The code is laid out like this. You launch the app from the root; everything it imports lives under
`src/`:

```
Count cells.command / .bat        the launchers you double-click (see the Quick start above)
count-cells.sh                    what those launchers run: builds the environment, starts the app

app.py                            the entry point — the window and everything it does
count_cells.py                    the counting run, started as its own process per section
draw_regions.py                   the napari window for outlining NCM and CMM
show_counts.py                    the napari window for reviewing counted cells
app_table.py                      the "View all data" table
_bootstrap.py                     puts the src/ folders below on the import path (see the file)

src/app/                          the window's own logic: the queue, jobs, pausing, export, state
src/model/                        the detector — network definitions (train_*) and the ps6 wrapper
src/regions/                      proposing and preparing the NCM/CMM outlines, plus region_constants.json
src/boundaries/                   the geometry that finds the region borders (rule_*, band_*, scan_*, …)
src/review/                       the review and diagnostic sheets (review_*)

ps6_cnn_backbone.pt, ps6_cnn.pt   the trained detector
```

The five scripts at the root are the ones that start their own process — the app, and the four
windows and jobs it launches. Everything else is a library one of them imports. `docs/code-map.md`
walks through it in more detail.

These files are generated, not edited here: the app is developed in a larger research folder, and
`package_app.py` there computes which files it actually needs by following the imports out of
`app.py`, then copies those and the model into this repository. Fixes belong upstream in that folder;
a change made here is lost at the next rebuild. `README.md`, `.gitignore` and `LICENSE` are the
exception — the rebuild leaves those alone. **The `src/` grouping above is applied in this
repository, not upstream: a rebuild flattens the files back into one folder, so the folders and
`_bootstrap.py` are re-created here rather than inherited.**
