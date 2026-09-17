# A map of the code

This is a reading guide to the files, one level below the summary in `README.md`. It exists because
the app is a flat set of modules that import each other by bare name, and a flat list of sixty files
tells you nothing about which calls which. The `src/` folders group them by job; this file says what
each group is and where to start.

## How it is put together, and why the layout is what it is

Every module imports its neighbours by bare name -- `import ps6`, `from app_state import ...`. That
only works when the folder holding them is on Python's import path, and Python puts just one folder
there on its own: the directory of the script it was told to run. So the code cannot simply be
scattered into subfolders and left; something has to add those subfolders back to the path.

That something is **`_bootstrap.py`**. Each of the scripts you can start -- `app.py` and the four it
launches as separate processes -- imports it before its first local import, and it adds the five
`src/` folders to `sys.path`. After that the bare imports resolve exactly as they did when every file
sat in one directory. No import statement had to change to move the files; read `_bootstrap.py` for
the full reasoning.

**The root holds only the scripts that start a process.** There are five: `app.py`, and the four it
launches -- `count_cells.py`, `draw_regions.py`, `show_counts.py`, `app_table.py`. `app.py` runs each
of the other four as its own subprocess, by absolute path, so a crash in a counting run or a drawing
window cannot take the main window down with it. Everything under `src/` is a library one of these
five imports; nothing under `src/` is started directly.

## The folders

### `src/app/` — the window and what it does
The application layer that `app.py` sits on top of. The job queue and the worker that drives it
(`app_jobs`), pausing a count while a drawing window is open (`app_pause`), keeping the machine awake
during a run (`app_awake`), finding which process is counting what (`app_running`), the folder the
sections live in and the per-section status (`app_folder`, `app_state`), the queue you can save and
reload (`app_queue`, `app_saved_queue`), building the command lines for the four subprocesses
(`app_actions`), and the two data views (`app_all_data`, `app_export`).

### `src/model/` — the detector
The trained cell detector and everything that defines its shape. `ps6` is the wrapper the rest of the
app calls to run it. The `train_*` files hold the network definitions themselves — `train_cnn` and
`train_backbone` are imported at counting time for the model classes, not only for training; the
others (`train_shape`, `train_band`, `train_boxes`, `train_regions`) define the region models reused
the same way. The trained weights live at the repository root, beside `count_cells.py`, which loads
them.

### `src/regions/` — proposing the outlines
Turning a section into a first guess at the NCM and CMM outlines, which you then correct in the
drawing window: `predict_regions` (which reads `region_constants.json`, kept in this folder beside
it), `prepare_regions`, `region_boxes`, and the shared `draw_helpers`.

### `src/boundaries/` — finding the borders
The geometry engine underneath the region proposals: the rules that trace a region border across the
tissue (`rule_*`), the band and atlas machinery they run on (`band_*`, `find_band`, `field_l_band`,
`ceiling_lines`), the polygon and scan helpers (`polygon_solid`, `scan_*`), scoring
(`score_*`), the hippocampus-side check (`hippocampus_check`), and the border diagnostics
(`diag_*`, `show_border_kinds`).

### `src/review/` — looking at the result
The napari sheets that draw a finished or in-progress result back onto the image so it can be checked
by eye: `review_regions`, `review_predictions`, `review_polygon`, and the rest of the `review_*`
family.

## Running it

Unchanged by this layout — see `README.md` (Quick start) and `INSTALL.md`. In short: double-click
`Count cells.command` (macOS) or `Count cells.bat` (Windows), or run `bash count-cells.sh` (Linux).
The first run builds the environment; after that it starts in about a second.

## A caveat worth repeating

This grouping lives in this repository only. Upstream, `package_app.py` regenerates the package by
following the imports out of `app.py` and copying the files it finds into one flat folder, so a
rebuild flattens `src/` back out. The folders, `_bootstrap.py`, and this file are therefore
maintained here, not inherited from upstream.
