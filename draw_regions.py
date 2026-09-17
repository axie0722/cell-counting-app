"""Draw NCM and CMM on a section, and mark which side is dorsal.

You draw two polygons. dNCM and vNCM are not drawn -- the counter derives them by
cutting NCM in half, so the subdivision comes from a stated rule applied the same
way in every section instead of from where a hand happened to fall.

THAT CUT IS NOW ON SCREEN, as a green line across NCM that follows the outline as
you change it (see update_split). The reason to look at it: NCM is never counted as
a whole, dNCM and vNCM are, and where the cut lands depends on where your outline
ENDS along Field L. Two outlines that look equally good can report different dNCM
densities. The half nearer the yellow dorsal marker is dNCM.

The dorsal marker is one click on the hippocampus side. Dorsal is anatomical, not
a fixed image direction, so this keeps the split correct even when a section is
mounted at an angle.

  polygon tool  draw the outlines
  Ctrl-S        save  (plain "s" is a Shapes-layer shortcut and never arrives)
  Ctrl-P        predict NCM and CMM again
  Ctrl-F        flip which side of Field L is NCM, when there is no dorsal marker to say
  Ctrl-R        clear the layer you are on, to redraw it from scratch
  1 / 2         jump to the NCM / CMM layer
  4             jump to the dorsal marker
  6             jump to the Field L ends layer -- TWO clicks, one at each end of the line
  5             jump to the HP layer, but only with --hp (see below)

IT OPENS WITH THE RULE'S PREDICTION ALREADY DRAWN, Alice 2026-08-19: *"when the user presses draw outlines you
open these predicted outlines, but also give them the option to edit/redraw them if they are bad"*. Drag the
vertices that are wrong, or Ctrl-R and draw over it. Two things about that:

  * THEY APPEAR WITH NO CLICK AT ALL, on a GUESSED side, and Ctrl-F flips it. Alice, 2026-08-19: *"can you
    activate the annotaions without the dorsal click?"*. The rule works in the Field L band's frame and needs
    one bit -- which side of the band is NCM -- which the image does not carry: no cue beat answering "the same
    side" every time, on 15 sections and 7 birds ([[diag_dorsal_bit]]). So it answers the same side every time,
    SAYS it guessed, and hands you the flip: cyan sitting on CMM is unmistakable, so your eye settles it in one
    keypress. Placing the marker (press 4) re-predicts from the real answer and takes over from the guess.
    Alice, when this waited for a keypress instead: *"i dont see the predicted outlines when i press draw
    outlines?"* -- and the only place that said what to do was a terminal the app hides.
  * THE MARKER IS STILL REQUIRED, for the other bit. Which END is dorsal is what cuts NCM into dNCM and vNCM,
    the guess there is near chance, and no outline on screen would reveal it. So counting still needs the
    click even when the outlines look right.
  * A PREDICTION IS NOT A LABEL. Anything still exactly as proposed is saved with `source = model`, and
    `review_regions.sections()` -- the corpus every fit and grade is built from -- drops those rows. Move one
    vertex and that polygon becomes `hand` and counts as yours. Without this the next model would be fitted on
    its own output ([[model-proposed-labels-are-circular]]).

HOW GOOD IS IT? Leave-one-bird-out, NCM averages about 0.81 IoU against your own outlines and CMM about 0.53.
So NCM usually needs a nudge and CMM usually needs work, mostly its far border away from Field L.

THE HIPPOCAMPUS LAYER IS OFF UNLESS YOU ASK FOR IT. Alice, 2026-08-20: *"can you also remove
the hp annotation option in napari when i go to draw outlines"*. It is a label for the
far-border problem and is never used for counting, so it does not belong on the window
opened to outline NCM and CMM -- one more layer to land on by accident. `--hp` puts it back,
and draw_hp_queue.py passes that itself, so the tool that collected the 14 outlines still
works.

Outlines already drawn are SAFE either way: a hippocampus polygon in the file is written back
untouched when this saves, even though nothing on screen shows it. See `carried` below -- the
alternative is a Ctrl-S that quietly deletes a label.

THE FIELD L ENDS ARE TWO CLICKS, and they are the highest-value thing to add right now. The counter
walks the Field L line outward until it leaves the tissue or hits a visible separation, and on about a
third of sections it walks too far -- past the end of NCM into whatever is next, the hippocampus on
YW113 1-1-2. Until now the target for "where should it stop" was INFERRED from where the NCM polygon's
long side happened to end, which is not the same thing. Click the two ends and it becomes a measured
quantity. Two clicks per section, and it does not have to be exact.

One image per run: close the window, run it again for the next section.

Usage:
  python draw_regions.py                     # first bird in ps6.BIRDS
  python draw_regions.py "Gre595_....TIF"    # a specific image
  python draw_regions.py Gre595_RH_3-1-4     # any unique fragment, either folder
  python draw_regions.py Gre595_RH_3-1-4 --hp   # with the hippocampus layer (press 5)
"""

import sys
from pathlib import Path

import napari
import numpy as np
import pandas as pd

# Put the app's own module folders (src/*) on sys.path so the bare imports below resolve.
# Must run before the first of them. See _bootstrap.py for why the code is laid out this way.
import _bootstrap  # noqa: F401  -- imported for its import-time side effect, not for a name

import ps6
from draw_helpers import resolve_image

# A SCRIPT, NOT A MODULE -- AND IT SAYS SO RATHER THAN PROVING IT. This file has no main() to hide
# behind: everything below runs the moment it is read, which is what makes `python draw_regions.py`
# work and also means `import draw_regions` opens a napari window and saves a regions file.
#
# THAT HAPPENED ON 2026-09-16. A check whose whole job was "do the app's modules import?" imported
# this one, which opened Alice's first section and wrote its outlines back over hers. Nothing was
# lost -- same 58 vertices, moved by 0.0002 px in a float round-trip -- but a check has no business
# writing to her data, and the next such import might land on a section mid-edit. The guard costs
# nothing and turns a silent write into a sentence.
if __name__ != "__main__":
    raise ImportError(
        "draw_regions is a script, not a module: importing it opens a napari window and saves over "
        'a section\'s outlines. Run it as `python draw_regions.py "<section>.TIF"`, or let the app '
        "launch it (app_actions.draw_command), which is how the Draw outlines button works."
    )

# Anything starting with "--" is a switch, so an image fragment can still be given by name.
switches = {argument for argument in sys.argv[1:] if argument.startswith("--")}
arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]

# THE HIPPOCAMPUS LAYER IS OFF BY DEFAULT, and was on until 2026-08-20. Alice: *"can you also
# remove the hp annotation option in napari when i go to draw outlines"*. It is not used for
# counting and never was -- it is a label for the far-border problem -- so on the window she
# opens to outline NCM and CMM it is one more layer to select by accident.
#
# A SWITCH RATHER THAN A DELETION, for one reason: draw_hp_queue.py is what collected the 14
# outlines that exist, and eight analysis scripts read them. Deleting the layer would delete
# that tool, which is not what was asked. `python draw_regions.py <image> --hp` still has it,
# and the queue script passes the switch itself.
WITH_HP = "--hp" in switches

# Drawn by hand because they follow visible anatomy: the lamina separates CMM
# from NCM, and the tissue edge bounds both.
DRAWN_REGIONS = [
    {"name": "NCM", "color": "cyan"},
    {"name": "CMM", "color": "magenta"},
]

if WITH_HP:
    # Added 2026-08-13. NCM's far border IS the hippocampus border -- Alice: *"thats the most
    # important part"* -- and it is the one part of the region the model keeps getting wrong. Three
    # decodes of a learned border all scored worse than a constant ([[border-detector-hurts-the-
    # region]]), because nothing in the training data has ever shown the structure that stops NCM.
    # An outline here is a NEW kind of label, not more of the same.
    DRAWN_REGIONS.append({"name": "HP", "color": "orange"})

# NO ARGUMENT MEANS BIRDS[0] HERE, AND NOTHING IN AN EXPORTED COPY. Opening the first section in
# ps6.BIRDS is how this file has always been run by hand in this folder, so that stays. But a package
# built by package_app.py ships the program and not the sections, and there a bare run died inside
# tifffile with "No such file or directory: 'YW113_RH_1-1-7_NCM_Slide 3.TIF'" -- a stranger's first
# guess at how to use the file, answered with the name of one of Alice's birds. The app never reaches
# this branch: app_actions.draw_command always passes the image.
if arguments:
    image_path = resolve_image(arguments[0])
else:
    image_path = ps6.BIRDS[0]["image"]

    if not Path(image_path).exists():
        raise SystemExit(
            'Usage: python draw_regions.py "<section>.TIF"\n\n'
            "Name the section to outline. Normally you do not run this by hand -- the app opens\n"
            "this window for you when you press Draw outlines."
        )

output_path = ps6.region_path(image_path)
dorsal_output_path = ps6.dorsal_path(image_path)
field_l_output_path = ps6.field_l_ends_path(image_path)

print(f"Loading {Path(image_path).name}...")
image, green = ps6.load_green(image_path)

microns = ps6.pixel_size_um(image_path)
print(f"  {green.shape[0]} x {green.shape[1]} px", end="")
if microns:
    print(f", {microns:.3f} um/px")
else:
    print(", pixel size unknown (counts will have no density)")

viewer = napari.Viewer(title=f"Regions: {Path(image_path).stem}")

viewer.add_image(
    image,
    name="PS6",
    rgb=image.ndim == 3 and image.shape[-1] in (3, 4),
)

# Existing polygons, so this is resumable rather than starting over.
existing = {}

# WHAT THE MODEL PROPOSED, so `save_regions` can tell a proposal you accepted from an outline you drew. Filled
# both from a saved file's `source = model` rows and by `propose()` below. A polygon leaves this dict the
# moment its vertices differ from what is in here, which is the test `source_of` makes.
proposed = {}

# ROWS IN THE FILE THAT THIS WINDOW DOES NOT DRAW, written back untouched when saving. HP with the switch
# off, and whatever else is retired later.
#
# THE BUG THIS EXISTS TO PREVENT. save_regions rewrites the whole file from the layers on screen, so a region
# with no layer is a region that gets deleted -- silently, by a Ctrl-S that looked like it only saved NCM.
# Fourteen sections have a hippocampus outline and eight scripts read them, and they were drawn by hand one
# section at a time. Removing a layer from a window must not be able to remove labels from a disk.
carried = None

if Path(output_path).exists():
    table = pd.read_csv(output_path)
    sources = table["source"] if "source" in table.columns else None

    for name, group in table.groupby("region", sort=False):
        existing[str(name)] = group[["axis-0", "axis-1"]].to_numpy()

        # Re-opening a file must not launder a proposal into a label: if it was the model's last time and you
        # have not touched it, it is still the model's.
        if sources is not None and (group["source"] == "model").all():
            proposed[str(name)] = existing[str(name)].astype(float)

    print(f"  loaded {list(existing)} from {Path(output_path).name}")

    if proposed:
        print(f"  {', '.join(sorted(proposed))} came from the model, not from you")

    kept = table[~table["region"].astype(str).isin([r["name"] for r in DRAWN_REGIONS])].copy()

    if len(kept):
        # An old file with no source column: those rows are hers, since the model has never
        # proposed anything but NCM and CMM.
        if "source" not in kept.columns:
            kept["source"] = "hand"

        carried = kept[["region", "axis-0", "axis-1", "source"]]

        print(f"  {sorted(set(carried['region']))} is in the file but not shown here; it will be "
              f"kept exactly as it is")

layers = {}

for region in DRAWN_REGIONS:
    name = region["name"]
    shapes = [existing[name]] if name in existing else []

    layers[name] = viewer.add_shapes(
        shapes,
        name=name,
        shape_type="polygon",
        edge_color=region["color"],
        edge_width=12,
        face_color="transparent",
        opacity=0.9,
    )

dorsal_existing = ps6.load_dorsal_point(dorsal_output_path)

dorsal_layer = viewer.add_points(
    np.array([dorsal_existing]) if dorsal_existing is not None else np.empty((0, 2)),
    name="Dorsal side (hippocampus)",
    size=120,
    face_color="yellow",
    border_color="black",
    border_width=0.08,
)

if dorsal_existing is not None:
    print(f"  loaded dorsal marker from {Path(dorsal_output_path).name}")

field_l_existing = ps6.load_field_l_ends(field_l_output_path)

field_l_layer = viewer.add_points(
    field_l_existing if field_l_existing is not None else np.empty((0, 2)),
    name="Field L ends (2 clicks)",
    size=120,
    face_color="cyan",
    border_color="black",
    border_width=0.08,
)

if field_l_existing is not None:
    print(f"  loaded Field L ends from {Path(field_l_output_path).name}")


PREDICTED = ["NCM", "CMM"]

# WHICH SIDE OF FIELD L THE PREDICTION IS CURRENTLY USING when there is no dorsal marker to say. Ctrl-F
# toggles it. Alice, 2026-08-19: *"can you activate the annotaions without the dorsal click?"* -- so the
# outlines appear immediately on a guessed side rather than waiting, and the guess is labelled as one.
flipped = False

# The 8 um canvas the rules were fitted on, kept once built: it costs a tissue mask over the full-resolution
# image, and neither the image nor the mask changes when the dorsal marker moves.
canvas = None

viewer.text_overlay.visible = True
viewer.text_overlay.font_size = 14


def say(text):
    """On the image AND in the terminal.

    The terminal is not good enough on its own: the app launches this in a subprocess whose output you never
    see, so an instruction that only prints is an instruction that does not exist.
    """
    viewer.text_overlay.text = text
    print(f"  {text}")


# WHY THERE IS NO CANVAS, when there is none. Set once and reported instead of retried: some sections cannot
# go on it at all -- they are physically larger than the square the rule was fitted in -- and rebuilding to
# fail again on every keypress would cost ten seconds to say the same thing.
no_canvas = ""


def ensure_canvas():
    """Build the 8 um canvas once, and never let a failure reach the window. None if it cannot be built.

    Alice, 2026-08-19: *"for all the images that need regions the napari window doesn't open properly and then
    crasehs"*. Two uploads in birds/ are 4.3 x 5.3 mm where the canvas covers 3.84 mm, so `pad_into` raised --
    and it raised where no one could see it, killing a half-built window. Nothing about the prediction is
    worth a closed window: every failure becomes a sentence on the image, and drawing by hand is unaffected.
    """
    global canvas, no_canvas

    import predict_regions

    if canvas is None and not no_canvas:
        print("  building the 8 um canvas the rules were fitted on (once, about 10 s)...")

        try:
            canvas = predict_regions.canvas_for(image_path, green=green)
        except Exception as problem:  # noqa: BLE001 -- a window that vanishes is worse than any error
            import traceback

            traceback.print_exc()
            no_canvas = str(problem) or problem.__class__.__name__

    return canvas


def propose(_event=None):
    """Put the rules' own outlines into every NCM/CMM layer that is empty or still the model's.

    CALLED BY THE DORSAL CLICK ITSELF, not only by Ctrl-P. The click is what the prediction needs, so the
    click is the natural trigger: on a section you have not touched, placing the marker makes the outlines
    appear. Alice, 2026-08-19: *"i dont see the predicted outlines when i press draw outlines?"* -- the
    sections that need a prediction are exactly the ones with no marker yet, so waiting for a keypress meant
    the window opened empty and said so only in a terminal she cannot see.

    NO MARKER IS NO LONGER A REFUSAL. Which side of Field L is NCM is the only bit these outlines need, and
    nothing in the image predicts it, so with no marker the prediction runs on a GUESSED side and says so --
    Ctrl-F swaps it. The marker, once placed, always wins.

    NEVER OVER YOUR OWN WORK. A layer holding a polygon you have drawn or edited is left alone and reported,
    because losing a hand outline to a keypress would cost far more than the prediction is worth.
    """
    points = np.asarray(dorsal_layer.data)
    want = [name for name in PREDICTED if not len(layers[name].data) or name in proposed]
    mine = [name for name in PREDICTED if name not in want]

    if not want:
        say(f"Your own {' and '.join(mine)} are loaded, untouched. Ctrl-R clears a layer, Ctrl-P predicts it")
        return

    import predict_regions

    if ensure_canvas() is None:
        say(f"No predicted outlines: {no_canvas}")
        return

    try:
        predicted, why, guessed = predict_regions.predict(
            image_path,
            dorsal_point=points[0] if len(points) else None,
            canvas=canvas,
            flip=flipped,
        )
    except Exception as problem:  # noqa: BLE001 -- same reason as in ensure_canvas
        import traceback

        traceback.print_exc()
        say(f"The rule failed on this section ({problem.__class__.__name__}) -- draw NCM and CMM by hand")
        return

    if not predicted:
        say(f"No prediction: {why}")
        return

    for name in want:
        if name not in predicted:
            print(f"  the rule produced no {name} on this section")
            continue

        vertices = np.asarray(predicted[name], float)
        layers[name].data = []
        layers[name].add(vertices, shape_type="polygon")
        proposed[name] = vertices
        print(f"  proposed {name}: {len(vertices)} vertices")

    kept = " and ".join(f"your own {name}" for name in mine)
    made = " and ".join(n for n in want if n in predicted)

    # THE GUESS IS SAID EVERY TIME, not once at startup, because this is the one error that is silent in the
    # numbers and obvious on the screen: cyan on CMM and magenta on NCM. Everything else here you can see.
    if guessed:
        say(f"Predicted {made}{' (' + kept + ' left alone)' if mine else ''} -- but WHICH SIDE IS NCM WAS "
            f"GUESSED{' (flipped)' if flipped else ''}. If cyan and magenta are swapped, press Ctrl-F. "
            f"Press 4 and click the hippocampus side to settle it (needed for dNCM / vNCM).")
        return

    say(f"Predicted {made}{' (' + kept + ' left alone)' if mine else ''} from your dorsal marker "
        f"-- drag what is wrong, Ctrl-R to redraw a layer")


# THE CLICK IS THE TRIGGER. `events.data` also fires when the marker is dragged or deleted, which is right:
# moving it moves the band, so the prediction it produced is stale.
dorsal_layer.events.data.connect(propose)


# ------------------------------------------------------------------------------------------------------
# WHERE dNCM ENDS AND vNCM BEGINS, drawn while you work.
#
# Alice, 2026-08-20: *"lets say i dont like the predicted outline and i draw another ncm outline, is it
# possible to automatically split it into dncm and vncm and make it show in napari"*. It is, and the cut is
# worth seeing for a second reason: NCM is never counted as a whole. dNCM and vNCM are, and where the cut
# lands depends on the ENDS of your outline along Field L -- so two outlines that look equally good can
# report quite different dNCM densities. review_split.py has always shown this, but only after saving and
# only for a section already drawn.
#
# THE SAME FUNCTION THE COUNTER USES, ps6.split_axis, rather than a second copy of the rule here. A preview
# that can disagree with the count is worse than no preview: it would be believed.
# ------------------------------------------------------------------------------------------------------

# Long side of the working copy, in pixels. The cut is a centroid, a direction and the midpoint of an
# extent, and none of those move when the image is sampled coarsely, so this is about accuracy of a few
# microns bought for a redraw fast enough to follow a dragged vertex.
SPLIT_PIXELS = 600

split_layer = viewer.add_shapes(
    [],
    name="dNCM / vNCM cut",
    shape_type="line",
    edge_color="lime",
    edge_width=10,
    opacity=0.9,
)

# Nothing here is drawn by hand, and an accidental click on this layer that added a shape would look like a
# second cut. It is output, not input.
split_layer.editable = False

split_grid = None


def split_view():
    """The section's tissue mask on a decimated grid, and the step. Built once, then reused.

    `tissue_sigma` is divided by the step for the reason canvas_for gives: it is a distance in pixels, so
    leaving it alone would blur many times too far in microns. The tissue matters because ps6.split_ncm
    measures NCM's extent over tissue only -- a polygon drawn a little loose must not drag the cut outward.
    """
    global split_grid

    if split_grid is None:
        step = max(1, int(np.ceil(max(green.shape) / SPLIT_PIXELS)))
        geometry = ps6.geometry_for(image_path)
        geometry.tissue_sigma = geometry.tissue_sigma / step
        split_grid = (ps6.compute_tissue(np.ascontiguousarray(green[::step, ::step]), geometry), step)

    return split_grid


def cut_across(vertices, axis, centroid, midpoint):
    """The cut as a segment spanning the polygon, in image pixels. None if the line misses it.

    The rule gives an INFINITE line -- the points p with dot(p - centroid, axis) == midpoint -- so to draw
    it, it has to be cut down to the outline it belongs to. Each polygon edge whose two ends fall on
    opposite sides of the line crosses it exactly once, at the point where the straddling values reach zero.

    Taking the two EXTREME crossings rather than assuming there are only two: NCM's far border is the tissue
    edge and can be concave, so a line can leave and re-enter the outline. The span between the outermost
    crossings is what a cut through the region looks like either way, and it never ends inside the polygon,
    which a wrongly chosen pair would.
    """
    values = (vertices - centroid) @ axis - midpoint
    following = np.roll(values, -1)
    straddles = np.nonzero((values <= 0) != (following <= 0))[0]

    if len(straddles) < 2:
        return None

    crossings = []

    for index in straddles:
        here, there = values[index], following[index]
        share = here / (here - there)
        crossings.append(vertices[index] + share * (vertices[(index + 1) % len(vertices)] - vertices[index]))

    crossings = np.asarray(crossings)

    # Ordered ALONG the cut, which is perpendicular to `axis`.
    along = crossings @ np.array([-axis[1], axis[0]])

    return np.stack([crossings[along.argmin()], crossings[along.argmax()]])


def update_split(_event=None):
    """Redraw the cut from whatever NCM currently is. Never raises; never blocks the window.

    CALLED BY THE LAYERS THEMSELVES. `events.data` fires on drawing, on dragging a vertex, on deleting and
    on `propose` assigning a prediction, so the line follows the outline with nothing to press. The dorsal
    marker is connected too, though only to relabel: see below.

    THE LINE DOES NOT DEPEND ON THE DORSAL CLICK. ps6.split_axis uses the marker only for the SIGN of the
    axis -- which end is dorsal -- and the cut sits at the midpoint of the extent either way. So the line
    can be drawn before any click, and the marker decides only which half is called dNCM. Since the marker
    is already on screen in yellow, "the half nearer the yellow dot is dNCM" needs no extra drawing.

    SILENT WHEN IT WORKS, on purpose. The green line is the answer, and say() would overwrite whatever
    propose() has just put on screen. It only speaks when the cut cannot be computed -- which is a real
    warning, because count_cells.py calls the same rule and would refuse the same way, forty minutes in.
    """
    try:
        shapes = [np.asarray(s, dtype=np.float64) for s in layers["NCM"].data]
        outline = max((s for s in shapes if len(s) >= 3), key=len, default=None)

        if outline is None:
            split_layer.data = []

            return

        tissue, step = split_view()
        small = outline / step

        from skimage.draw import polygon2mask

        mask = polygon2mask(tissue.shape, small)

        if not (mask & tissue).any():
            split_layer.data = []
            say("NCM's outline does not overlap the tissue, so it cannot be split")

            return

        points = np.asarray(dorsal_layer.data)

        try:
            direction = ps6.field_l_direction(small)
        except ValueError as problem:
            split_layer.data = []
            say(f"NCM cannot be split into dNCM and vNCM: {problem}")

            return

        # A stand-in when there is no marker yet, far enough out that only its side matters. The line it
        # produces is the one the real click will give; only the labels d and v are still open.
        stand_in = small.mean(axis=0) + direction * 1000.0
        point = points[0] / step if len(points) else stand_in

        axis, centroid, midpoint = ps6.split_axis(mask, point, tissue, small)
        segment = cut_across(small, axis, centroid, midpoint)

        if segment is None:
            split_layer.data = []

            return

        split_layer.data = []
        split_layer.add(segment * step, shape_type="line")

        # The halves, for the terminal only. Which is which is only known once the marker exists.
        projection = (np.argwhere(mask & tissue) - centroid) @ axis
        dorsal_share = float((projection >= midpoint).mean())

        print(f"  cut drawn: {'dNCM' if len(points) else 'the marked half'} is "
              f"{dorsal_share * 100:.0f}% of NCM's area"
              f"{'' if len(points) else ' -- press 4 and click to say which half is dorsal'}")
    except Exception:  # noqa: BLE001 -- an exception in a layer callback kills the window
        import traceback

        traceback.print_exc()
        split_layer.data = []


layers["NCM"].events.data.connect(update_split)
dorsal_layer.events.data.connect(update_split)


def source_of(name, vertices):
    """"model" while a polygon is EXACTLY as proposed, "hand" from the first vertex you move.

    Exact equality, not a tolerance: napari hands back the same array until something drags a vertex, so any
    difference at all is an edit. Erring towards "hand" would be the dangerous direction -- it would let a
    proposal into the corpus -- and erring towards "model" only costs you an outline you did draw.
    """
    was = proposed.get(name)

    return "model" if was is not None and was.shape == vertices.shape and np.array_equal(was, vertices) \
        else "hand"


def write_if_changed(frame, path, what):
    """Write `frame` to `path` unless the file there already says exactly this. Returns whether it wrote.

    A SAVE THAT CHANGES NOTHING MUST TOUCH NOTHING. Two things downstream watch this file's
    modification time, and both were being fooled. app_table.is_stale calls a count out of date when
    the outlines are NEWER than it, and app_actions.draw_section throws away the cached region areas
    after a drawing window closes. So opening a section just to look at the predicted outlines, and
    letting the save on close do its thing, marked its count stale and blanked its density -- with the
    polygons on disk byte for byte identical. Alice, 2026-08-20: *"i feel like i had already counted a
    lot of the images but all of them say stale or not counted yet"*.

    Comparing the CSV text rather than the numbers, because the text is what the reader will get: a
    float that survives the round trip to the same string is the same file, and one that does not is a
    real difference however small it looks here.
    """
    text = frame.to_csv(index=False)
    path = Path(path)

    if path.exists() and path.read_text() == text:
        print(f"{what} unchanged, so {path} was left alone "
              f"(its count stays current and its areas stay cached)")

        return False

    path.write_text(text)

    return True


def save_regions():
    rows = []

    for name, layer in layers.items():
        for index, polygon in enumerate(layer.data):
            vertices = np.asarray(polygon)

            if len(vertices) < 3:
                print(f"  skipped {name} shape {index}: only {len(vertices)} vertices")
                continue

            # Only the first shape can be the proposal -- a second polygon on the same layer is yours.
            source = source_of(name, vertices) if index == 0 else "hand"

            for y, x in vertices:
                rows.append({"region": name, "axis-0": y, "axis-1": x, "source": source})

    if rows:
        saved = pd.DataFrame(rows)

        # The regions this window did not show, put back before the file is overwritten. See the
        # comment where `carried` is set: this is what stops a hidden layer becoming a deletion.
        if carried is not None and len(carried):
            saved = pd.concat([saved, carried], ignore_index=True)

        drawn = saved["region"].unique().tolist()

        if write_if_changed(saved, output_path, f"{', '.join(drawn)}"):
            print(f"Saved {drawn} to {output_path}")

            untouched = sorted(set(saved.loc[saved["source"] == "model", "region"]))

            if untouched:
                print(f"  {', '.join(untouched)} saved as the MODEL's, not yours -- it will be used for "
                      f"counting but kept out of training until you edit it")
    else:
        print("Nothing drawn yet; nothing saved.")

    points = np.asarray(dorsal_layer.data)

    if len(points):
        # Only the first click matters; extras are almost always accidents.
        if len(points) > 1:
            print(f"  {len(points)} dorsal markers; using the first")

        if write_if_changed(
            pd.DataFrame([{"axis-0": points[0][0], "axis-1": points[0][1]}]),
            dorsal_output_path,
            "the dorsal marker",
        ):
            print(f"Saved dorsal marker to {dorsal_output_path}")
    elif rows:
        print("  WARNING: no dorsal marker. NCM cannot be split into dNCM and vNCM without one, and any "
              "outline still exactly as predicted was placed on a GUESSED side of Field L.")

    ends = np.asarray(field_l_layer.data)

    if len(ends) >= 2:
        # The FIRST TWO, and a warning when there are more. Silently taking the last two would make a
        # stray click overwrite a deliberate one, and silently taking all of them would write a file
        # whose reader only ever looks at two rows.
        if len(ends) > 2:
            print(f"  {len(ends)} Field L clicks; using the first two")

        if write_if_changed(
            pd.DataFrame([{"axis-0": y, "axis-1": x} for y, x in ends[:2]]),
            field_l_output_path,
            "the Field L ends",
        ):
            print(f"Saved Field L ends to {field_l_output_path}")
    elif len(ends) == 1:
        print("  1 Field L click; two are needed (one at each end), so it was not saved")


# Plain "s" is a Shapes-layer shortcut, so it never reaches a viewer binding.
# Ctrl-S with overwrite is the reliable one.
@viewer.bind_key("Control-S", overwrite=True)
def save_now(_viewer):
    save_regions()


# CTRL-P AND CTRL-R, not plain P and R. Plain "P" is napari's own "add polygon mode" and plain "C" is
# "auto-contrast", so those never reach a viewer binding -- the same trap that made "s" unusable for saving.
@viewer.bind_key("Control-P", overwrite=True)
def predict_now(_viewer):
    propose()


@viewer.bind_key("Control-F", overwrite=True)
def flip_side(_viewer):
    """Swap which side of Field L the prediction thinks is NCM, and predict again.

    Only meaningful with no dorsal marker: with one, the marker decides and this would be overriding a fact
    with a guess. It clears the layers it proposed first, so the flipped outlines replace them instead of
    being refused as "already predicted" -- and it still never touches a layer you have drawn on.
    """
    global flipped

    if len(np.asarray(dorsal_layer.data)):
        say("Your dorsal marker decides which side is NCM -- move it (press 4) rather than flipping")
        return

    flipped = not flipped

    for name in PREDICTED:
        if name in proposed:
            layers[name].data = []
            proposed.pop(name, None)

    propose()


@viewer.bind_key("Control-R", overwrite=True)
def clear_active(_viewer):
    """Empty the layer you are on, so a bad outline can be redrawn instead of dragged into shape."""
    layer = viewer.layers.selection.active

    if layer is None or layer.name not in layers:
        keys = "1, 2 or 5" if "HP" in layers else "1 or 2"
        say(f"Ctrl-R clears an outline layer; press {keys} first")

        return

    layer.data = []
    proposed.pop(layer.name, None)
    layer.mode = "add_polygon"
    print(f"  cleared {layer.name}; draw it again (Ctrl-P puts the model's back)")


@viewer.bind_key("1")
def select_ncm(_viewer):
    viewer.layers.selection.active = layers["NCM"]
    layers["NCM"].mode = "add_polygon"


@viewer.bind_key("2")
def select_cmm(_viewer):
    viewer.layers.selection.active = layers["CMM"]
    layers["CMM"].mode = "add_polygon"


# "3" is a layer-mode shortcut in napari, so the marker layer uses "4".
@viewer.bind_key("4")
def select_dorsal(_viewer):
    viewer.layers.selection.active = dorsal_layer
    # Selecting a Points layer does not arm it, so without this the first click after pressing 4 does
    # nothing and the key looks broken.
    dorsal_layer.mode = "add"


@viewer.bind_key("5")
def select_hippocampus(_viewer):
    """The HP layer, when there is one. Otherwise a sentence saying where it went.

    BOUND EVEN WITH THE LAYER OFF, deliberately. Pressing 5 out of habit and getting nothing at
    all is the same experience as a broken key, and the terminal that would explain it is hidden
    when the app launches this ([[app-messages-must-be-on-screen]]).
    """
    if "HP" not in layers:
        say("The hippocampus layer is off. Run:  python draw_hp_queue.py   to draw those outlines")

        return

    viewer.layers.selection.active = layers["HP"]
    layers["HP"].mode = "add_polygon"


@viewer.bind_key("6")
def select_field_l_ends(_viewer):
    viewer.layers.selection.active = field_l_layer
    field_l_layer.mode = "add"


def open_with_prediction():
    """The canvas build and the first prediction, run AFTER the window is on screen.

    NOT BEFORE `napari.run()`, which is where this used to be. Ten seconds of numpy before the event loop
    starts means the window is created and never painted -- indistinguishable from a crash, and an exception
    in it IS one. Scheduled on the loop instead: the section appears first, then the outlines land on top of
    it, and anything that goes wrong is a sentence over a working window.

    The whole body is guarded. `ensure_canvas` and `propose` already catch what they can name; this catches
    what they cannot.
    """
    try:
        if any(not len(layers[name].data) or name in proposed for name in PREDICTED):
            ensure_canvas()

        propose()

        # The cut for an outline that was ALREADY on disk. propose() assigning a prediction fires the layer
        # event by itself, but a loaded polygon was put there before this layer existed, so nothing fired.
        update_split()
    except Exception:  # noqa: BLE001 -- last line before the window dies
        import traceback

        traceback.print_exc()
        say("The predicted outlines failed on this section -- draw NCM and CMM by hand, the rest still works")


def schedule_prediction():
    """Say what is about to happen, let the window paint, then do it.

    Two hops rather than one: the message set here cannot appear while the same callback goes on to block the
    loop for ten seconds, so the work goes in a second callback and the paint happens in between.

    BOTH PATHS GO THROUGH THE TIMER, including the one with nothing to predict. It is no longer free: it
    still draws the dNCM/vNCM cut, which costs a tissue mask over a decimated copy of the section. Anything
    measured in seconds before napari.run() means a window created and never painted.
    """
    if all(len(layers[name].data) and name not in proposed for name in PREDICTED):
        say("Drawing where dNCM and vNCM divide...")
    else:
        say("Building this section's 8 um canvas and predicting NCM and CMM -- about 10 seconds...")

    try:
        from qtpy.QtCore import QTimer

        QTimer.singleShot(50, open_with_prediction)
    except ImportError:
        # No Qt (the headless test harness stubs napari). Nothing to yield to, so just do the work.
        open_with_prediction()


schedule_prediction()

viewer.layers.selection.active = layers["NCM"]
layers["NCM"].mode = "add_polygon"

print(
    "\nThe predicted NCM (cyan) and CMM (magenta) are already drawn. Fix what is wrong by dragging.\n"
    "If they are on the wrong sides of Field L, press Ctrl-F. Then press 4 and click the hippocampus\n"
    "side -- that is still needed to split NCM into dNCM and vNCM.\n"
    "The GREEN line across NCM is where the counter will cut dNCM from vNCM. It follows your outline\n"
    "as you draw, and the half nearer the yellow marker is dNCM.\n"
    "  1 / 2   switch to NCM / CMM      4  dorsal marker      6  Field L ends (2 clicks)\n"
    "  Ctrl-S  save     Ctrl-P  predict again     Ctrl-F  flip the sides     Ctrl-R  clear this layer\n\n"
    "An outline still exactly as proposed is saved as the model's and stays out of training.\n"
    "dNCM and vNCM are computed from NCM -- do not draw them.\n"
    f"Then check the split:  python review_split.py {Path(image_path).name!r}\n"
)

try:
    napari.run()
finally:
    save_regions()
