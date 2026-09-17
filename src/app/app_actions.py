"""Run the three jobs a section needs: draw outlines, count cells, refresh areas.

Step 3 of the logic layer. app_state.py says what exists, app_table.py says what the
numbers are, and this module DOES things. The window will call these functions and show
what they return; it will not build a command line or touch a subprocess itself.

EVERY JOB RUNS IN A SEPARATE PROCESS, and for two distinct reasons.

The first is unavoidable: draw_regions.py calls napari.run(), which starts a Qt event loop
that blocks until the window closes. The app will itself be a Qt window with its own event
loop, and two event loops in one process fight -- the usual symptom is a frozen window or a
hard crash. A subprocess has its own loop and napari behaves exactly as it does when run by
hand today.

The second is the reason worth remembering: a subprocess cannot take the app down with it,
and it cannot lose work. draw_regions.py saves in a `finally:` block, so outlines reach the
disk even if it exits badly -- and because the files ARE the state (see app_state.py), the
app can crash at any moment without costing anything. The app is a LAUNCHER, not a
container. No component ever holds the only copy of anything.

INVALIDATING THE AREA CACHE IS THIS MODULE'S JOB. app_table.py caches region areas to a
sidecar because computing them means opening a TIF. That cache is only correct while the
outlines are unchanged, so redrawing a section must delete it -- otherwise the table shows
densities computed against outlines that no longer exist. app_table.py already DETECTS this
for counts by comparing modification times; here it can be PREVENTED at the source, which
is better. Detect what you cannot prevent; prevent what you can.

Usage:
  python app_actions.py                          # show what each section would run
  python app_actions.py --draw "<stem or path>"   # open the drawing window
  python app_actions.py --count "<stem or path>"  # count one section
  python app_actions.py --review "<stem or path>" # look at the cells it counted
  python app_actions.py --areas "<stem or path>"  # recompute its cached areas
"""

import subprocess
import sys
from pathlib import Path

import app_awake
from app_state import (
    COUNTED,
    NEEDS_DORSAL,
    NEEDS_REGIONS,
    READY_TO_COUNT,
    find_sections,
    sidecar_paths,
)
from app_table import areas_path

# WHERE THE APP'S OWN CODE LIVES, worked out from this file rather than from the working
# directory. Every script launched below used to be named bare -- "count_cells.py" -- which
# means "count_cells.py in whatever folder the app happened to be started from". That is true
# when the app is run by typing `python app.py` inside the project, and false every other way:
# double-clicked from Finder, started by a launcher, or installed on someone else's computer.
# The symptom is not a subtle one, but it is a confusing one: "can't open file 'count_cells.py'"
# in the log pane of an app whose files are plainly all present.
#
# The same reasoning as sidecar_paths in app_state.py, and as MODEL_PATH in count_cells.py:
# ANCHOR A PATH TO SOMETHING THAT CANNOT MOVE. A file's own location cannot move relative to
# its neighbours; the working directory is whatever the person who started the process was
# looking at.
HERE = Path(__file__).resolve().parent

# What each status most usefully leads to. The window will use this to decide which button
# to highlight, so the suggestion lives here with the rest of the logic rather than being
# re-derived from status strings in the interface.
NEXT_ACTION = {
    NEEDS_REGIONS: "draw",
    NEEDS_DORSAL: "draw",
    READY_TO_COUNT: "count",
    COUNTED: "review",
}


def python_executable():
    """The interpreter running this app, not whatever `python3` resolves to.

    sys.executable rather than the string "python3": the app may be launched from a virtual
    environment, and a bare "python3" would find the system interpreter instead -- which has
    no torch or napari installed. This is the single most common way a subprocess that works
    in a terminal fails from inside an app.
    """
    return sys.executable


def script(name):
    """One of the app's own scripts, by absolute path. See HERE."""
    return str(HERE / name)


def image_argument(image_path):
    """The image, by absolute path, for handing to another process.

    RESOLVED BECAUSE THE CHILD DOES NOT SHARE OUR WORKING DIRECTORY -- run() starts it in HERE,
    so a relative path like "birds/Gre595....TIF" would be looked for in the app's folder rather
    than in the folder the person chose. find_sections returns paths relative to the root it was
    given, and app_folder.startup_folder's default root is ".", so relative paths are the ordinary
    case and not an edge one.

    Resolving also fixes the sidecars for free: count_cells writes "<stem> counted cells.csv"
    beside the image it was given, so an absolute image path means the counts land next to the
    TIF instead of next to the app.
    """
    return str(Path(image_path).resolve())


def draw_command(image_path):
    return [python_executable(), script("draw_regions.py"), image_argument(image_path)]


def count_command(image_path, model_path=None, save_points=True):
    """The counting command. Points are saved by default.

    --points writes "<stem> counted cells.csv", which is what app_table.py reads counts
    from and what review_counts.py draws. Without it a count would print numbers and leave
    nothing behind, so the app would show a section as still needing counting.

    `-u` IS WHY THE LOG MOVES. Python decides how to buffer stdout by looking at where it
    goes: to a terminal it flushes every line, but into a PIPE -- which is what the app gives
    it -- it fills an 8 KB buffer first. A count prints a few hundred bytes over 40 minutes,
    so it never fills one, so the window showed nothing at all until the process exited and
    the buffer was flushed on the way out. Measured 2026-08-20: after a count had printed ten
    lines, the app's log had zero of them. -u turns buffering off, and the only cost is a
    write syscall per line.

    This is safe to do only because count_cells.py now shrugs off a broken pipe
    (ignore_a_lost_listener). Unbuffered output means a count actually notices when the app
    closes; without that guard, streaming would turn "the window went away" into "the count
    died", which is the bug Alice hit on Gre595.
    """
    command = [python_executable(), "-u", script("count_cells.py"), image_argument(image_path)]

    if model_path:
        command.append(str(model_path))

    if save_points:
        command.append("--points")

    return command


def review_command(image_path, regions=True):
    """The command that shows the counted cells on the image.

    Regions are ON here, unlike show_counts.py's own default. That default is off because
    outlining costs another full-size array on a 40 Mpx section, but a reviewer's most common
    question is "is that cell inside CMM or just outside it" -- and without the boundary drawn
    that question cannot be answered at all. Worth the memory for the one job this exists to
    do.
    """
    command = [python_executable(), script("show_counts.py"), image_argument(image_path)]

    if regions:
        command.append("--regions")

    return command


def review_section(section, on_output=None):
    """Show the counted cells, and let them be corrected. Refuses if there is nothing to show.

    Checked here rather than letting show_counts.py exit: its own message is a good one, but
    it tells the person to run a command line, which is the thing the app exists to replace.

    NOT READ-ONLY ANY MORE, 2026-09-12. Alice: *"could you change the app so that review cells
    lets the user add and delete cells"*. Cells added or deleted in the window are written back
    to "<stem> counted cells.csv" when it closes, which is the file app_table.py counts rows in
    -- so the app's numbers follow with no extra work here. What that costs is the old argument
    for why review was safe to press on anything; what replaces it is show_counts.py writing
    nothing at all unless something actually changed, plus an append-only log of every edit.
    """
    paths = sidecar_paths(section["path"])

    if not paths["counts"].exists():
        return None, ["Not counted yet -- there are no cells to review."]

    return run(review_command(section["path"]), on_output)


def resolve_section(name, sections=None):
    """Find a section by stem, filename or path. Returns the section dict.

    Accepts any of the three because they all get used: the window has the dict, a person
    typing at a terminal has the stem, and a path is what gets pasted from Finder. Matching
    on the stem last means an exact path always wins over a partial name.
    """
    sections = sections if sections is not None else find_sections()
    wanted = Path(name).stem

    for section in sections:
        if section["path"] == name or section["stem"] == wanted:
            return section

    partial = [s for s in sections if wanted.lower() in s["stem"].lower()]

    if len(partial) == 1:
        return partial[0]

    if len(partial) > 1:
        raise SystemExit(
            f"{name!r} matches {len(partial)} sections:\n  "
            + "\n  ".join(s["stem"] for s in partial)
        )

    raise SystemExit(f"No section matching {name!r}. Run app_state.py to list them.")


def invalidate_areas(image_path):
    """Delete the cached areas for a section, so the table recomputes them.

    Called after drawing, because new outlines mean new areas. Deleting is safe: the cache
    is derived data, reproducible from the TIF and the outlines at any time. Nothing
    original is ever removed here.
    """
    path = areas_path(image_path)

    if path.exists():
        path.unlink()

        return True

    return False


def run(command, on_output=None, on_start=None):
    """Run a command, streaming its output. Returns (exit code, captured lines).

    Streams rather than waiting, so the window can show progress during a count that takes
    a minute -- subprocess.run() would return nothing until the end, and a UI that shows
    nothing for sixty seconds looks broken.

    READING IS NOT ENOUGH; THE CHILD HAS TO FLUSH. This loop reads whatever arrives, but the
    child decides when anything arrives, and a piped Python process buffers 8 KB by default.
    Reading eagerly at this end while the other end holds everything back looks exactly like
    a hung job. That is why count_command passes -u; anything long added here needs the same.

    stderr is merged into stdout so warnings appear in order with the output they relate to.
    Two separate streams get interleaved unpredictably and a traceback ends up detached from
    the step that caused it. Note the consequence: this pipe is shared, so when the app goes
    away BOTH of the child's streams break at once.

    `on_start` is handed the Popen object the moment it exists, and exists for exactly one
    caller: app_pause.Pause needs the handle so a count can be suspended while a drawing
    window is open. Without it nothing outside this function ever sees the process, because
    the loop below owns it from launch to exit. It is NOT a general hook -- anything else
    reaching for it should get its own reason written down here first.
    """
    lines = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # STARTED IN THE APP'S OWN FOLDER, not wherever the app was launched from. Every command
        # built here names its script absolutely, so this is not what makes them findable -- it is
        # what makes the files they read and write land in the same places they always have:
        # count_cells.py's ps6_counts.csv and ps6_counts_history.csv are relative by design, and a
        # count started by double-clicking would otherwise scatter them into the user's home folder
        # or, on Windows, into C:\.
        cwd=HERE,
    )

    if on_start:
        on_start(process)

    for line in process.stdout:
        line = line.rstrip("\n")
        lines.append(line)

        if on_output:
            on_output(line)

    return process.wait(), lines


def draw_section(section, on_output=None):
    """Open the drawing window, then invalidate the area cache IF the outlines changed.

    It used to invalidate unconditionally, on this reasoning: deciding whether the outlines
    really moved would mean comparing polygons, and a cache wrongly kept is a wrong density on
    a report while a cache wrongly deleted costs one TIF read.

    WHAT CHANGED, 2026-08-20: draw_regions.write_if_changed now leaves the file untouched when a
    save would rewrite it identically. So "did the outlines move?" no longer needs polygons
    compared -- the file's modification time answers it, because nothing else touches that file.
    And the cost of over-invalidating turned out not to be one TIF read: the app shows a blank
    area and a blank DENSITY until something recomputes it, and nothing does so automatically.
    Opening a section to look at the predicted outlines wiped its density, which is how Alice
    lost Gre595's. Same argument, better information.

    The mtime is read BEFORE napari opens, because the window is where the change happens.
    """
    paths = sidecar_paths(section["path"])
    before = paths["regions"].stat().st_mtime if paths["regions"].exists() else None

    code, lines = run(draw_command(section["path"]), on_output)

    after = paths["regions"].stat().st_mtime if paths["regions"].exists() else None

    if after == before:
        if on_output:
            on_output("  outlines unchanged; the cached areas and any count stay as they were")

        return code, lines

    if invalidate_areas(section["path"]) and on_output:
        on_output("  outlines changed, so the cached areas were cleared and will be recomputed")

    return code, lines


def count_section(section, model_path=None, on_output=None, on_start=None):
    """Count one section. Refuses if the section is not ready.

    Checked here rather than letting count_cells.py fail: ps6.resolve_regions raises a
    FileNotFoundError without a dorsal marker, and "no dorsal marker yet" is a far more
    useful thing for the app to say than a traceback.

    THE ONLY JOB THAT PASSES `on_start`, because it is the only one that gets paused: a count
    is the long job, so it is the one that has to give way when a drawing window opens. Drawing
    and reviewing are the other way round -- a person is sitting in front of them.

    THE COMPUTER IS HELD AWAKE for as long as the count lives -- see app_awake.py, which does it by
    a different mechanism on each of the three systems and says so on screen where it cannot. Here
    rather than in the window, because it is a property of counting rather than of the app: a count
    started from a terminal is left overnight just as often, and it needs it just as much.
    """
    paths = sidecar_paths(section["path"])

    if not paths["regions"].exists():
        return None, ["No outlines drawn yet -- draw them first."]

    if not paths["dorsal"].exists():
        return None, ["No dorsal marker yet -- NCM cannot be split without one."]

    def started(process):
        app_awake.keep_awake(process.pid)

        if on_start:
            on_start(process)

    return run(count_command(section["path"], model_path), on_output, started)


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    sections = find_sections()

    def echo(line):
        print(f"  | {line}")

    if "--draw" in sys.argv:
        section = resolve_section(arguments[0], sections)
        print(f"drawing {section['stem']}\n")
        code, _ = draw_section(section, echo)
        raise SystemExit(code)

    if "--count" in sys.argv:
        section = resolve_section(arguments[0], sections)
        print(f"counting {section['stem']}\n")
        code, lines = count_section(section, on_output=echo)

        if code is None:
            print("\n".join(lines))
            raise SystemExit(1)

        raise SystemExit(code)

    if "--review" in sys.argv:
        section = resolve_section(arguments[0], sections)
        print(f"reviewing {section['stem']}\n")
        code, lines = review_section(section, echo)

        if code is None:
            print("\n".join(lines))
            raise SystemExit(1)

        raise SystemExit(code)

    if "--areas" in sys.argv:
        section = resolve_section(arguments[0], sections)
        invalidate_areas(section["path"])
        print(f"recomputing areas for {section['stem']}\n")
        code, _ = run(
            [python_executable(), script("app_table.py"), "--areas"], echo
        )
        raise SystemExit(code)

    print(f"interpreter: {python_executable()}\n")
    print(f"{'section':44s} {'status':22s} next    command")
    print("-" * 96)

    for section in sections:
        action = NEXT_ACTION[section["status"]]
        command = (
            draw_command(section["path"])
            if action == "draw"
            else count_command(section["path"])
        )

        # Only the script and its arguments, so the line stays readable; the interpreter is
        # printed once above.
        shown = " ".join(command[1:])

        print(
            f"{section['stem'][:44]:44s} {section['status']:22s} "
            f"{action:7s} {shown[:44]}"
        )

    print(
        "\nNothing was run. Use --draw, --count or --areas with a section name:\n"
        f"  python app_actions.py --count {sections[0]['stem'][:28]!r}"
    )


if __name__ == "__main__":
    main()
