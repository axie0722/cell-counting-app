"""Put the app's five folders on Python's import path. The one file that makes the folders work.

Alice, 2026-09-17: *"i care about organizing the actual app not the research"*. The app that goes to
other people was 67 files in one flat folder, and this is what lets it be a tree instead.

WHY ANYTHING IS NEEDED AT ALL. Every import in this project is by bare name -- `import ps6`,
`import find_band`, `from rule_ncm import rule_border` -- 853 of them. Python resolves a bare import
by looking through `sys.path`, and `sys.path` starts with the folder of the script that was started
and NOT with any of its sibling folders. So `app.py` at the root can import `app_state` from `app/`
only if something has put `app/` on that list.

On THIS computer something already does: `.venv/lib/python3.12/site-packages/cell_counting_project.pth`
names the six folders, and Python reads it at startup. That file is part of the virtual environment,
which is 1.5 GB of compiled binaries for one machine and is never sent to anybody. A stranger has no
such file, which is why the exported app used to be flat -- flat is the one arrangement that needs no
import setup, because every module is already in the folder of the script that started.

WHAT THIS DOES INSTEAD, and it is two things, not one:

  * `sys.path` -- for THIS process. Enough for app.py itself.
  * `PYTHONPATH` in the environment -- for the processes this one STARTS. A count, a drawing window
    and the areas refresh are separate processes (see app_actions.py for why), each a fresh Python
    that reads none of our sys.path. Children inherit the environment, so setting PYTHONPATH here is
    what carries the folders to them. Miss this and the app opens perfectly and then fails the moment
    a button is pressed, which is the worst version of this bug: it looks like the button is broken.

IT DOES NOTHING HARMFUL IN A FLAT FOLDER. Each folder is added only if it exists, so a copy of the
app that is still flat -- an old zip somebody kept -- gets the root added and nothing else, which is
what it had anyway. That also means this file can be called from anywhere without asking which
layout it is in.

WHY NOT MAKE IT A REAL PACKAGE. The proper Python answer to "my modules are in folders" is
`from .rules import rule_ncm` and an __init__.py in each. That would mean rewriting all 853 imports,
and every one of them would then be wrong when the file is run directly as a script -- which is how
most of them ARE run, since every module here has a __main__ that can be checked from a terminal.
The cost is this file; the alternative is a different project.

Usage:
  python app_path.py        # print what it would add, and whether each folder is there
"""

import os
import sys
from pathlib import Path

# THE ROOT IS THIS FILE'S OWN FOLDER, which is the one arrangement that survives being moved,
# renamed, unzipped somewhere unexpected, or run from a different working directory. app.py sits
# beside this file in both layouts -- the project folder and the exported app -- so "beside
# app_path.py" and "the root of the app" are the same place by construction.
ROOT = Path(__file__).resolve().parent

# The five, in the order they are searched. Same list as app_actions.SCRIPT_FOLDERS and
# package_app.CODE_FOLDERS; if one ever gains a folder they all three have to.
FOLDERS = ["app", "regions", "review", "training", "counting"]


def paths():
    """The folders to add, root first. Only the ones that are actually there."""
    return [ROOT] + [ROOT / name for name in FOLDERS if (ROOT / name).is_dir()]


def setup():
    """Add them to sys.path for this process and to PYTHONPATH for every process it starts.

    Safe to call twice: nothing is added that is already present, so the app calling this and then a
    child of the app calling it again does not build a sys.path with six copies of everything in it.
    """
    wanted = [str(path) for path in paths()]

    # INSERTED AT THE FRONT, in reverse, so the final order matches `paths()` -- root first. It
    # matters for exactly one reason: if a name ever existed in two folders, the earlier one wins,
    # and "the root wins" is the only rule anybody could remember. package_app.py refuses to build
    # a package where two files share a name at all, so this should never decide anything.
    for path in reversed(wanted):
        if path not in sys.path:
            sys.path.insert(0, path)

    # APPENDED TO WHATEVER PYTHONPATH ALREADY SAID, not replacing it. Someone may have set it for
    # their own reasons, and silently dropping it would break something invisible to us.
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(wanted + [p for p in inherited if p not in wanted])


def report():
    """What it would add, for checking by eye from a terminal."""
    print(f"root: {ROOT}\n")

    for name in FOLDERS:
        folder = ROOT / name
        count = len(list(folder.glob("*.py"))) if folder.is_dir() else 0
        state = f"{count:2d} Python files" if folder.is_dir() else "not here (flat layout?)"
        print(f"  {name + '/':12s} {state}")

    print("\nwould put on the import path, in this order:")

    for path in paths():
        print(f"  {path}")


if __name__ == "__main__":
    report()
