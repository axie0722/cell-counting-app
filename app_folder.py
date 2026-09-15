"""Which folder the app looks in, remembered between launches, and whether it can be written to.

Separate from app_state.py on purpose. app_state's whole claim is that nothing in it can go
stale, because it derives everything from files that exist right now. A REMEMBERED FOLDER IS
EXACTLY THE OPPOSITE: it is stored state, and it can be wrong -- the folder can be renamed,
deleted, or sit on a USB drive that is not plugged in. Every function here has to cope with
that, so it lives apart rather than weakening the guarantee next door.

WHY THE WRITE TEST. Sidecars are written NEXT TO EACH IMAGE (see ps6.sidecar_path), so a
folder the app can read but not write is a folder where drawing outlines appears to work and
then loses everything at save time. That is the worst possible failure: it costs a person
half an hour of careful clicking and gives no warning. Places this actually happens are
ordinary -- a read-only network share, a mounted disk image, a colleague's folder shared with
read permission, macOS quarantining an external drive.

So the check happens WHEN THE FOLDER IS CHOSEN, not when a save fails. And it is a real
write of a real file, not os.access: os.access asks the permission bits, which are only one
of the reasons a write can fail, and it reports "writable" for read-only network mounts and
full disks. The only honest test of whether a folder can be written to is to write to it.

Usage:
  python app_folder.py                 # what is remembered, and can it be written to
  python app_folder.py /path/to/folder # check a folder without remembering it
"""

import json
import sys
from pathlib import Path

from app_state import find_images

# In the home folder, not the project folder. This is a preference belonging to the PERSON,
# not data belonging to the images: if the images move to another disk the choice should
# still be remembered, and if the project folder is read-only -- which is the exact case the
# write test exists to catch -- the app must still be able to save it.
SETTINGS_PATH = Path.home() / ".ps6_cell_counter.json"

# The name of the file the write test creates and deletes. Leading dot so it is hidden, and
# recognisable so that if a crash ever leaves one behind, a person finding it can tell what
# made it rather than wondering whether it matters.
WRITE_TEST_NAME = ".ps6_write_test"


def load_settings():
    """Whatever is stored, as a dict. An unreadable or corrupt file reads as empty.

    Never raises. A preferences file is not worth crashing over -- if it cannot be
    understood, the right behaviour is to fall back to the default and carry on, exactly as
    if the app had never been run before.
    """
    try:
        settings = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}

    return settings if isinstance(settings, dict) else {}


def store(**values):
    """Merge `values` into the settings file. True if it was written.

    MERGED, not replaced, so two features can keep something in one file without either
    erasing the other -- app_saved_queue.py stores the counting queue here beside the folder.
    A value of None removes its key, which is how a queue is forgotten without leaving a null
    behind for the next reader to interpret.

    Never raises: a preferences file that cannot be written is not worth losing the app over.
    The caller gets False and can say so, which is all anything here does with it.
    """
    settings = load_settings()

    for key, value in values.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value

    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except OSError:
        return False

    return True


def remember(folder):
    """Store a folder as the one to open next time. Returns True if it was saved.

    Absolute, because the app's working directory is not guaranteed to be the same next
    launch -- a relative path remembered from a terminal run would point somewhere else when
    the app is opened from Finder.
    """
    return store(folder=str(Path(folder).expanduser().resolve()))


def remembered():
    """The stored folder if it still exists, otherwise None.

    Existence is checked HERE rather than by the caller, so a folder that has been renamed
    or unplugged simply reads as "nothing remembered" and the app falls back to its default.
    A stored path that no longer resolves is not information; it is a wrong answer.
    """
    folder = load_settings().get("folder")

    if not folder:
        return None

    path = Path(folder)

    return path if path.is_dir() else None


def startup_folder(default="."):
    """The folder to open with: what was remembered, or the default.

    The default is the working directory, so running `python app.py` in a folder of images
    works with no setup at all -- which is how this project was used before there was a
    picker, and should keep working.
    """
    return remembered() or Path(default)


def can_write(folder):
    """Whether a file can actually be created in `folder`. Returns (ok, reason).

    Writes a real file and deletes it again. See the module docstring for why nothing
    cheaper is trustworthy.
    """
    probe = Path(folder) / WRITE_TEST_NAME

    try:
        # "x" so an existing file is an error rather than being silently overwritten. If a
        # probe is somehow already there, that is worth reporting, not clobbering.
        with open(probe, "x") as handle:
            handle.write("ok")
    except FileExistsError:
        # Left behind by an interrupted check. The folder was writable when it was made, and
        # saying so is better than reporting a failure that is really our own litter.
        return True, "writable (a previous write test was left behind)"
    except OSError as error:
        return False, error.strerror or str(error)

    try:
        probe.unlink()
    except OSError:
        # The write succeeded, which is the question that was asked. A folder that allows
        # creating but not deleting is odd but not a problem for saving sidecars.
        return True, "writable, but the test file could not be removed"

    return True, "writable"


def inspect(root):
    """Everything worth knowing about a folder before committing to it.

    Returns a dict:
      root        the folder, resolved
      images      how many distinct sections were found
      duplicates  (loser, winner) pairs for repeated filenames
      unwritable  [(folder, reason)] for folders holding images that cannot be written to

    EVERY FOLDER HOLDING AN IMAGE IS TESTED, not just the root, because sidecars go beside
    the image and the scan is recursive -- a writable root with a read-only subfolder inside
    it would pass a root-only check and still lose someone's outlines.
    """
    root = Path(root).expanduser()

    if not root.is_dir():
        return {
            "root": root,
            "images": 0,
            "duplicates": [],
            "unwritable": [],
            "missing": True,
        }

    images, duplicates = find_images(root)

    # Deduplicated, because 16 images in one folder is one write test, not 16.
    folders = sorted({image.parent for image in images} | {root})

    unwritable = []
    for folder in folders:
        ok, reason = can_write(folder)

        if not ok:
            unwritable.append((folder, reason))

    return {
        "root": root,
        "images": len(images),
        "duplicates": duplicates,
        "unwritable": unwritable,
        "missing": False,
    }


def describe(report):
    """The inspection as lines for a person to read. Worst news first.

    Returned as a list rather than printed so the app can put them in its log pane and a
    terminal can print them, without this module knowing which it is talking to.
    """
    root = report["root"]

    if report["missing"]:
        return [f"{root} is not a folder."]

    lines = [f"{root}", f"  {report['images']} sections found"]

    if not report["images"]:
        lines.append(
            "  Nothing to count here. The scan looks in this folder and every folder"
        )
        lines.append(
            "  inside it for .tif files, so check you picked the right one."
        )

    for folder, reason in report["unwritable"]:
        lines.append(f"  CANNOT WRITE TO {folder} -- {reason}")

    if report["unwritable"]:
        lines.append(
            "  Outlines and counts are saved next to each image, so drawing would appear"
        )
        lines.append(
            "  to work and then lose everything. Copy the images somewhere you own first."
        )

    for loser, winner in report["duplicates"]:
        lines.append(f"  same name twice: using {winner}, ignoring {loser}")

    return lines


def main():
    if len(sys.argv) > 1:
        for line in describe(inspect(sys.argv[1])):
            print(line)

        return

    print(f"settings file: {SETTINGS_PATH}")
    print(f"  exists: {SETTINGS_PATH.exists()}")

    stored = load_settings().get("folder")
    print(f"  stored folder: {stored or '(none)'}")

    if stored and not Path(stored).is_dir():
        print("  -- that folder no longer exists, so it will be ignored")

    print(f"\nstarting folder would be: {startup_folder().resolve()}\n")

    for line in describe(inspect(startup_folder())):
        print(line)


if __name__ == "__main__":
    main()
