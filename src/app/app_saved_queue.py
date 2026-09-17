"""The counting queue, remembered across a quit.

Alice, 2026-08-20: *"since when i reopened it i had to add all the images to queue again"*.

WHAT WAS LOST AND WHY. Closing the app offers to leave the RUNNING count going -- it is a
separate process and survives on its own. The queue behind it does not: it is a list inside the
window, so quitting deleted it, and eight sections picked one at a time had to be picked again.
The work lost is not computation, it is choosing, and choosing is the part with her judgement in
it.

WHAT THIS IS NOT. It does not restart anything by itself. A queue is hours of machine time, and
an app that resumed a five-hour batch because it was opened to look up one number would be a
worse bug than the one being fixed. This remembers the LIST and offers it; starting stays a
click.

WHAT IT DELIBERATELY DROPS on the way back:
  * sections no longer in the folder                  -- the queue outlived the file
  * sections that cannot be counted                   -- outlines deleted since, say
  * sections a count is working on right now           -- including a count left running when
    the app closed, which is the single most likely thing to be in a restored queue
  * sections counted since the queue was saved         -- the count it was waiting for HAPPENED,
    which is what "leave it running" means

That last one is why the save carries a timestamp. Without it, the section that was counting when
the app closed comes back at the top of the offered queue, and accepting spends forty minutes
recomputing a file that is already on disk.

THE FOLDER IS PART OF THE QUEUE. Stems are only unique within a folder, so a queue saved in one
folder is not offered in another. Restoring "Gre595_RH_3-1-4" against a different folder's
identically named section would count the wrong image.

Usage:
  python app_saved_queue.py     # what is remembered, and check the rules
"""

import sys
import time
from pathlib import Path

import app_folder
import app_queue
import ps6

# The key inside ~/.ps6_cell_counter.json. One settings file for the app, not one per feature.
KEY = "queue"

# Why a remembered section is not being offered again.
GONE = "no longer in this folder"
COUNTED_SINCE = "it has been counted since"
COUNTING_NOW = "it is being counted right now"


def remember(folder, stems, now=None):
    """Store the sections still waiting to be counted. True if it was written.

    Called on every change to the queue rather than only on the way out, because the way out is
    not always taken: a power cut, a force-quit and a SIGKILL all skip closeEvent, and those are
    exactly the moments a queue is long.

    An EMPTY list is stored as an empty list, not skipped. Finishing a batch has to be able to
    clear the offer, or the next launch keeps proposing work that is already done.
    """
    return app_folder.store(
        **{
            KEY: {
                "folder": str(Path(folder).expanduser().resolve()),
                "stems": list(stems),
                "saved_at": time.time() if now is None else now,
            }
        }
    )


def remembered(folder):
    """The stored queue for this folder as (stems, saved_at). ([], 0) if there is none.

    Never raises and never trusts what it reads. A settings file can be hand-edited, half
    written, or left by an older version of the app, and the right answer to anything
    unrecognisable is "no queue was remembered".
    """
    stored = app_folder.load_settings().get(KEY)

    if not isinstance(stored, dict):
        return [], 0

    try:
        same = Path(stored.get("folder", "")) == Path(folder).expanduser().resolve()
    except (OSError, ValueError, TypeError):
        return [], 0

    if not same:
        return [], 0

    stems = stored.get("stems")

    if not isinstance(stems, list):
        return [], 0

    saved_at = stored.get("saved_at")

    return [str(stem) for stem in stems], saved_at if isinstance(saved_at, (int, float)) else 0


def forget():
    """Drop the remembered queue. Used when a batch finishes and when an offer is declined."""
    return app_folder.store(**{KEY: None})


def counted_since(section, saved_at):
    """Has this section's count file been written since the queue was saved?

    The FILE'S OWN TIMESTAMP, not the has_counts flag, because a section can have an older
    count and still be worth re-counting -- that is the whole reason app_queue refuses to treat
    "already counted" as a reason to skip. What matters here is narrower: did the count this
    queue was waiting for actually happen while the app was shut?
    """
    if not saved_at:
        return False

    counts = Path(ps6.counts_path(section["path"]))

    try:
        return counts.stat().st_mtime > saved_at
    except OSError:
        return False


def restorable(stems, sections, saved_at=0, outside=(), running=None):
    """Split a remembered queue into (to_offer, dropped). Pure; no disk except file times.

    Order comes from the REMEMBERED list, not from the folder, because the order a batch was
    queued in is a decision -- the first section picked is usually the one wanted first.

    `dropped` is (stem, reason) pairs so the log can say what happened to each. A queue that
    silently comes back three sections shorter is worse than no queue: it looks complete.
    """
    by_stem = {section["stem"]: section for section in sections}

    to_offer = []
    dropped = []

    for stem in stems:
        section = by_stem.get(stem)

        if section is None:
            dropped.append((stem, GONE))
            continue

        if stem == running or stem in outside:
            dropped.append((stem, COUNTING_NOW))
            continue

        if counted_since(section, saved_at):
            dropped.append((stem, COUNTED_SINCE))
            continue

        reason = app_queue.reason_to_skip(section)

        if reason:
            dropped.append((stem, reason))
            continue

        to_offer.append(section)

    return to_offer, dropped


def describe(to_offer, dropped):
    """The restored queue as lines to read. Worst news first, like every describe() here."""
    lines = []

    for stem, reason in dropped:
        lines.append(f"  NOT BROUGHT BACK {stem} -- {reason}")

    if not to_offer:
        if dropped:
            lines.append("Nothing left of the queue that was waiting when the app last closed.")

        return lines

    lines.append(
        f"{len(to_offer)} section{'s' if len(to_offer) > 1 else ''} "
        f"{'were' if len(to_offer) > 1 else 'was'} still waiting in the queue when the app last "
        f"closed, {app_queue.estimate(len(to_offer))} of counting:"
    )

    for index, section in enumerate(to_offer, start=1):
        lines.append(f"  {index}. {section['stem']}")

    return lines


def main():
    stored = app_folder.load_settings().get(KEY)
    print(f"settings file: {app_folder.SETTINGS_PATH}")
    print(f"  stored queue: {stored if stored else '(none)'}\n")

    ready = app_queue.fake_section("ready")
    also = app_queue.fake_section("also")
    no_regions = app_queue.fake_section("no_regions", regions=False)
    sections = [ready, also, no_regions]

    # Order is the remembered order, not the folder's.
    to_offer, dropped = restorable(["also", "ready"], sections)
    assert [s["stem"] for s in to_offer] == ["also", "ready"], to_offer
    assert dropped == []

    to_offer, dropped = restorable(["ready", "vanished", "no_regions"], sections)
    assert [s["stem"] for s in to_offer] == ["ready"], to_offer
    assert dropped == [("vanished", GONE), ("no_regions", app_queue.NO_REGIONS)], dropped

    # The section that was counting when the app closed, in both of its forms.
    to_offer, dropped = restorable(["ready", "also"], sections, running="ready")
    assert [s["stem"] for s in to_offer] == ["also"], to_offer
    assert dropped == [("ready", COUNTING_NOW)], dropped

    to_offer, dropped = restorable(["ready"], sections, outside={"ready"})
    assert dropped == [("ready", COUNTING_NOW)], dropped

    # Counted since: a real file, with its modification time set on either side of the save.
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        image = Path(folder) / "Bird_1-1-1.TIF"
        image.write_bytes(b"")

        section = dict(app_queue.fake_section("Bird_1-1-1"), path=str(image))
        counts = Path(ps6.counts_path(str(image)))
        counts.write_text("x,y\n")

        import os

        os.utime(counts, (1000, 1000))

        assert counted_since(section, saved_at=500) is True
        assert counted_since(section, saved_at=2000) is False
        assert counted_since(section, saved_at=0) is False

        to_offer, dropped = restorable(["Bird_1-1-1"], [section], saved_at=500)
        assert dropped == [("Bird_1-1-1", COUNTED_SINCE)], dropped

        # And with no count file at all, nothing is dropped.
        counts.unlink()
        assert counted_since(section, saved_at=500) is False

    lines = describe(*restorable(["ready", "vanished"], sections))
    assert lines[0].startswith("  NOT BROUGHT BACK vanished"), lines
    assert "1 section was still waiting" in lines[1], lines
    assert describe([], []) == []
    assert describe([], [("x", GONE)])[-1].startswith("Nothing left of the queue"), "empty case"

    # Round trip through the real settings file, then put back exactly what was there.
    before = app_folder.load_settings().get(KEY)

    try:
        assert remember("/tmp", ["a", "b"], now=1234)
        assert remembered("/tmp") == (["a", "b"], 1234), remembered("/tmp")

        # A queue saved in one folder is not offered in another.
        assert remembered("/usr") == ([], 0), remembered("/usr")

        assert forget()
        assert remembered("/tmp") == ([], 0)
    finally:
        app_folder.store(**{KEY: before})

    assert app_folder.load_settings().get(KEY) == before, "the stored queue was not restored"

    print("remembered queue: all cases correct")

    return 0


if __name__ == "__main__":
    sys.exit(main())
