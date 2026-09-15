"""What a batch of counts will actually do, worked out before any of it starts.

Counting one section takes 35-45 minutes and there is no cancel button (see app_jobs.py for
why not). A batch is therefore a commitment of HOURS made with a single click, and the worst
version of this feature is one that starts quietly and reveals what it is doing gradually.
So every question a person would want answered first is answered here, by functions that
touch nothing: no disk, no subprocess, no widgets.

That also makes them checkable from a terminal against invented sections, which is the whole
reason the app's logic lives outside the window. See main().

WHY "ALREADY COUNTED" IS NOT A REASON TO SKIP. It would be easy to leave counted sections
out of a batch automatically, and tempting -- it saves hours. But skipping something a person
explicitly ticked is a surprise, and re-counting is the only way to redo a section whose
count you distrust. Ticking it IS the request. The protection against the real accident
-- selecting everything by mistake -- is the estimate, which the window shows and asks about
before starting, not a silent exclusion.

Usage:
  python app_queue.py     # check the rules against invented sections
"""

# Why a section cannot be counted at all. Only structural reasons appear here: things that
# would make count_cells.py refuse or fail, not things a person might merely not have meant.
NO_REGIONS = "no outlines drawn"
NO_DORSAL = "no dorsal marker, so NCM cannot be split"

# Why a section cannot be ADDED to a queue that is already running. Not reasons it cannot be
# counted -- reasons it is already going to be. Separate constants because the message a
# person needs is different: "you cannot" versus "you already did".
ALREADY_RUNNING = "already being counted right now"
ALREADY_QUEUED = "already waiting in the queue"

# And the one that is invisible to this window: a count_cells.py process started by a terminal,
# or by an app that has since quit, working on that section right now. app_running.counting_now
# finds them. Refused for the same reason as the two above and more urgently -- the two counts
# would race to write the same file, and neither window would know the number it displayed was
# the loser.
COUNTING_OUTSIDE = "already being counted by a job started outside this window"

# Measured, not guessed: 43m46s for a 15 Mpx section and about 35 minutes for a smaller one
# while another job competed. Used only for the estimate shown before a batch starts, so
# being roughly right is enough -- and a range is quoted rather than this single number.
MINUTES_EACH = 40


def reason_to_skip(section):
    """Why this section cannot be counted, or None if it can.

    The order matches app_state.status_for: outlines first, because drawing them is what
    prompts the dorsal click, so "no outlines" is the more fundamental complaint and the one
    worth reporting when both are true.
    """
    if not section["has_regions"]:
        return NO_REGIONS

    if not section["has_dorsal"]:
        return NO_DORSAL

    return None


def reason_not_now(section, running=None, queued=(), outside=()):
    """Why this section cannot be counted AT THIS MOMENT, or None if it can.

    Two kinds of reason, in that order. First the structural ones from reason_to_skip -- things
    that would make count_cells.py refuse whenever it was asked. Then the three "it is already
    happening" ones, which are true only right now and only because of what is running.

    Structural first, deliberately. A section with no outlines that is somehow also being
    counted must report the outlines: that is the thing she has to fix, and "already counting"
    would be a claim about a job that could not have started.

    ONE FUNCTION so plan() and plan_addition() cannot drift apart. They differ only in which of
    these arguments the caller knows about.
    """
    reason = reason_to_skip(section)

    if reason:
        return reason

    if section["stem"] == running:
        return ALREADY_RUNNING

    if section["stem"] in queued:
        return ALREADY_QUEUED

    if section["stem"] in outside:
        return COUNTING_OUTSIDE

    return None


def plan(sections, outside=()):
    """Split sections into (to_count, skipped).

    `skipped` is a list of (section, reason) pairs rather than just sections, because "three
    were left out" is not useful and "three were left out because they have no outlines yet"
    tells a person what to do next.

    Order is preserved from the input, so a batch runs in the order shown in the list. Not
    sorted by size or by anything clever: predictable beats optimal when the person is going
    to walk away and come back to a half-finished queue.
    """
    to_count = []
    skipped = []

    for section in sections:
        reason = reason_not_now(section, outside=outside)

        if reason:
            skipped.append((section, reason))
        else:
            to_count.append(section)

    return to_count, skipped


def plan_addition(sections, running=None, queued=(), outside=()):
    """Split sections into (to_add, skipped) for a queue that is ALREADY RUNNING.

    Alice, 2026-08-20: *"i thought i was able to make a count cells queue but when i count
    cells for one image i can't press count cells for another image"*. She was right about
    what the feature should be and wrong about what it was: the queue existed, but only as a
    selection made before anything started. Deciding the whole night's work up front is the
    part worth removing -- what you want to count next is usually clearer after the first one
    is under way.

    Same structural rules as plan(), plus the two that only exist once something is running.
    Adding a section that is already in the queue would count it twice, which wastes forty
    minutes and produces two files racing to be the answer -- so it is refused, not silently
    tolerated. Adding the one being counted RIGHT NOW is refused for the same reason, and it
    is the easy mistake to make, since that section is the one highlighted in the list.

    `outside` is the stems being counted by a process this window did not start -- see
    app_running.py. Refused too, and it is the only one of the three that a person could not
    have worked out by looking at the window.

    Deliberately NOT a reason to refuse: already counted. Same argument as plan() -- ticking
    it is the request, and re-counting is the only way to redo a count you distrust.
    """
    to_add = []
    skipped = []

    for section in sections:
        reason = reason_not_now(section, running, set(queued), outside)

        if reason:
            skipped.append((section, reason))
        else:
            to_add.append(section)

    return to_add, skipped


def estimate(count, minutes_each=MINUTES_EACH):
    """How long a batch of `count` sections will take, in words.

    A RANGE, always, and never a finishing time. Nothing in the pipeline knows its own
    progress, the per-section duration varies with region size, and a single number invites
    someone to plan around it. Rounded to hours above 90 minutes because "7 hours" is the
    useful precision for a decision to leave the laptop running overnight -- "6 h 40 m" is
    false confidence.
    """
    if count <= 0:
        return "no time at all"

    low = count * (minutes_each - 5)
    high = count * (minutes_each + 5)

    if high <= 90:
        return f"about {low}-{high} minutes"

    low_hours = round(low / 60)
    high_hours = round(high / 60)

    # Collapsed to one number when both ends round the same way, because "about 2-2 hours"
    # reads as a bug rather than as a range. Three sections is 1h45m-2h15m; "about 2 hours"
    # is the true statement, and a spurious range would suggest a precision that is not there.
    if low_hours == high_hours:
        return f"about {low_hours} hours"

    return f"about {low_hours}-{high_hours} hours"


def describe(to_count, skipped):
    """The batch as lines for a person to read, before it starts. Worst news first.

    Returned rather than printed so the window can put them in its log and a terminal can
    print them, without this module knowing which it is talking to -- the same arrangement as
    app_folder.describe.
    """
    lines = []

    for section, reason in skipped:
        lines.append(f"  SKIPPING {section['stem']} -- {reason}")

    if not to_count:
        lines.append("Nothing to count in this selection.")

        return lines

    lines.append(
        f"Counting {len(to_count)} section{'s' if len(to_count) > 1 else ''}, "
        f"{estimate(len(to_count))}:"
    )

    for index, section in enumerate(to_count, start=1):
        lines.append(f"  {index}. {section['stem']}")

    if len(to_count) > 1:
        # Said explicitly because both are easy to assume wrongly, and both matter to
        # someone deciding whether to walk away.
        lines.append(
            "They run one at a time -- two counts at once would exhaust memory on this "
            "machine. There is no way to stop the batch once it starts, other than quitting "
            "the app, which loses only the section in progress."
        )

    return lines


def describe_addition(added, skipped, waiting):
    """What joining a running queue did, as lines to read. Worst news first.

    `added` is what the runner actually took, not what was asked for -- the runner is the one
    that owns the queue, so a message built from the request could claim a section was added
    that the runner refused. `waiting` is how many are now behind the running one.
    """
    lines = []

    for section, reason in skipped:
        lines.append(f"  NOT ADDED {section['stem']} -- {reason}")

    if not added:
        lines.append("Nothing added to the queue.")

        return lines

    lines.append(
        f"Added {len(added)} section{'s' if len(added) > 1 else ''} to the queue. "
        f"{waiting} now waiting behind the one being counted, "
        f"{estimate(waiting)} of queue after it finishes:"
    )

    for index, section in enumerate(added, start=1):
        lines.append(f"  {index}. {section['stem']}")

    return lines


def fake_section(stem, regions=True, dorsal=True, counts=False):
    """A section dict for checking the rules without any files.

    Here rather than in main() so the shape of a section is written down in one place -- it
    is a plain dict built by app_state.find_sections, so nothing else documents it.
    """
    return {
        "stem": stem,
        "path": f"{stem}.TIF",
        "has_regions": regions,
        "has_dorsal": dorsal,
        "has_counts": counts,
        "status": "invented",
    }


def main():
    ready = fake_section("ready")
    counted = fake_section("counted", counts=True)
    no_regions = fake_section("no_regions", regions=False)
    no_dorsal = fake_section("no_dorsal", dorsal=False)

    assert reason_to_skip(ready) is None
    assert reason_to_skip(counted) is None, "already counted must NOT be a reason to skip"
    assert reason_to_skip(no_regions) == NO_REGIONS
    assert reason_to_skip(no_dorsal) == NO_DORSAL

    # Both missing reports the outlines, because that is the one to fix first.
    assert reason_to_skip(fake_section("neither", regions=False, dorsal=False)) == NO_REGIONS

    to_count, skipped = plan([ready, no_regions, counted, no_dorsal])

    assert [s["stem"] for s in to_count] == ["ready", "counted"], "order must be preserved"
    assert [(s["stem"], r) for s, r in skipped] == [
        ("no_regions", NO_REGIONS),
        ("no_dorsal", NO_DORSAL),
    ]

    assert plan([]) == ([], [])

    # ---- adding to a queue that is already running ----
    waiting_already = fake_section("waiting")
    running_now = fake_section("running")

    to_add, skipped = plan_addition(
        [ready, running_now, waiting_already, no_regions, counted],
        running="running",
        queued={"waiting"},
    )

    assert [s["stem"] for s in to_add] == ["ready", "counted"], [s["stem"] for s in to_add]
    assert [(s["stem"], r) for s, r in skipped] == [
        ("running", ALREADY_RUNNING),
        ("waiting", ALREADY_QUEUED),
        ("no_regions", NO_REGIONS),
    ], skipped

    # With nothing running, it is the same answer as plan() -- so the two cannot drift apart.
    assert plan_addition([ready, no_dorsal]) == plan([ready, no_dorsal])

    # A section that cannot be counted is reported for THAT reason even when it is also the
    # one running: "no outlines" is what she has to fix, and "already counting" would be a lie
    # about a section that could not have been started.
    stuck = fake_section("stuck", regions=False)
    _, why = plan_addition([stuck], running="stuck")
    assert why[0][1] == NO_REGIONS, why

    # ---- counted by something this window did not start ----
    elsewhere = fake_section("elsewhere")

    addable, refused = plan_addition([ready, elsewhere], outside={"elsewhere"})
    assert [s["stem"] for s in addable] == ["ready"], addable
    assert refused == [(elsewhere, COUNTING_OUTSIDE)], refused

    # The idle path refuses it too. This is the case that matters: nothing is running HERE, so
    # without `outside` the button would happily start a second count of the same image.
    countable, refused = plan([ready, elsewhere], outside={"elsewhere"})
    assert [s["stem"] for s in countable] == ["ready"], countable
    assert refused == [(elsewhere, COUNTING_OUTSIDE)], refused

    # And a section being counted outside is reported for its structural problem first, on the
    # same argument as the ALREADY_RUNNING case above.
    _, why = plan_addition([no_dorsal], outside={"no_dorsal"})
    assert why[0][1] == NO_DORSAL, why

    lines = describe_addition(to_add, skipped, waiting=3)
    assert lines[0].startswith("  NOT ADDED running"), lines
    assert "Added 2 sections" in lines[3] and "3 now waiting" in lines[3], lines
    assert describe_addition([], skipped, waiting=1)[-1] == "Nothing added to the queue."

    assert estimate(0) == "no time at all"
    assert estimate(1) == "about 35-45 minutes"
    assert estimate(2) == "about 70-90 minutes"
    assert estimate(3) == "about 2 hours", estimate(3)
    assert estimate(20) == "about 12-15 hours", estimate(20)

    # No estimate may ever read as a collapsed range like "2-2 hours".
    for count in range(1, 41):
        text = estimate(count)
        assert "-" not in text or text.split()[1].split("-")[0] != text.split()[1].split("-")[1], (
            f"{count} sections gave {text!r}"
        )

    print("queue rules: all cases correct\n")

    for line in describe(*plan([ready, counted, no_regions])):
        print(line)

    print()

    for count in (1, 2, 3, 4, 8, 16, 20):
        print(f"  {count:2d} section{' ' if count == 1 else 's'} -> {estimate(count)}")


if __name__ == "__main__":
    main()
