"""Which sections are being counted by a process this window did not start.

Alice, 2026-08-20: *"if its still counting while closed, when i open it i can't tell which one
is being counted still so can you change their status to counting"*.

THE GAP THIS FILLS. Everywhere else in the app, status comes from the files -- app_state reads
which sidecars exist and that IS the truth, which is what makes the app impossible to confuse
by editing a folder behind its back. But a count in progress has written nothing yet. It is
forty minutes of work that exists only as a process, so the folder cannot see it, and neither
can a window that has just opened.

That is not a hypothetical. Measured on this machine at 01:38 on 2026-08-20:

    PID 11062, ppid 1, running 16:58, count_cells.py "Gre595_RH_3-1-4_NCM_Slide 1.TIF"

ppid 1 means its parent is gone: the app that started it exited and launchd adopted the count,
which is exactly the arrangement count_cells.ignore_a_lost_listener was written to allow. It
worked. But the app she opened afterwards showed that section as "ready to count", and pressing
Count would have started a SECOND count of the same image -- two forty-minute jobs competing for
the same 8 GB and racing to write the same file. So this is not only a label, it is a guard.

HOW IT LOOKS: at the process list, once, through psutil. Not a lock file, which is the other
obvious design and the wrong one -- a lock file has to be deleted by whoever made it, and the
case that matters here is precisely the one where the app that made it died. A process either
exists or it does not, and nothing has to remember to tidy up.

psutil RATHER THAN `ps`, 2026-09-13. `ps` does not exist on Windows, and process_list caught the
resulting OSError and returned nothing -- so on Windows this guard silently found no counts ever,
which is the failure mode a guard must not have: quiet, and indistinguishable from "all clear".
The line format below is still exactly what `ps -Awwo pid=,etime=,command=` printed, because that
is this module's internal seam and the parsing is what main() checks against fixed input.

MATCHED ON THE FILE NAME, not the stem. Stems can be prefixes of each other (`...1-1-1` and
`...1-1-10`), and a substring test on the shorter one would happily match the longer one's
command line and label the wrong section. The file name ends in `.TIF`, so it cannot be a
prefix of another file name.

Usage:
  python app_running.py           # check the parsing, then look at this machine
"""

import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover -- a dependency, but not one worth failing to start over
    psutil = None

# Everything that counts cells goes through this script, whether it was started by the app, by
# a terminal, or by an app that has since quit. One name to look for.
COUNTER = "count_cells.py"


def elapsed_text(seconds):
    """`seconds` written the way `ps` writes elapsed time: [[dd-]hh:]mm:ss.

    Exists so process_list can keep producing the format minutes_running already reads, rather
    than every caller learning a new one. Checked against minutes_running in main(), which is the
    only thing that makes two representations of one fact safe to have.
    """
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 24 * 60 * 60)
    hours, rest = divmod(rest, 60 * 60)
    minutes, seconds = divmod(rest, 60)

    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def process_list():
    """Every process as "<pid> <elapsed> <command line>" lines. Empty on any failure.

    EVERY process, not just this user's, for the same reason `ps -A` was used before: a count
    started from a terminal and a count started by the app are the same thing and both must be
    found. Where the operating system refuses to say what another user's process is running,
    psutil raises and that process is skipped -- which is a real limit, but a count started by a
    different user account is not a case this app has ever had.

    THE WHOLE COMMAND LINE, never a truncated one. `ps` truncates by default and needed `-ww` to
    stop; psutil hands back the arguments as a list and the image path is the last of them, which
    is the part that must survive -- matching is done on the file name.

    Failure returns "" rather than raising: not knowing about a background count is a worse label,
    not a broken app, and this is called on a timer every 15 seconds.

    FAST ENOUGH TO LEAVE ALONE, measured 2026-09-13 on 377 processes: 8 ms warm, 89 ms on the first
    call while psutil fills its caches. It runs on the Qt thread, so that was worth checking. Asking
    psutil for `name` first and reading the command line only for python-named processes -- the
    obvious optimisation -- measured SLOWER (13 ms), because the extra field costs more than the
    command lines it skips. Left simple.
    """
    if psutil is None:
        return ""

    now = time.time()
    lines = []

    try:
        for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                about = process.info
                command = " ".join(about["cmdline"] or ())
            except (psutil.Error, TypeError):
                continue

            if not command:
                continue

            started = about["create_time"] or now
            lines.append(f"{about['pid']} {elapsed_text(now - started)} {command}")
    except psutil.Error:
        return ""

    return "".join(f"{line}\n" for line in lines)


def minutes_running(etime):
    """`ps` elapsed time -- [[dd-]hh:]mm:ss -- as whole minutes. -1 if it cannot be read.

    Shown to a person deciding whether to wait, so minutes is the right resolution: a count
    takes 35-45 of them and the seconds are noise.
    """
    text = etime.strip()
    days = 0

    if "-" in text:
        left, _, text = text.partition("-")

        try:
            days = int(left)
        except ValueError:
            return -1

    parts = text.split(":")

    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return -1

    if len(numbers) == 2:
        hours, (minutes, seconds) = 0, numbers
    elif len(numbers) == 3:
        hours, minutes, seconds = numbers
    else:
        return -1

    return days * 24 * 60 + hours * 60 + minutes


def counting_now(sections, output=None):
    """{stem: {"pid": int, "minutes": int}} for sections a count_cells.py process is working on.

    `output` is there so the parsing can be checked against a fixed process list rather than
    whatever happens to be running -- see main(). Default is this machine, right now.

    Includes counts this window started. Sorting that out is the caller's job, because only the
    caller knows what it started, and a function that quietly hid them would make the app's own
    count vanish from the list if it were ever used for the display.
    """
    lines = [
        line for line in (output if output is not None else process_list()).splitlines()
        if COUNTER in line
    ]

    if not lines:
        return {}

    found = {}

    for section in sections:
        name = Path(section["path"]).name

        for line in lines:
            if name not in line:
                continue

            pieces = line.split(None, 2)

            if len(pieces) < 3:
                continue

            try:
                pid = int(pieces[0])
            except ValueError:
                continue

            found[section["stem"]] = {"pid": pid, "minutes": minutes_running(pieces[1])}

            break

    return found


def describe(found, minutes_each=40):
    """Lines for the log. Says what it is, how long it has been going, and what to do.

    "Press Refresh" is in there because nothing tells this window when an outside count lands
    -- it polls, and the poll is 15 seconds, so a person who has just watched Activity Monitor
    go quiet should not have to wonder why the number has not appeared yet.
    """
    if not found:
        return []

    lines = [
        f"{len(found)} count{'s' if len(found) > 1 else ''} already running, started outside "
        "this window:"
    ]

    for stem, about in found.items():
        minutes = about["minutes"]
        so_far = f"{minutes} minutes in" if minutes >= 0 else "running"
        left = ""

        if 0 <= minutes < minutes_each:
            left = f", so roughly {minutes_each - minutes} to go"

        lines.append(f"  {stem} -- {so_far}{left} (process {about['pid']})")

    lines.append(
        "They keep going whether this window is open or not, and save their numbers when they "
        "finish. This window checks every 15 seconds and will say when one lands."
    )

    return lines


def main():
    # A fixed process list, so what is being checked is the parsing and not the machine. Real
    # `ps -Awwo pid=,etime=,command=` output, including the two traps: a path with spaces in
    # it, and a stem that is a prefix of another section's stem.
    output = (
        "  501       01:23 /usr/bin/python3 app.py\n"
        "11062       16:58 /Library/Frameworks/Python.framework/Versions/3.12/Resources/"
        "Python.app/Contents/MacOS/Python count_cells.py /Users/a/cell counting/"
        "Gre595_RH_3-1-4_NCM_Slide 1.TIF --points\n"
        "12995    01-02:03:04 /usr/bin/python3 draw_regions.py /Users/a/Bird_1-1-1.TIF\n"
        " 9001       02:10 /usr/bin/python3 count_cells.py /Users/a/Bird_1-1-10.TIF --points\n"
    )

    sections = [
        {"stem": "Gre595_RH_3-1-4_NCM_Slide 1", "path": "/Users/a/cell counting/"
         "Gre595_RH_3-1-4_NCM_Slide 1.TIF"},
        {"stem": "Bird_1-1-1", "path": "/Users/a/Bird_1-1-1.TIF"},
        {"stem": "Bird_1-1-10", "path": "/Users/a/Bird_1-1-10.TIF"},
        {"stem": "Bird_2-2-2", "path": "/Users/a/Bird_2-2-2.TIF"},
    ]

    found = counting_now(sections, output)

    assert set(found) == {"Gre595_RH_3-1-4_NCM_Slide 1", "Bird_1-1-10"}, found
    assert found["Gre595_RH_3-1-4_NCM_Slide 1"] == {"pid": 11062, "minutes": 16}, found
    assert found["Bird_1-1-10"] == {"pid": 9001, "minutes": 2}, found

    # The prefix trap: Bird_1-1-1 is a prefix of Bird_1-1-10, and only the longer one is being
    # counted. Matching on stems instead of file names would have labelled both.
    assert "Bird_1-1-1" not in found, "a stem prefix was mistaken for the section itself"

    # Being drawn is not being counted. Only count_cells.py lines are looked at.
    assert minutes_running("01-02:03:04") == 26 * 60 + 3
    assert minutes_running("16:58") == 16
    assert minutes_running("2:03:04") == 123
    assert minutes_running("what") == -1

    assert counting_now(sections, "") == {}
    assert counting_now([], output) == {}

    # elapsed_text and minutes_running are two views of one number, so they are checked against
    # each other rather than against my idea of what ps prints.
    assert elapsed_text(16 * 60 + 58) == "16:58"
    assert elapsed_text(2 * 3600 + 3 * 60 + 4) == "2:03:04"
    assert elapsed_text(26 * 3600 + 3 * 60 + 4) == "1-02:03:04"
    assert elapsed_text(-5) == "00:00"

    for seconds in (0, 59, 60, 3599, 3600, 90061, 400000):
        assert minutes_running(elapsed_text(seconds)) == seconds // 60, seconds

    lines = describe(found)
    assert "2 counts already running" in lines[0], lines
    assert "16 minutes in, so roughly 24 to go" in lines[1], lines
    assert describe({}) == []

    print("background counts: parsing correct\n")

    # And now this machine, for real. TIMED, because this runs on a 15-second timer on the Qt
    # thread: if reading the process list ever costs a noticeable fraction of a second, the window
    # will hitch, and the number should be visible here rather than discovered as a stutter.
    if psutil is None:
        print("psutil is not installed, so background counts cannot be seen at all.")

        return 1

    before = time.monotonic()
    listing = process_list()
    took = time.monotonic() - before

    print(f"read {len(listing.splitlines())} processes in {took * 1000:.0f} ms")

    live = [line for line in listing.splitlines() if COUNTER in line]

    if not live:
        print("Nothing is counting on this machine right now.")
    else:
        print(f"{len(live)} count_cells.py process(es) running right now:")

        for line in live:
            print(f"  {line.strip()[:150]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
