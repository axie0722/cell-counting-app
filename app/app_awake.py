"""Stop the computer falling asleep while a count is running.

Alice, 2026-08-20: *"i added images to queue to count overnight but they didn't count"*. Nothing
in the app had failed. The machine had:

  * fallen asleep over and over on battery -- the pmset log for that night is a wall of
    "Entering Sleep state ... Using Batt (Charge:59%)"
  * and then restarted, at 00:15 and again at 09:57

A SLEEPING COMPUTER DOES NOT COUNT. The counting process is not killed by sleep, it is frozen: the
CPU stops, the process keeps its memory, and it carries on from exactly where it was when the
machine wakes -- the same mechanism app_pause.py uses on purpose. So eight hours of "overnight"
produced about twenty minutes of counting, and the queue looked as though it had silently done
nothing.

ALL THREE SYSTEMS, since 2026-09-13. Alice: *"i want this to run on any computer not just mac"*.
`caffeinate` is a macOS program, so on Windows and Linux keep_awake used to catch the OSError and
return None -- which is the honest thing for a function to do and a terrible thing for this
particular function to do quietly, because the failure it hides is "the overnight run produced
twenty minutes of work" and there was nothing on screen to say so. See cannot_hold and describe.

WHY -w RATHER THAN A HANDLE WE HAVE TO RELEASE. `caffeinate -i -w PID` waits for that pid and then
exits, releasing the assertion by dying. Nothing has to remember to let go: not on a crash, not on
a force-quit, not when the app is killed from the terminal and leaves the count running. The one
thing that must be true -- the machine stays awake for exactly as long as the count lives -- is true
by construction rather than by bookkeeping. This is the same argument as reading the process list
instead of writing a lock file: tie the fact to something the OS already maintains.

THE OTHER TWO SYSTEMS ARE BUILT TO COPY THAT, because it is the property worth keeping:

  * Linux: `systemd-inhibit` holds an idle/sleep lock for as long as the command it runs, so it is
    given a tiny Python helper that waits for the count's pid. Lock released by the helper exiting.
  * Windows: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` is the equivalent call,
    and the requirement it sets belongs to the THREAD that made it -- Windows drops it when that
    thread goes away. So the same helper sets it and then waits for the pid, and dying releases it.
    Calling it from the app's own Qt thread would have been one line and wrong: the hold would last
    as long as the APP rather than as long as the count, so a count that finished at 2 a.m. would
    leave the machine awake until morning, and quitting the app during a count would drop the hold
    while the count was still going. Both errors, in opposite directions, from one shortcut.

WHAT NONE OF THEM FIX: THE LID. Clamshell sleep is not something a process is allowed to veto on
any of the three systems, and no flag, entitlement or tool changes that. An honest fix is therefore
two things: this module, and telling her to leave the lid open. That is describe()'s whole job.

Usage:
  python app/app_awake.py          # what is holding the machine awake right now, then a real check
"""

import shutil
import subprocess
import sys
import time

MAC = sys.platform == "darwin"
WINDOWS = sys.platform.startswith("win")
LINUX = sys.platform.startswith("linux")

# -i is "prevent idle SYSTEM sleep". Not -d (that only keeps the display on, which wastes battery
# and helps nothing) and not -s (which only applies on AC power -- the night this was written she
# was on battery, which is exactly when it was needed).
FLAG = "-i"

# THE HELPER, for Windows and Linux. Passed with -c rather than shipped as a file because it is not
# a program anyone runs on purpose: it exists only as something for a sleep lock to be attached to,
# and a file called keep_awake_helper.py in the folder would invite the question of what it does.
#
# `while pid_exists` rather than psutil's wait(): waiting on a process that is not our child polls
# anyway, and this way the same four lines work whether the count was started by this app or by a
# terminal somewhere else -- which is the case ByPid exists for in app_pause.py. Five seconds is
# chosen to be cheap; the cost of being late is that the machine is held awake five seconds longer
# than it needed to be.
HELPER = (
    "import sys, time\n"
    "import psutil\n"
    "pid = int(sys.argv[1])\n"
    "if sys.platform.startswith('win'):\n"
    "    import ctypes\n"
    # ES_CONTINUOUS (0x80000000) means "until I say otherwise" rather than "nudge the idle timer
    # once"; ES_SYSTEM_REQUIRED (0x1) is the system, not the display -- the same choice as -i.
    "    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)\n"
    "while psutil.pid_exists(pid):\n"
    "    time.sleep(5)\n"
)


def cannot_hold():
    """Why this computer cannot be held awake, or "" if it can. A sentence fit to show someone.

    CHECKED RATHER THAN ASSUMED, because the answer is not "which system is this" on Linux: a
    machine without systemd, or a container, has no sleep lock to take and also very often no
    ability to sleep in the first place. Saying "leave it plugged in and watch it" to someone whose
    server cannot sleep is a small annoyance; saying nothing to someone whose laptop can is how the
    overnight run at the top of this file happened.
    """
    if MAC:
        if shutil.which("caffeinate") is None:
            return "this Mac has no caffeinate, so nothing here can stop it idle-sleeping"

        return ""

    if WINDOWS:
        return ""

    if LINUX:
        if shutil.which("systemd-inhibit") is None:
            return ("this computer has no systemd-inhibit, so nothing here can stop it sleeping "
                    "-- if it sleeps, a count stops until it wakes")

        return ""

    return (f"on {sys.platform} there is no way for this app to stop the computer sleeping "
            "-- if it sleeps, a count stops until it wakes")


def awake_command(pid):
    """The command that holds this computer awake until `pid` exits. None if there is no such thing.

    One function so the three systems are side by side and the shape they share is visible: every
    one of them is a process whose LIFETIME is the hold, so keep_awake below is the same three lines
    everywhere and nothing releases anything.
    """
    if cannot_hold():
        return None

    if MAC:
        return ["caffeinate", FLAG, "-w", str(pid)]

    waiter = [sys.executable, "-c", HELPER, str(pid)]

    if WINDOWS:
        # The helper takes the hold itself -- see HELPER.
        return waiter

    return [
        "systemd-inhibit",
        # idle is the one that matters (an unattended count is an idle machine); sleep covers a
        # deliberate suspend, which is worth blocking too while forty minutes of work is in flight.
        "--what=idle:sleep",
        "--who=cell counting",
        "--why=counting cells",
        # block, not delay: delay only postpones the sleep by a few seconds, which is no use to a
        # job measured in hours.
        "--mode=block",
        *waiter,
    ]


def keep_awake(pid):
    """Hold the computer awake until process `pid` exits. Returns the helper's pid, or None.

    Never raises. Failing to launch the helper is worth reporting but not worth losing a count
    over: without it the count still runs, it just may be interrupted by sleep.
    """
    command = awake_command(pid)

    if command is None:
        return None

    try:
        awake = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # NO CONSOLE WINDOW ON WINDOWS. Launching a python.exe with no window flag pops a black
            # console box on screen for the whole life of the count, which looks exactly like
            # something has gone wrong. 0 elsewhere, where the flag does not exist.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, ValueError):
        return None

    return awake.pid


def holders():
    """Every process holding a "do not sleep" assertion right now, as (pid, name) pairs.

    Read from pmset rather than remembered, for the usual reason: two app windows and a terminal
    can each have started a count, and only the machine knows what all of them did.

    MACOS ONLY, and empty everywhere else rather than wrong. Windows has no equivalent listing at
    all -- `powercfg /requests` needs administrator rights and reports drivers, not our thread's
    requirement -- and `systemd-inhibit --list` reports locks by who took them, not by pid. Both
    could be parsed for a report; neither is worth it, because nothing but this file's own main()
    ever asked. What the app needs is cannot_hold(), which is a fact about the system rather than a
    reading of it.
    """
    if not MAC:
        return []

    try:
        listing = subprocess.run(
            ["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    found = []

    for line in listing.splitlines():
        # The lines that name a holder look like:
        #   pid 4822(caffeinate): [0x...] 00:01:52 PreventUserIdleSystemSleep named: "..."
        if "PreventUserIdleSystemSleep" not in line or "pid " not in line:
            continue

        rest = line.split("pid ", 1)[1]
        number = rest.split("(", 1)[0].strip()
        name = rest.split("(", 1)[1].split(")", 1)[0] if "(" in rest else "?"

        if number.isdigit():
            found.append((int(number), name))

    return found


def describe(on_battery=None):
    """Lines for a person about to leave a count running unattended. Worst news first.

    ON SCREEN, not printed: the app hides stdout, so anything a person needs to act on has to be
    something the window can say.

    THE REFUSAL COMES FIRST WHEN THERE IS ONE, ahead of even the lid. On a system with no sleep lock
    the lid is no longer the only risk -- the idle timer is -- and a person who reads "leave the lid
    open" and nothing else would reasonably conclude that the rest was handled.

    WHICH IS WHY THE REASSURING LINE IS LAST AND CONDITIONAL. It used to be part of the lid sentence
    ("the Mac is kept awake while a count runs, but closing the lid sleeps it anyway"), and that
    sentence cannot be told on a computer where the first half is false.
    """
    lines = []
    refusal = cannot_hold()

    if refusal:
        lines.append(f"NOTE: {refusal}.")
        lines.append("Set it never to sleep in the system's power settings before leaving it.")

    lines.append("LEAVE THE LID OPEN. Closing it sleeps the computer whatever is running --")
    lines.append("no program on any system can prevent that -- and a sleeping computer does not "
                 "count.")

    if on_battery:
        lines.append("Plug it in too: on battery it will sleep sooner and the count is hours long.")

    if not refusal:
        lines.append("Idle sleep is looked after for as long as the count lives. The lid is not.")

    return lines


def on_battery():
    """True if the computer is running off its battery, None if it cannot be told.

    Advisory only -- it adds one line to describe() -- so every branch answers None rather than
    raising, and a desktop with no battery at all reads as False (it is on mains) or None (nothing
    said), both of which lead to the same advice.
    """
    if MAC:
        try:
            state = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

        if "'AC Power'" in state:
            return False

        if "'Battery Power'" in state:
            return True

        return None

    if WINDOWS:
        # GetSystemPowerStatus fills a struct; ACLineStatus is its first byte -- 0 offline (so, on
        # battery), 1 online, 255 unknown. Read through ctypes rather than by running a program,
        # because the alternatives (wmic, powershell) are a subprocess and a second of startup for
        # one byte, and wmic is being removed from Windows.
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = Status()

        try:
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                return None
        except (OSError, AttributeError):
            return None

        if status.ACLineStatus == 0:
            return True

        if status.ACLineStatus == 1:
            return False

        return None

    # Linux, and anything else with the standard power supply files. An "online" mains supply is
    # the direct answer; a machine with no mains supply listed and no battery either says nothing.
    from pathlib import Path

    mains = None

    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        try:
            kind = (supply / "type").read_text().strip()
            online = (supply / "online").read_text().strip()
        except OSError:
            continue

        # Any mains supply that is online means we are on mains; None until one has been seen at all,
        # which is a machine that cannot tell us rather than one on battery.
        if kind == "Mains":
            mains = bool(mains) or online == "1"

    return None if mains is None else not mains


def main():
    refusal = cannot_hold()

    print(f"this is {sys.platform}, and " + (refusal or "it can be held awake"))
    print("\nholding the machine awake right now:")

    found = holders()

    for pid, name in found:
        print(f"  pid {pid} ({name})")

    if not found:
        print("  nothing" if MAC else "  (only macOS can be asked; see holders)")

    power = on_battery()
    print(f"\non battery: {power}")

    # THE HELPER'S WAITING HALF, CHECKED ON EVERY SYSTEM INCLUDING THE ONE THAT DOES NOT USE IT.
    # HELPER is what ties the hold to the count on Windows and Linux, and neither of those can be
    # run here -- but the half that decides WHEN TO LET GO is the same four lines everywhere, so a
    # Mac can check it. An imperfect check that runs on the machine the code was written on beats a
    # perfect one that only runs on a computer nobody here owns.
    brief = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    helper = subprocess.Popen([sys.executable, "-c", HELPER, str(brief.pid)])

    brief.wait()

    assert helper.wait(timeout=30) == 0, "the Windows and Linux helper did not exit with its count"

    print("\nhelper: lets go when the count it was given ends")

    if refusal:
        for line in describe(power):
            print(f"  {line}")

        print("\nkeeping this computer awake: not possible here, and it says so")

        return 0

    # A REAL HOLD, tied to a process that ends on its own, and then gone. Two things are checked and
    # only one of them can be checked everywhere:
    #
    #   * that the hold is really taken -- pmset can be asked, so on macOS it is asked;
    #   * that it is RELEASED WHEN THE COUNT ENDS, which is this design's whole claim and is
    #     observable on every system, because the release IS the helper exiting.
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"])
    awake = keep_awake(sleeper.pid)

    assert awake is not None, f"the helper did not start: {awake_command(sleeper.pid)}"

    if MAC:
        assert any(pid == awake for pid, _ in holders()), "the assertion is not being held"

    print(f"\nheld by pid {awake} while {sleeper.pid} runs")

    sleeper.wait()

    # It notices the pid has gone within a second on macOS, within HELPER's five-second poll
    # elsewhere. Fifteen seconds of patience covers both.
    import psutil

    def released(pid):
        """True once the helper is holding nothing. A ZOMBIE COUNTS, and that is worth a note.

        keep_awake drops the Popen object on purpose -- the whole design is that nothing has to
        remember anything -- so on POSIX an exited helper stays in the process table as a zombie
        until the subprocess module reaps it, which it does the next time anything here launches a
        process. It runs no code and holds no assertion in that state, which the pmset check below
        confirms independently on this machine. Windows has no such state and drops the pid.
        """
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.Error:
            return True

    for _ in range(30):
        if released(awake):
            break

        time.sleep(0.5)
    else:
        raise AssertionError("the helper outlived the process it was waiting for")

    if MAC:
        assert not any(pid == awake for pid, _ in holders()), "the assertion outlived its count"

    print("released when the process ended")

    for line in describe(on_battery=True):
        print(f"  {line}")

    print("\nkeeping this computer awake: all cases correct")

    return 0


if __name__ == "__main__":
    sys.exit(main())
