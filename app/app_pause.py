"""Hold a count still while a napari window is open, and let it go again afterwards.

Alice, 2026-08-20: *"is it possible for me to draw outlines while cell counting is going on"*.
Until now: no, and on purpose -- app_jobs.JobRunner runs one job at a time because two heavy
jobs on an 8 GB machine finish LATER than the same two run back to back. Measured, not assumed:
a count sharing the machine dropped to 24.5% CPU, and a napari window holding the same section
was swapped out entirely -- 0.0 GB resident, 0.1% CPU -- which from her side is indistinguishable
from a freeze. Alice: *"whats going on with the napari viewer?"* Nothing. It was out of memory.

SO THE COUNT STOPS INSTEAD OF COMPETING. Freezing the counting process holds it where it stands;
letting it go starts it again from exactly there. Nothing is lost and nothing is recomputed, because
the process never finds out it was interrupted -- this is the same mechanism as Ctrl-Z in a shell.

THROUGH psutil, NOT THROUGH SIGNALS, since 2026-09-13. Alice: *"i want this to run on any computer
not just mac"*. SIGSTOP and SIGCONT do not exist on Windows, so `signal.SIGSTOP` was an
AttributeError raised inside a Qt slot -- and PyQt6 aborts the process on an unhandled exception in
a slot, so on Windows this file turned "draw while counting" into "the app disappears". psutil does
the same thing by the same mechanism on POSIX and by NtSuspendProcess on Windows.

THE SECOND BUG THAT FOUND, worse than the first: ByPid.poll asked whether a count was still alive
with `os.kill(pid, 0)`. On POSIX signal 0 delivers nothing and only runs the permission checks, which
is the standard way to ask. On Windows os.kill has no such convention -- it calls TerminateProcess --
so the question "is this count still running?" would have ENDED the count it was asking about, every
15 seconds, on a machine nobody had tested. Portability bugs are not always cosmetic.

WHAT THIS DOES NOT DO IS FREE THE MEMORY. The stopped count still holds its 1.5-2 GB. What
changes is that it stops TOUCHING those pages, so the operating system can page them out once
and leave them out, instead of the two jobs faulting each other's pages back and forth forever.
That thrashing is the thing that made napari look dead, so removing the competition is the fix
even though the footprint is unchanged.

THE PAUSED TIME IS COUNTED, and that is not a nicety. The window's status line shows how long the
current section has been going, and its only job is to answer "is this wedged?" -- by comparing
against the usual 35-45 minutes. Draw for twenty minutes without subtracting it and the clock
reads 55m on a count that has done 35m of work, which is precisely the wrong answer.

`process` here is anything with `poll()` and either a `pid` or its own `freeze`/`thaw`, so the rules
can be checked against an invented one. See freeze() and main().

IT ALSO WORKS ON A COUNT THIS APP DID NOT START -- see ByPid. Alice, 2026-08-20: *"why isnt my
review cells opening"*. It was opening: show_counts.py was alive and in an uninterruptible disk
wait, 2.5 minutes in, while a count started from ANOTHER app window ran at 113% CPU and the machine
sat on 10.1 GB of its 11.3 GB of swap. The pause had nothing to hold, because the window looked for
its own subprocess and there wasn't one. A count is a count; where it was started from is
bookkeeping, not physics.

Usage:
  python app/app_pause.py     # check the rules, first against a fake process then against a real one
"""

import time

# A HARD IMPORT, unlike app_running's guarded one, and the difference is deliberate. There, a missing
# psutil costs a label and the module says so by finding no counts. Here it would cost the freeze
# itself, and a pause that silently does not pause is the exact failure this file exists to prevent:
# two heavy jobs on one machine, which measured SLOWER than running them one after the other. Better
# to refuse to start, in a Terminal window where the reason is readable, than to start and thrash.
import psutil


def freeze(process):
    """Hold `process` still. True if it is now stopped.

    A process object may answer for itself by providing its own `freeze` -- FakeProcess does, so the
    rules in Pause can be checked without a real process to stop. Anything else is expected to carry
    a `pid`, which covers both of the real cases: a Popen this app started, and a ByPid for a count
    it did not.
    """
    if hasattr(process, "freeze"):
        return process.freeze()

    psutil.Process(process.pid).suspend()

    return True


def thaw(process):
    """Let `process` go again. True if it is now running.

    Raises nothing of its own: psutil.NoSuchProcess and psutil.AccessDenied both mean "not going to
    resume", and Pause.resume treats every failure the same way -- see the note there about banking
    the time before the attempt.
    """
    if hasattr(process, "thaw"):
        return process.thaw()

    psutil.Process(process.pid).resume()

    return True


class Pause:
    """One count, suspended or not. Nothing else -- no widgets, no threads, no disk.

    Deliberately a small object with state rather than two loose functions: "is something
    currently suspended" is the fact every caller needs, and a bare `suspend()` would leave
    each of them tracking it separately. There is exactly one paused count at a time because
    there is exactly one counting runner.
    """

    def __init__(self):
        # The process that is currently stopped, or None. This IS the "are we paused" flag --
        # a separate boolean could disagree with it, and the disagreement that matters is the
        # one where we think nothing is paused and a count is frozen forever.
        self.process = None

        # When the current pause began, and how long every pause so far has lasted. Separate
        # because the clock needs the total DURING a pause too, not only after it.
        self.since = None
        self.total = 0.0

    def paused(self):
        return self.process is not None

    def suspend(self, process):
        """Stop `process`. True if it was actually stopped, False if there was nothing to stop.

        THREE WAYS THIS DECLINES, and all three are normal rather than errors:
          * no process -- nothing is counting, so drawing needs no pause at all;
          * `poll()` is not None -- the count finished between the button press and here, which
            is a real race on a job whose last second is unpredictable;
          * already paused -- a second Draw click, or a resume that never happened.

        Returning False rather than raising, because every caller's response is the same: carry
        on and open the window. A refusal here is not a reason to stop drawing.
        """
        if process is None or self.paused() or process.poll() is not None:
            return False

        try:
            freeze(process)
        except psutil.Error:
            # It went away between poll() and here, or belongs to someone else. Same answer as the
            # three refusals above: carry on and open the window.
            return False

        self.process = process
        self.since = time.monotonic()

        return True

    def resume(self):
        """Let the count go again. True if something was actually resumed.

        SAFE TO CALL WHENEVER, and it is called from more than one place for that reason: when
        the drawing window closes, and again when the app quits. A count left stopped is worse
        than any other failure here -- it holds its memory, makes no progress, and shows no sign
        of being stuck, because a stopped process looks exactly like an idle one.

        THE TIME IS BANKED EVEN IF THE THAW FAILS. `thaw` raises if the process died while stopped
        (it cannot die by itself, but it can be killed), and losing the accounting on the way past
        would leave the clock permanently wrong. So the state is cleared first.
        """
        if not self.paused():
            return False

        process = self.process
        self.total += time.monotonic() - self.since
        self.process = None
        self.since = None

        try:
            thaw(process)
        except (psutil.Error, OSError, ValueError):
            # Gone already. Nothing to resume and nothing to report -- the job's own `finished`
            # will say what happened to it.
            return False

        return True

    def paused_for(self):
        """Total seconds spent paused, including a pause that is still going.

        Used to correct the elapsed clock, so it has to include the CURRENT pause -- the clock
        ticks every second while the drawing window is open, and a total that only counted
        finished pauses would let it run away and then jump backwards.
        """
        return self.total + (time.monotonic() - self.since if self.paused() else 0.0)

    def forget(self):
        """Drop the accounting, for when a new section starts and the clock restarts with it.

        Not the same as resume: this is about the CLOCK, not the process. Called from the
        window's per-section handler, because the paused total belongs to one section's elapsed
        time and carrying it into the next would make the next one read short.
        """
        self.total = 0.0


class ByPid:
    """A process we did not start, in the shape Pause needs: a `pid` and `poll()`.

    WHY THIS EXISTS. A count started in another app window, or from a terminal, is not a Popen
    here -- all this window has is the pid app_running found in the process list. Pause's rules
    are all correct for it; the only thing missing was the method. So the adapter is four
    lines and Pause is untouched, rather than Pause growing a second code path for pids.

    NO `freeze` OR `thaw` OF ITS OWN, on purpose: it carries a pid, so the module-level pair handles
    it by exactly the same route as a Popen. One mechanism, two handles.

    `poll()` RETURNS 0, NOT AN EXIT CODE, once the process is gone. We never started it, so its
    real code went to whoever did. Pause only ever asks "has it finished" -- None or not-None --
    and inventing a plausible-looking number is how a caller comes to trust one.

    ABOUT PID REUSE, the one real hazard in signalling a stranger: a pid is recycled once its
    process is reaped, so a stale pid can name something else entirely -- and SIGSTOP to the wrong
    process would freeze it silently. Two things keep that away from here. The pid is read from
    the process list by its command line (app_running.counting_now, which matches count_cells.py
    and the section's filename), and the window re-reads it immediately before suspending rather
    than trusting the 15-second poll. What is NOT claimed is that this is airtight in theory; it
    is that the window between reading and freezing is under a second and no operating system
    recycles pids that fast.

    `stem` is carried only so callers can say WHOSE count was stopped. Nothing here uses it.
    """

    def __init__(self, pid, stem=None):
        self.pid = pid
        self.stem = stem

    def poll(self):
        """None while the process is alive, 0 once it is gone.

        NOT `os.kill(pid, 0)`. On POSIX that is the standard way to ask -- the kernel runs every
        permission check and delivers nothing -- but Windows has no signal 0 convention and os.kill
        there calls TerminateProcess, so the question would have killed the count. See the module
        docstring. psutil asks without touching anything, on every system.

        A process owned by somebody else reads as ALIVE, because it is; the freeze will then fail
        and Pause already treats that as "not paused".
        """
        try:
            if psutil.Process(self.pid).is_running():
                return None
        except psutil.Error:
            return 0

        return 0


class FakeProcess:
    """A process that writes down what was done to it instead of having it done. For main(), and tests.

    Here rather than in main() so the shape Pause needs is written down in one place. IT ANSWERS FOR
    ITSELF, with its own `freeze` and `thaw`, which is what lets every rule in Pause be checked
    without a real 1.5 GB count to stop -- and, since 2026-09-13, without a real operating system
    either: SIGSTOP does not exist on Windows, so a fake that recorded signal numbers could only ever
    have been checked on a Mac.
    """

    def __init__(self, exited=None):
        self.exited = exited
        self.done = []

    def poll(self):
        return self.exited

    def freeze(self):
        self.done.append("stop")

        return True

    def thaw(self):
        self.done.append("go")

        return True


def main():
    pause = Pause()
    process = FakeProcess()

    assert not pause.paused()
    assert pause.suspend(None) is False, "nothing counting is not an error"
    assert pause.suspend(process) is True
    assert pause.paused()
    assert process.done == ["stop"]

    # A second click while already paused must not stop it twice. On POSIX stops do not nest, so one
    # resume would let it go anyway -- but on Windows suspends DO nest, one resume per suspend, so a
    # second freeze here would leave the count stopped after the drawing window closed. And either
    # way the accounting would be wrong.
    assert pause.suspend(process) is False
    assert process.done == ["stop"]

    assert pause.paused_for() > 0.0, "a pause in progress must already count"

    assert pause.resume() is True
    assert not pause.paused()
    assert process.done == ["stop", "go"]

    # Resuming twice is what happens when the app quits after a normal resume. It must be quiet.
    assert pause.resume() is False
    assert process.done == ["stop", "go"]

    banked = pause.paused_for()
    assert banked > 0.0
    assert pause.paused_for() == banked, "a finished pause must not keep growing"

    pause.forget()
    assert pause.paused_for() == 0.0

    # A count that finished while the button was being clicked.
    assert Pause().suspend(FakeProcess(exited=0)) is False

    # A process killed while stopped: the state must still come back clean.
    class Dead(FakeProcess):
        def thaw(self):
            raise psutil.NoSuchProcess(-1)

    dead = Pause()
    dead.suspend(Dead())
    assert dead.resume() is False
    assert not dead.paused(), "a failed thaw must not leave us thinking a count is paused"
    assert dead.paused_for() > 0.0, "the time must be banked even when the thaw fails"

    print("pause rules: all cases correct")

    # AND AGAINST A REAL PROCESS, because everything above only proves the bookkeeping. This
    # proves the mechanism: a child that prints every 0.1 s stops printing when suspended and
    # carries on from where it was afterwards.
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import time\n"
         "for i in range(1000):\n"
         "    print(i, flush=True)\n"
         "    time.sleep(0.1)\n"],
        stdout=subprocess.PIPE,
        text=True,
    )

    try:
        real = Pause()
        time.sleep(0.4)

        assert real.suspend(child) is True
        assert child.poll() is None, "a stopped process is still running, not finished"

        time.sleep(1.0)
        assert real.resume() is True
        assert 0.9 < real.paused_for() < 1.6, real.paused_for()

        time.sleep(0.3)
        child.terminate()

        printed = [int(line) for line in child.stdout.read().split()]

        # THE POINT OF THE WHOLE FILE: no gap and no repeat. It carried on from where it was,
        # so a paused count loses no work and redoes none.
        assert printed == list(range(len(printed))), printed[:20]
        assert len(printed) < 15, f"{len(printed)} lines in ~1.7 s means it never stopped"

        print(f"real process: stopped and resumed cleanly, {len(printed)} lines over "
              f"{real.paused_for():.1f} s paused")
    finally:
        child.kill()
        child.wait()

    # AND THE SAME THING KNOWING ONLY A PID, which is all the window has for a count started
    # somewhere else. Same assertions, different handle.
    stranger = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import time\n"
         "for i in range(1000):\n"
         "    print(i, flush=True)\n"
         "    time.sleep(0.1)\n"],
        stdout=subprocess.PIPE,
        text=True,
    )

    try:
        outside = Pause()
        handle = ByPid(stranger.pid, "some section")

        assert handle.poll() is None, "a running process must not look finished"
        assert outside.suspend(handle) is True
        assert handle.poll() is None, "a stopped process is still running, not finished"

        time.sleep(1.0)

        assert outside.resume() is True

        time.sleep(0.3)
        stranger.terminate()

        # THE PROOF IS THE EFFECT, not the call. It used to be `ps -o state=` looking for T, which
        # was the right instinct -- os.kill succeeds whether or not it did anything useful, so
        # "we sent SIGSTOP" was never the claim. But `ps` is a Unix program and T is a Unix state
        # letter, so that proof could only be run on the machine that already worked. Counting the
        # lines works everywhere: 1.7 s at ten a second is seventeen lines, and it printed a handful.
        printed = [int(line) for line in stranger.stdout.read().split()]

        assert printed == list(range(len(printed))), printed[:20]
        assert len(printed) < 15, f"{len(printed)} lines in ~1.7 s means it never stopped"

        stranger.kill()
        stranger.wait()

        # A pid that has been reaped: it must read as finished, not as something to stop. This is
        # the case os.kill(pid, 0) got right on POSIX and would have got catastrophically wrong on
        # Windows, where the question itself terminates whatever now owns the pid.
        assert ByPid(stranger.pid).poll() == 0
        assert Pause().suspend(ByPid(stranger.pid)) is False, "a dead pid is nothing to stop"

        print("by pid: a count this app did not start stops and resumes the same way")
    finally:
        stranger.kill()


if __name__ == "__main__":
    main()
