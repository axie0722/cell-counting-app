"""Run a counting or drawing job off the main thread, so the window stays alive.

Step 4a. app_actions.py knows HOW to run a job; this knows how to run one WITHOUT FREEZING
THE WINDOW, and nothing else. Kept separate from the window itself because threading is the
part most likely to need fixing later, and it is far easier to reason about 100 lines of
threading with no widgets in them.

THE PROBLEM. app_actions.run() blocks until the subprocess exits, and a count takes minutes.
Called straight from a button handler it blocks the main thread, which is the thread that
repaints the window and handles clicks -- so the window stops redrawing, macOS shows the
spinning beachball and offers to force-quit an app that is working perfectly. The streamed
output run() was written to provide would also arrive in one lump at the end, since nothing
can repaint to show it.

THE RULE THAT SHAPES THIS FILE. Only the main thread may touch widgets. This is true of
essentially every GUI toolkit, not just Qt, and breaking it does not reliably crash -- it
corrupts state and crashes later somewhere unrelated, which is the worst kind of bug to own.
So the worker never touches the interface. It emits SIGNALS, which Qt queues and delivers on
the main thread, where a slot may safely update a label.

This is why app_actions.run() takes an `on_output` callback rather than printing: the worker
hands it a function that emits a signal.

BATCHES RUN HERE TOO, one section after another -- see JobRunner.run_many. Sequentially and
never in parallel: the reason is memory, and it is written out in that method rather than
here because it is the single most likely thing to be "improved" by someone who has not
measured it.

ONE RUNNER IS STILL ONE JOB, and that stayed true when drawing was allowed during a count.
The window keeps a SECOND JobRunner for drawing rather than letting this one run two things,
so the invariant is structural instead of remembered. The two do not compete: `process()`
hands the counting subprocess to app_pause.Pause, which stops it for as long as the drawing
window is open. See app_pause.py for why stopping beats sharing.

WHAT IS DELIBERATELY NOT HERE. No progress percentage, because neither draw_regions.py nor
count_cells.py knows how far through it is -- a fake bar that jumps to 90% and waits is worse
than a line of real output.

STOPPING IS NOW SAFE, and that changed for a reason worth keeping. There used to be no cancel
because killing a count mid-write could leave a truncated "<stem> counted cells.csv" that
later reads as a real but wrong count. count_cells.save_csv removed that: every CSV is written
to a neighbour file and renamed into place, so at any instant the real file is either the whole
old one or the whole new one. With that true, `stop()` below is just a signal, and the window
can offer it when someone closes the app during a count. A batch's finished sections are on
disk already; stopping costs the section in progress and nothing else.

There is still no Stop BUTTON on the window -- only the question asked when quitting -- because
a button invites stopping a 40-minute job by accident. `stop()` is here for whenever that
changes.

Usage:
  python app/app_jobs.py "<stem>"              # run one count through the worker
  python app/app_jobs.py "<stem>" "<stem>" ... # run a batch, printing every signal
"""

import sys

from qtpy.QtCore import QObject, QThread, Signal

from pathlib import Path

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. This file can be run as its own
# process, and Python then puts only ITS folder on sys.path -- so `import ps6` at the root, or a
# sibling folder's module, would not be found. app_path.py explains the whole arrangement; the line
# before it is there because app_path is at the root, which is not on the path yet either. The
# condition also covers a flat copy of the app, where app_path sits right here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

import app_actions

# qtpy rather than PyQt6 or PySide6 by name: napari already goes through qtpy, so the binding
# is whatever napari resolved to. Naming one directly means the app breaks if the environment
# ever has the other -- and this project's whole drawing step is napari's.


class Job(QObject):
    """One background job. Lives on a worker thread; talks back only through signals.

    A QObject rather than a QThread subclass. The worker-object pattern is the one Qt
    documents: subclassing QThread puts the object on the CREATING thread while only run()
    executes on the new one, which makes it very easy to touch the wrong thread by accident.
    Moving a plain object onto a thread has no such trap.
    """

    # Signals are class attributes, and the type in the brackets is what gets delivered.
    output = Signal(str)
    finished = Signal(int, str)

    # The subprocess, the moment it exists. Count jobs only -- see app_actions.count_section.
    # A SIGNAL rather than an attribute the window reads, for the reason at the top of this
    # file: the Popen is created on the worker thread and the pause is triggered from the main
    # thread, so the handover goes through Qt's queue like every other cross-thread fact here.
    launched = Signal(object)

    def __init__(self, kind, section, model_path=None):
        super().__init__()

        self.kind = kind
        self.section = section
        self.model_path = model_path

    def say(self, line):
        """Emit one output line, unless this job has already been torn down.

        THE RACE THIS CLOSES. When the app quits during a job, the window and everything Qt
        owns can be destroyed while this method is still running on the worker thread; the next
        emit then raises `RuntimeError: wrapped C/C++ object of type Job has been deleted`. Seen
        2026-08-20 when closing the app with a count suspended and a drawing window open: two
        tracebacks on the way out, from the count job and the drawing job.

        Nothing was broken -- the count stopped exactly as asked and the process was exiting
        anyway -- but an error message printed by a correct shutdown is a false alarm someone
        will one day spend an afternoon on. Swallowing it is right rather than lazy: the signal's
        only purpose is to reach a window, and the window is gone.
        """
        try:
            self.output.emit(line)
        except RuntimeError:
            pass

    def done(self, code, message=""):
        """Emit `finished`, unless this job has already been torn down. See say()."""
        try:
            self.finished.emit(code, message)
        except RuntimeError:
            pass

    def start(self):
        """Run the job. Called on the worker thread, never directly.

        Every exception is caught and reported through `finished` rather than raised. An
        exception escaping a slot on a worker thread does not surface anywhere a person can
        see -- the job simply stops, the window waits forever, and there is nothing to read.
        Turning it into a normal failure means a traceback lands in the output pane.
        """
        try:
            if self.kind == "draw":
                code, _ = app_actions.draw_section(self.section, self.say)
            elif self.kind == "review":
                code, lines = app_actions.review_section(self.section, self.say)

                # Same refusal shape as count: None means "cannot, and here is why", which
                # is a readable failure rather than an exception.
                if code is None:
                    self.done(1, "\n".join(lines))

                    return
            elif self.kind == "count":
                code, lines = app_actions.count_section(
                    self.section, self.model_path, self.say, self.launched.emit
                )

                # count_section returns None when the section is not ready, with the reason
                # in `lines`. That is a refusal rather than a crash, so it is reported as a
                # failed job with a readable message.
                if code is None:
                    self.done(1, "\n".join(lines))

                    return
            else:
                self.done(1, f"Unknown job {self.kind!r}")

                return

            self.done(code)
        except Exception as error:
            import traceback

            self.say(traceback.format_exc())
            self.done(1, f"{type(error).__name__}: {error}")


class JobRunner(QObject):
    """Keeps one job and its thread alive, and cleans both up afterwards.

    THIS CLASS EXISTS BECAUSE OF GARBAGE COLLECTION. A QThread whose last Python reference
    goes out of scope is destroyed while still running, and Qt aborts the process with
    "QThread: Destroyed while thread is still running". Holding both the thread and the job
    on a longer-lived object is the fix, and it is the single most common way a first Qt
    threading attempt fails.
    """

    output = Signal(str)

    # One section started. (stem, position in the batch, batch size) -- the window uses it to
    # restart its elapsed clock, which has to happen per section rather than per batch.
    started = Signal(str, int, int)

    # One section finished. Emitted for every section in a batch, so the table refreshes as
    # each count lands rather than only at the end of a five-hour queue.
    finished = Signal(int, str)

    # The whole batch finished. (succeeded, failed). A SEPARATE signal from `finished` because
    # the window has to do different things: after one section it refreshes and leaves the
    # buttons disabled; after the batch it re-enables them. Deciding which by inspecting
    # busy() from inside the finished handler would work but reads as a coincidence.
    queue_finished = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.job = None

        # The running job's subprocess, for whoever needs to signal it. Set by the job's
        # `launched` signal and cleared the moment the job ends, so `process()` can never hand
        # out a handle to something that has already exited. Only count jobs ever set it.
        self.launched_process = None

        # True from the moment a batch is accepted until its last section has finished --
        # including the instants between sections, when `thread` is None but the runner is
        # certainly not free. See busy().
        self.batch_active = False

        # The batch. `queue` holds sections not yet started; the one running is not in it.
        self.queue = []
        self.kind = None
        self.model_path = None
        self.stem = None

        # Position of the running section within the batch, 1-based, purely for display.
        self.position = 0
        self.total = 0
        self.succeeded = 0
        self.failed = 0

    def busy(self):
        """Whether the runner is occupied, so nothing new may start and buttons stay off.

        NOT simply "a thread exists". Between two sections of a batch there is a moment with
        no thread -- the finished one has been torn down and the next has not been created --
        and during it the window is refreshing in response to `finished`. Reporting False
        there would flicker every button back on mid-batch and let a fast click start a second
        job alongside the queue.

        `batch_active` covers exactly that gap. It is cleared just before the LAST section's
        `finished` is emitted, so the buttons come back in the same refresh that shows the
        final numbers rather than one refresh later.
        """
        return self.thread is not None or self.batch_active

    def process(self):
        """The running subprocess, or None. Safe to call from the main thread.

        Exists so app_pause.Pause can stop a count while a drawing window is open. Returns None
        between sections of a batch and for job kinds that do not report a process, which is
        the correct answer in both cases: there is nothing to suspend.
        """
        return self.launched_process

    def stop(self):
        """Empty the queue and ask the running subprocess to exit. Returns what was stopped.

        Returns (stopped, dropped): whether a running process was signalled, and how many
        queued sections were thrown away. Both are worth reporting -- "stopped the count" and
        "and abandoned the four sections behind it" are different pieces of news.

        terminate() is SIGTERM, the polite kill: the default handler ends the process, but it
        can be caught, and nothing here catches it. Not SIGKILL, which cannot be caught at all
        and would skip any cleanup a library wanted to do. Either is safe as far as the data
        goes -- see the note about save_csv at the top of this file -- so the gentler one wins.

        THE QUEUE IS EMPTIED FIRST, before the signal. _done runs when the process dies and
        immediately starts the next section, so clearing afterwards would race: the next count
        could be launched in between and survive the stop. Order is the whole correctness
        argument here.

        A count that is PAUSED must be resumed before this is called. A stopped process does
        not run, so it cannot act on SIGTERM until it does; the window handles that ordering.
        """
        dropped = len(self.queue)
        self.queue = []

        process = self.launched_process
        stopped = process is not None and process.poll() is None

        if stopped:
            process.terminate()

        return stopped, dropped

    def remaining(self):
        """How many sections have not started yet, excluding the one running."""
        return len(self.queue)

    def queued_stems(self):
        """Stems still waiting, as a set, for the window to mark them in its list.

        A method rather than letting the window read `self.queue` directly: the window only
        ever wants to ask "is this one waiting?", and handing it the list of section dicts
        invites it to start using them for something else.
        """
        return {section["stem"] for section in self.queue}

    def run(self, kind, section, model_path=None):
        """Start one job. Returns False if something is already running.

        Kept as a one-section wrapper around run_many so that every existing caller -- and
        app_jobs.main -- goes through exactly the same path a batch does. One code path means
        a batch of one cannot behave differently from a single job.
        """
        return self.run_many(kind, [section], model_path)

    def run_many(self, kind, sections, model_path=None):
        """Start a batch, one section at a time. Returns False if busy, or if given nothing.

        ONE AT A TIME, NOT IN PARALLEL, and that is the point of the queue rather than a
        limitation of it. Each count holds the image's green channel as float32 -- 354 MB for
        the largest section here -- plus the tissue mask, three region masks and another
        float32 score array during detection, peaking around 1.5-2 GB. This machine has 8 GB
        and is typically already using most of its swap, so two concurrent counts thrash and
        finish later than two sequential ones. Measured, not assumed: a count sharing the
        machine with one other job dropped to 24.5% CPU with shrinking resident memory.

        What the queue removes is the WAITING, not the work. Eight sections still take most
        of a day; the difference is that nobody has to come back every forty minutes to start
        the next one.
        """
        if self.busy() or not sections:
            return False

        self.kind = kind
        self.model_path = model_path
        self.queue = list(sections)
        self.total = len(self.queue)
        self.position = 0
        self.succeeded = 0
        self.failed = 0
        self.batch_active = True

        self._start_next()

        return True

    def add(self, sections):
        """Append sections to a batch that is already running. Returns the ones taken.

        THE QUEUE IS NOW JOINABLE, which is what Alice expected it to be all along (2026-08-20:
        *"when i count cells for one image i can't press count cells for another image"*).
        run_many refuses while busy -- correctly, since starting a second count would thrash
        this machine -- but refusing to START a second count is a different thing from refusing
        to REMEMBER one, and the two were conflated.

        Returns what was actually appended rather than True/False, so the window can name the
        sections it took. A section already running or already waiting is dropped: queueing a
        stem twice would spend forty minutes producing a second answer to the same question,
        and the two would race to be the file on disk. The window filters for the same reason
        before it gets here (app_queue.plan_addition, which also explains it to the person) but
        this check is the one that has to be right -- the runner owns the queue, and a caller
        that forgot to filter must not be able to corrupt it.

        `total` grows, so the "3 of 5" in the status line stays true as the batch gets longer.

        Refuses unless a COUNT batch is running. A draw or review job is one window a person is
        sitting in front of; there is no queue there to join.
        """
        if not self.busy() or self.kind != "count" or not sections:
            return []

        waiting = self.queued_stems()
        taken = []

        for section in sections:
            if section["stem"] == self.stem or section["stem"] in waiting:
                continue

            taken.append(section)
            # Added as it goes rather than from the snapshot above, so two copies of the same
            # stem in one selection cannot both get through.
            waiting.add(section["stem"])

        self.queue.extend(taken)
        self.total += len(taken)

        return taken

    def drop(self, stems):
        """Take sections out of the queue before they start. Returns the ones removed.

        THE OTHER HALF OF A JOINABLE QUEUE. Alice, 2026-08-20: *"can i also remove things from
        queue"*. Adding one section at a time as she noticed it meant the queue could only ever
        grow, so one mis-click committed the machine to another forty minutes and the only way
        out was closing the app -- which drops the whole queue, including the seven she did
        want. Removing is the cheap, reversible operation the adding always implied.

        ONLY WHAT HAS NOT STARTED. The section being counted right now is not in `self.queue`,
        so it cannot be dropped here by accident; ending that one is stop(), a different
        decision with a different cost -- it throws away work already done.

        `total` comes down with the queue, because it is what the status line's "3 of 5" is
        divided by. Leaving it alone would make a shortened batch count towards a total that
        can no longer be reached, and the progress line would stop at "4 of 5" forever.

        `position` is deliberately NOT touched: it counts sections that have been STARTED, and
        removing something from the back of the queue does not un-start anything.
        """
        stems = set(stems)

        if not stems:
            return []

        removed = [section for section in self.queue if section["stem"] in stems]
        self.queue = [section for section in self.queue if section["stem"] not in stems]
        self.total -= len(removed)

        return removed

    def _start_next(self):
        """Take the next section off the queue and run it."""
        section = self.queue.pop(0)

        self.position += 1
        self.stem = section["stem"]

        self.thread = QThread()
        self.job = Job(self.kind, section, self.model_path)

        # The move is what puts the work on the other thread. After this, any slot of `job`
        # invoked by a signal runs there.
        self.job.moveToThread(self.thread)

        # started -> start() is what actually launches the work, rather than the thread
        # doing anything by itself.
        self.thread.started.connect(self.job.start)

        self.job.output.connect(self.output.emit)
        self.job.finished.connect(self._done)
        self.job.launched.connect(self._launched)

        self.thread.start()

        self.started.emit(self.stem, self.position, self.total)

    def _launched(self, process):
        """Remember the running subprocess. Runs on the MAIN thread, delivered by Qt."""
        self.launched_process = process

    def _done(self, code, message):
        """Tear the thread down, report, then start the next section. Order matters.

        quit() asks the thread's event loop to stop and wait() blocks until it has actually
        stopped -- both, because quit() alone returns immediately and the thread might still
        be running when the next job tries to start one. The wait is on the main thread but
        lasts microseconds: the work is already over. The references are cleared because
        reusing a finished QThread is not supported.

        `finished` is emitted BEFORE the next section starts, so signals arrive in the order
        the events actually happened and a log reads correctly: section 1 finished, then
        section 2 started. An earlier version started the next job first, to keep busy() from
        momentarily reading False -- which fixed the flicker but reported the two events
        backwards. `batch_active` fixes it without lying about the order; see busy().

        A FAILED SECTION DOES NOT STOP THE BATCH. It is tallied and the queue moves on,
        because the alternative -- abandoning six good sections because the second one had no
        dorsal marker -- wastes the hours the queue exists to save. The failure is reported
        through `finished` as it happens and counted in the summary at the end.
        """
        self.thread.quit()
        self.thread.wait()

        self.thread = None
        self.job = None

        # Cleared here, not in _start_next, so nothing can be handed a dead process during the
        # gap between sections of a batch -- a freeze aimed at an exited pid is at best a no-op
        # and at worst stops whatever inherited that number.
        self.launched_process = None

        if code == 0:
            self.succeeded += 1
        else:
            self.failed += 1

        more = bool(self.queue)

        # Cleared before the last `finished` so the window's refresh sees busy() as False and
        # re-enables the buttons in the same pass that shows the final numbers.
        if not more:
            self.batch_active = False

        self.finished.emit(code, message)

        if more:
            self._start_next()
        else:
            self.queue_finished.emit(self.succeeded, self.failed)


def main():
    """Run a count, or a batch of them, through the worker from the terminal.

    Worth having: it exercises the threading with no window at all, so a hang here is
    definitely threading and not layout. The QApplication is still needed -- signals are
    delivered by an event loop, so without one running nothing would ever arrive.
    """
    from qtpy.QtWidgets import QApplication

    if len(sys.argv) < 2:
        raise SystemExit('usage: python app/app_jobs.py "<section stem>" ["<stem>" ...]')

    application = QApplication([])
    sections = [app_actions.resolve_section(name) for name in sys.argv[1:]]

    runner = JobRunner()
    runner.output.connect(lambda line: print(f"  | {line}"))

    runner.started.connect(
        lambda stem, position, total: print(f"\n--- {position}/{total} {stem}")
    )

    def section_done(code, message):
        print(f"  finished with code {code}{': ' + message if message else ''}")

    def batch_done(succeeded, failed):
        print(f"\nbatch over: {succeeded} succeeded, {failed} failed")
        application.quit()

    runner.finished.connect(section_done)
    runner.queue_finished.connect(batch_done)

    print(f"counting {len(sections)} section(s) on a worker thread")
    runner.run_many("count", sections)

    application.exec()


if __name__ == "__main__":
    main()
