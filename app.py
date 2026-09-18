"""The window. Sections on the left, numbers on the right, buttons and output below.

Step 4b, and the last piece. Everything it shows or does already exists in a module that runs
from a terminal:

    app_state.py    what sections exist and how far each has got
    app_table.py    counts, areas, densities, and whether a row is stale
    app_actions.py  how to run drawing and counting as subprocesses
    app_jobs.py     how to run one of those without freezing this window
    app_queue.py    which of several chosen sections can be counted, and how long it will take
    app_pause.py    how to hold a count still while a napari window is open
    app_export.py   the same numbers as rows a spreadsheet will take
    app_all_data.py the second window: the whole folder at once, and the two exports

THIS FILE ADDS NO LOGIC, ON PURPOSE. Every number shown comes from app_table.build_table and
every job from app_jobs.JobRunner. That is not tidiness for its own sake: logic inside a
window can only be tested by clicking, which means it stops being tested. Anything that
decides something belongs in a module with a __main__, so it can be checked from a terminal
before it is ever wired to a button.

THE FILES ARE THE STATE, so this window owns nothing. It can be closed at any point without
losing anything -- outlines, counts and cached areas are all sidecars next to the images, and
draw_regions.py saves in a finally: block. This is also why refresh() just re-reads the disk
rather than tracking changes: the disk cannot disagree with itself.

CLOSING DURING A COUNT ASKS FIRST. The count is a separate process, so it does not have to die
with this window -- see closeEvent, which offers to leave it running (it finishes and saves on
its own) or to stop it. Before 2026-08-20 there was no question and no reliable answer: the
count sometimes survived and sometimes died, depending on whether it happened to print anything
after the pipe to this window broke.

CTRL-C ASKS THE SAME QUESTION, and did not until 2026-08-20. closeEvent is called when the
WINDOW closes; killing the app from the terminal it was started in bypassed it entirely, so the
question never appeared and a count was orphaned by accident. ask_before_dying at the bottom of
this file routes SIGINT and SIGTERM through the same close.

A COUNT THIS WINDOW DID NOT START IS STILL SHOWN. Alice, 2026-08-20: *"if its still counting
while closed, when i open it i can't tell which one is being counted still"*. Those sections read
"counting" and are marked in the list like any other running count, from app_running.py, which
looks at the process list because a count in progress has written no files yet -- the one fact in
this window that the folder cannot supply. It is also a guard: Count refuses a section something
else is already counting, since two counts would race to write one file.

THE QUEUE OUTLIVES A QUIT. Alice, 2026-08-20: *"since when i reopened it i had to add all the
images to queue again"*. The running count survives closing on its own -- it is a separate process
-- but the list of sections waiting behind it lived only in this window, so quitting deleted a
decision made one click at a time. app_saved_queue.py writes it down; offer_saved_queue brings it
back and ASKS. Nothing restarts by itself: a queue is hours of machine time, and an app that
resumed a five-hour batch because it was opened to look up one number would be worse than the bug
it fixed.

WHAT IS DELIBERATELY MISSING. No progress bar (nothing knows its own progress; a bar that
sticks at 90% is a lie), no Stop button (stopping is safe now that count_cells writes its CSVs
atomically, but a button invites ending a 40-minute job by accident -- the question on quitting
is enough), and no editing of numbers in the table (a hand-edited density is unreproducible,
and the point of the app is that every number can be traced back to a file).

There IS a status line with an elapsed clock, which is not the same thing. A clock reports a
measured fact and never has to guess how much is left; the reason it is needed is that a
count prints nothing for its first several minutes, and silence with no clock beside it is
indistinguishable from a frozen window.

MANY SECTIONS AT ONCE. The list is multi-select, and Count counts every section selected, one
after another. Sequentially and never in parallel -- two counts at once exhaust this machine's
memory and finish later than two run in turn (measured; app_jobs.run_many has the numbers).
What the queue removes is the waiting, not the work: eight sections still take most of a day,
but nobody has to come back every forty minutes to start the next one.

THE QUEUE CAN BE JOINED WHILE IT RUNS. Alice, 2026-08-20: *"i thought i was able to make a count
cells queue but when i count cells for one image i can't press count cells for another image"*.
It could not be, and she was right that it should: pick another section during a count and the
button reads "Add to queue" instead of "Count cells". Nothing new starts -- the section is
appended behind the one running -- so the memory rule is untouched, and the night's work no
longer has to be decided before the first count begins. The section being counted and any
already waiting are refused by name rather than silently ignored, since queueing one stem twice
would spend forty minutes computing a second answer to the same question.

REVIEW CAN CHANGE THE COUNT. Alice, 2026-09-12: *"could you change the app so that review cells
lets the user add and delete cells"*. The review window used to be read-only and the tooltip said
so as though it were a safety feature; it was a limitation. Seeing a false positive and being
unable to remove it leaves the wrong number on the report even though the one person who can tell
has already looked at it. Cells added and deleted in napari go back into "<stem> counted cells.csv"
on close, which is where this app reads its cell numbers from, so Refresh shows them -- and
show_counts.py updates ps6_counts.csv and keeps an append-only record of every change. The one
thing to be careful about is reviewing the section a count is working on, because that count will
rewrite the cell file when it finishes; the log and the list both say so.

DRAWING AND REVIEWING DURING A COUNT ARE ALLOWED, and both work by STOPPING the count rather
than sharing the machine with it. Alice, 2026-08-20: *"is it possible for me to draw outlines
while cell counting is going on"*, and later *"why can't i review cells for the images that are
already counted"* -- she could not, because a running count disabled Review for every section in
the folder, which reads as the app claiming the COUNTS are missing. Pressing either button now
suspends the counting process where it stands and resumes it from exactly there when napari
closes -- no work lost, none repeated. Sharing was tried and measured: the count fell to 24.5%
CPU and napari was swapped out entirely, which looks like a frozen app. app_pause.py has the
mechanism and its one honest limit. What stays disabled while a napari window is open is
counting, refreshing, switching folders AND the other napari button, because starting a count
INTO an open window is the same collision in the other direction and there is only one pause to
give out.

Because that is a commitment of hours with one click, and there is no cancel, the button says
how many it will run, the log lists them in order with an estimate, anything unready is left
out with the reason, a confirmation appears before the batch starts, and the list marks which
section is counting now and which are still waiting. All of it is one idea: a queue you cannot
see is a queue you cannot trust.

GETTING THE NUMBERS OUT. View all data opens a second window (app_all_data.py) holding every
counted region in the folder, with a second tab averaging each bird's sections. The copy and
save buttons live in there rather than here, so what you copy is what you can see -- they used
to be on this window, where they exported the whole folder while the screen showed one section.
Neither talks to Google directly; app_export.py says why that was rejected.

WHICH FOLDER. The app is not tied to this project directory. Choose folder picks any folder,
searches it and everything inside it, and remembers the choice (app_folder.py). That works
because sidecars sit next to their image rather than in the working directory -- see
ps6.sidecar_path -- so a folder of images carries its own outlines and counts.

Usage:
  python app.py
"""

import signal
import sys
import time
from pathlib import Path

# BEFORE ANY OF THE APP'S OWN IMPORTS, and that is not a style choice. The app's modules live in
# app/ regions/ review/ training/ counting/, and Python cannot import one until those folders are on
# sys.path -- so `import app_awake` twenty lines below would fail if this ran after it. It is also
# what gives the count and the drawing window their import path, since they are separate processes
# that inherit PYTHONPATH from this one. app_path.py explains the whole arrangement.
#
# `app_path` itself imports because it sits beside this file, and the folder of the script being run
# is always the first thing on sys.path. That is the one import that needs no setup, which is why
# the setup lives there.
import app_path

app_path.setup()

import pandas as pd
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QColor, QFont, QFontMetrics
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import app_awake
import app_folder
import app_pause
import app_queue
import app_running
import app_saved_queue
import ps6
from app_actions import NEXT_ACTION
from app_all_data import AllData
from app_jobs import JobRunner
from app_state import COUNTED, NEEDS_DORSAL, NEEDS_REGIONS, READY_TO_COUNT, find_sections
from app_table import areas_path, build_table, summarise_by_region

# One colour per status, so the list is readable without being read. The status is also drawn
# in italics under a plain file name -- see SectionDelegate -- which is what actually makes it
# stand out: colour alone left the two lines looking like one block of text.
#
# Darker than the obvious Material values (#9aa0a6, #c77b1e) because those sat near 2.5:1
# against white, below the 4.5:1 small text needs to stay readable. These clear it, and the
# four hues stay distinguishable from each other.
# Needs regions is RED because it is the one status that means "nothing here works yet": no
# outlines, so no areas, no densities, and nothing to count. Grey said "later"; red says now.
# A different red from STALE_COLOUR below, which is a different complaint -- and the two never
# appear in the same place, since the list shows statuses and only the tables show staleness.
STATUS_COLOURS = {
    NEEDS_REGIONS: "#d93025",
    NEEDS_DORSAL: "#b06000",
    READY_TO_COUNT: "#1558d6",
    COUNTED: "#0d652d",
}

STALE_COLOUR = "#b3261e"

# Sections waiting in a batch, and the one running. Marked in the list because a queue you
# cannot see is a queue you cannot trust: five hours in, the only way to know whether the
# right sections were picked is to look.
QUEUED_COLOUR = "#7b1fa2"
RUNNING_COLOUR = "#0b8043"

# Roughly how long one count takes on a laptop, from the two measured runs: 44 minutes for a
# 15 Mpx section and 35 for a smaller one under load. Shown as a range in the status label so
# nobody sits watching a window they think has frozen. A RANGE, not a countdown: nothing in
# the pipeline knows its own progress (see the docstring), and the honest thing to show is
# what past runs took plus how long this one has actually been going.
COUNT_MINUTES = "35-45 minutes"

# What each job is doing, in words, for the status label. Keyed by the same job names
# app_jobs uses, so a new job kind that forgets to add itself here shows a generic message
# rather than crashing.
#
# No "draw" or "review" entry: both run on the window runner now and neither has a clock --
# show_status writes their line directly, because a count and a napari window can be open at the
# same time and one keyed lookup cannot describe both. Counting is the only clocked job left, so
# this is a table of one; it stays a table because the lookup is what keeps a future job kind
# from crashing the status line.
JOB_MESSAGES = {
    "count": f"Counting cells -- this usually takes {COUNT_MINUTES}",
}

# Where the status line's colour is kept on each list item. A data role rather than the item's
# foreground, because the foreground applies to the whole item and the point of the delegate is
# that the two lines look different.
STATUS_COLOUR_ROLE = Qt.UserRole + 1

# Room around each row, in pixels, and how far the status line is indented under the name.
ROW_PADDING = 4
STATUS_INDENT = 14


class SectionDelegate(QStyledItemDelegate):
    """Draws one row of the section list: a plain file name, an italic status under it.

    THIS EXISTS BECAUSE A LIST ITEM HAS ONE FONT. Italicising a QListWidgetItem italicises the
    file name too, and then the status no longer stands out against it -- which was the whole
    point. Two fonts in one row means painting the row here instead.

    Only the TEXT is painted by hand. The background, the selection highlight and the focus ring
    are still drawn by the widget's own style, so a selected row looks exactly like a selected
    row in every other list on the machine. Reimplementing those is how a custom list starts
    looking subtly wrong on someone else's Mac, or in dark mode.
    """

    def paint(self, painter, option, index):
        name, _, status = (index.data(Qt.DisplayRole) or "").partition("\n")
        colour = index.data(STATUS_COLOUR_ROLE)

        # The style paints the row with no text of its own; ours goes on top.
        background = QStyleOptionViewItem(option)
        self.initStyleOption(background, index)
        background.text = ""

        widget = background.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, background, painter, widget)

        selected = bool(option.state & QStyle.State_Selected)

        painter.save()

        name_font = QFont(option.font)
        name_font.setItalic(False)
        painter.setFont(name_font)

        # A selected row's background is the system highlight colour, and a green or grey status
        # on top of it can be unreadable. So the palette's own highlighted-text colour wins
        # while a row is selected; the italics still tell the two lines apart, which is why
        # losing the colour there costs nothing.
        painter.setPen(
            option.palette.highlightedText().color()
            if selected
            else option.palette.text().color()
        )

        metrics = painter.fontMetrics()
        left = option.rect.left() + ROW_PADDING
        top = option.rect.top() + ROW_PADDING

        painter.drawText(left, top + metrics.ascent(), name)

        line_height = metrics.height()

        status_font = QFont(option.font)
        status_font.setItalic(True)
        painter.setFont(status_font)

        if not selected and colour:
            painter.setPen(QColor(colour))

        painter.drawText(
            left + STATUS_INDENT,
            top + line_height + painter.fontMetrics().ascent(),
            status,
        )

        painter.restore()

    def sizeHint(self, option, index):
        """Two lines plus padding, measured from the font rather than hard-coded.

        A fixed pixel height would clip the text as soon as the system font size changed, which
        is exactly the kind of breakage that only shows up on someone else's machine.
        """
        metrics = QFontMetrics(option.font)
        size = super().sizeHint(option, index)

        return QSize(size.width(), metrics.height() * 2 + ROW_PADDING * 2)


class CellCounter(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PS6 cell counter")
        self.resize(1180, 760)

        self.sections = []
        self.table = None

        # Where to look for images. Remembered from last time if that folder still exists,
        # otherwise the working directory -- so running this in a folder of images works with
        # no setup, exactly as it did before there was a picker.
        self.root = app_folder.startup_folder()

        self.runner = JobRunner(self)
        self.runner.output.connect(self.log)
        self.runner.started.connect(self.job_started)
        self.runner.finished.connect(self.job_finished)
        self.runner.queue_finished.connect(self.queue_finished)

        # A SECOND RUNNER, FOR THE ONE NAPARI WINDOW, so that opening it during a count does not
        # require loosening "one job at a time" -- each runner still enforces it, and the pair of
        # them can only ever be one count plus one window.
        #
        # REVIEWING MOVED HERE, 2026-08-20. It used to share the counting runner, on the argument
        # that "checking a count while counts are running is not a thing anyone needs". Alice:
        # *"why can't i review cells for the images that are already counted"* -- it is, and the
        # consequence of that argument was worse than being wrong: one running count greyed Review
        # out for every section in the folder, so the button appeared to be saying the COUNTS were
        # missing. Drawing had already solved the same problem the honest way (suspend the count,
        # do not share the machine), and reviewing is the same shape of job: one window, one person
        # in front of it. So it takes the same route.
        self.window_runner = JobRunner(self)
        self.window_runner.output.connect(self.log)
        self.window_runner.started.connect(self.window_started)
        self.window_runner.finished.connect(self.window_finished)

        # Holds the count still while that napari window is open. See app_pause.py.
        self.pause = app_pause.Pause()

        # Elapsed time for the running job. A ticking clock is not a progress bar: it reports
        # something measured rather than guessed, and it is what tells the difference between
        # "working" and "wedged" on a job with no output for minutes at a time.
        self.job_start_time = None
        self.job_kind = None
        self.job_stem = None
        self.job_position = 0
        self.job_total = 0

        # The section whose napari window is open, and whether it is being drawn or reviewed --
        # or None for both. Separate from job_stem because the two can be different sections at
        # the same time now: a count of one while another is drawn or reviewed.
        self.open_stem = None
        self.open_kind = None

        # Whose count the pause is holding, or None. Needed because it is not always this window's:
        # a count started in another window has no stem in `runner`, and "the count of None is
        # PAUSED" is what the log said before this existed. See count_to_hold.
        self.held_stem = None
        self.job_timer = QTimer(self)
        self.job_timer.setInterval(1000)
        self.job_timer.timeout.connect(self.show_status)

        # COUNTS THIS WINDOW DID NOT START. {stem: {pid, minutes}} from app_running, refreshed
        # from the process list. Alice, 2026-08-20: *"if its still counting while closed, when I
        # open it i can't tell which one is being counted still"*. Every other fact in this
        # window comes from the files, but a count in progress has written no files yet -- so
        # this is the one piece of state that has to come from somewhere else.
        self.outside_counts = {}

        # Set once, on the way out. Only queue_finished reads it: see there.
        self.quitting = False

        # THE QUEUE FROM BEFORE THE APP CLOSED, as stems, until it is started or dismissed. Kept
        # apart from the runner's own queue because these two are not the same thing: the runner's
        # queue is being worked through, this one is a list waiting for a decision. draw_list says
        # so in different words for exactly that reason.
        #
        # A LIST, not a set: the order sections were queued in is a decision she made, and it has
        # to survive being restored, written down again, and restored a second time.
        self.saved_stems = []

        # When that queue was written down. Kept so refresh() can ask "counted SINCE then?" with
        # the same clock app_saved_queue used when it decided what to bring back.
        self.saved_at = 0

        # Every 15 seconds, not every second like the clock above: one `ps` is about 10 ms, and
        # the only thing that can change is whether a forty-minute job is still going. The list
        # is redrawn only when the answer changes, so a person clicking through sections is not
        # fighting a widget that rebuilds under them.
        self.outside_timer = QTimer(self)
        self.outside_timer.setInterval(15000)
        self.outside_timer.timeout.connect(self.check_outside_counts)
        self.outside_timer.start()

        # The compiled-data window, made on first use and then kept. Held on self because a
        # second window whose only reference is a local variable is garbage collected the
        # instant the method returns, and simply vanishes.
        self.all_data = None

        self.build()
        self.report_folder()
        self.refresh()

        # The queue that was waiting when the app last closed. AFTER the window is painted, not
        # here: a dialog raised from __init__ appears in front of an empty grey rectangle, and it
        # has to be answered before the person can see which sections it is talking about. 200 ms
        # is long enough for the first paint and short enough to feel like part of opening.
        QTimer.singleShot(200, self.offer_saved_queue)

    # ---- layout ----

    def build(self):
        self.folder_label = QLabel("")
        # Elided from the left, because the meaningful end of a long path is the last part.
        # A wrapping path would push the table down every time a deep folder was chosen.
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.folder_button = QPushButton("Choose folder")
        self.folder_button.setToolTip(
            "Pick the folder holding your images. Every folder inside it is searched too.\n"
            "Outlines and counts are saved next to each image, so the folder stays\n"
            "self-contained: copy it to another computer and the work comes with it."
        )
        self.folder_button.clicked.connect(self.choose_folder)

        folder_bar = QHBoxLayout()
        folder_bar.addWidget(QLabel("Folder:"))
        folder_bar.addWidget(self.folder_label, 1)
        folder_bar.addWidget(self.folder_button)

        self.section_list = QListWidget()
        self.section_list.currentRowChanged.connect(self.section_changed)

        # The delegate is what draws each row's two lines in two different fonts. Held on self
        # because a delegate whose only reference is the setItemDelegate call gets garbage
        # collected, and the list then paints nothing.
        self.section_delegate = SectionDelegate(self.section_list)
        self.section_list.setItemDelegate(self.section_delegate)

        # Extended selection: shift-click for a run, cmd-click to add one. This is what makes
        # a batch possible, and it is the standard behaviour of every file list on the
        # machine, so it needs no explaining beyond the button label changing.
        self.section_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.section_list.itemSelectionChanged.connect(self.selection_changed)

        self.numbers = QTableWidget(0, 4)
        self.numbers.setHorizontalHeaderLabels(
            ["region", "cells", "area mm2", "cells/mm2"]
        )
        self.numbers.verticalHeader().setVisible(False)
        # Read-only: see the docstring. A typed-in density cannot be traced to a file.
        self.numbers.setEditTriggers(QTableWidget.NoEditTriggers)
        self.numbers.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.heading = QLabel("no section selected")
        self.heading.setFont(QFont("", 13, QFont.Bold))

        self.note = QLabel("")
        self.note.setWordWrap(True)

        self.draw_button = QPushButton("Draw outlines")
        self.draw_button.setToolTip(
            "Opens the section with NCM and CMM already predicted -- drag what is wrong.\n"
            "Works DURING a count: the count is paused while you draw and continues from\n"
            "exactly where it stopped, so nothing is lost and nothing is recounted."
        )
        self.draw_button.clicked.connect(lambda: self.start("draw"))

        # ONE BUTTON, THREE JOBS. Alice, 2026-08-20: *"can you make the remove from queue button
        # the same as the count cells/add to queue/remove from queue"*. Removing used to be a
        # second button sitting beside this one, greyed out nearly all the time; now it is a third
        # thing this button can say. What it says is what pressing it does -- see
        # count_button_action, which the label and the click both ask.
        self.count_button = QPushButton("Count cells")
        self.count_button.setToolTip(
            "Counts the selected sections one after another. Shift-click or cmd-click in the\n"
            "list to pick several and leave it running.\n"
            "You can also press it again WHILE a count runs -- the section joins the queue\n"
            "behind the one being counted, so the night's work does not have to be decided\n"
            "in advance.\n"
            "Once a section is waiting, this button turns into Remove from queue: it takes\n"
            "back sections that have not started yet. The one being counted now is ended by\n"
            "closing the app, which asks first, and nothing already counted is affected.\n"
            "Never two at once: each count needs 1.5-2 GB, and this machine has 8 GB."
        )
        self.count_button.clicked.connect(self.start_count)

        # IT USED TO SAY "read-only: nothing here can change a count", and the tooltip was the
        # honest description of a limitation rather than a feature. Alice, 2026-09-12: *"could
        # you change the app so that review cells lets the user add and delete cells"*. Seeing a
        # wrong cell and not being able to fix it leaves the wrong number on the report, so the
        # button now says what it can do instead of what it cannot.
        self.review_button = QPushButton("Review cells")
        self.review_button.setToolTip(
            "Open the section with every counted cell marked, coloured by region.\n"
            "The least confident calls get their own layer -- look at those first.\n"
            "Add the cells it missed and delete the ones that are not cells: Ctrl-A to\n"
            "add, Ctrl-E then Delete to remove. It saves when you close it, and a visit\n"
            "that changed nothing changes nothing on disk.\n"
            "Works DURING a count: the count is paused while the window is open and\n"
            "continues from exactly where it stopped."
        )
        self.review_button.clicked.connect(lambda: self.start("review"))

        # Opens the whole folder's numbers in their own window, where the copy and save buttons
        # live. It sits at the other end of the row from draw/count/review because those act on
        # what is SELECTED and this does not -- a person should not need a tooltip to tell which.
        self.all_data_button = QPushButton("View all data")
        self.all_data_button.setToolTip(
            "Show every counted region in this folder in one table, with a second tab\n"
            "averaging each bird's sections. Copy for Sheets and Save CSV are in there,\n"
            "so what you copy is what you can see."
        )
        self.all_data_button.clicked.connect(self.show_all_data)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Menlo", 10))
        # Bounded: a count prints a line per section and a long session would otherwise grow
        # the pane without limit. The interesting output is always the most recent.
        self.output.setMaximumBlockCount(2000)

        buttons = QHBoxLayout()
        buttons.addWidget(self.draw_button)
        buttons.addWidget(self.count_button)
        buttons.addWidget(self.review_button)
        buttons.addStretch(1)
        buttons.addWidget(self.all_data_button)
        buttons.addWidget(self.refresh_button)

        right = QVBoxLayout()
        right.addWidget(self.heading)
        right.addWidget(self.note)
        right.addWidget(self.numbers, 1)
        right.addLayout(buttons)

        right_panel = QWidget()
        right_panel.setLayout(right)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.section_list)
        top.addWidget(right_panel)
        top.setSizes([420, 760])

        # A splitter rather than fixed widths: section names are long and vary, and the
        # person should be able to give either side more room without the layout fighting.
        whole = QSplitter(Qt.Vertical)
        whole.addWidget(top)
        whole.addWidget(self.output)
        whole.setSizes([540, 220])

        # Above the log rather than inside it. A job's status has to be visible without
        # scrolling, and the log scrolls -- a "counting..." line appended there is gone the
        # moment count_cells.py prints its first message.
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setFont(QFont("", 12, QFont.Bold))

        self.summary = QLabel("")

        layout = QVBoxLayout(self)
        layout.addLayout(folder_bar)
        layout.addWidget(whole, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.summary)

    # ---- choosing a folder ----

    def choose_folder(self):
        """Ask for a folder, check it, and use it if it holds anything.

        The check happens BEFORE the folder is remembered, and a folder that cannot be
        written to is reported but still accepted. Refusing outright would be wrong: reading
        someone else's counts is a legitimate thing to want, and the app can show every
        number in a read-only folder perfectly well. What it cannot do is save new work
        there, which is what the warning says.
        """
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose the folder holding your images", str(self.root)
        )

        if not chosen:
            return

        report = app_folder.inspect(chosen)

        self.log("")
        for line in app_folder.describe(report):
            self.log(line)

        if not report["images"]:
            # Nothing found, so the previous folder is kept. Switching to an empty folder
            # would replace a working list with a blank one and look like a crash.
            self.log("Keeping the previous folder, since that one has nothing to count.")

            return

        self.root = Path(chosen)

        if not app_folder.remember(self.root):
            self.log(
                f"Note: could not save this choice to {app_folder.SETTINGS_PATH}, so the "
                "app will start in its own folder next time."
            )

        self.report_folder()
        self.refresh()

    def report_folder(self):
        """Show the current folder, and how many sections are in it."""
        self.folder_label.setText(str(Path(self.root).resolve()))

    # ---- the all-data window ----

    def show_all_data(self):
        """Open the compiled table, or bring it forward if it is already open.

        ONE WINDOW, REUSED. Clicking the button four times should not leave four stale copies
        of the same numbers on screen, each drifting further out of date. The window is created
        on first use rather than in __init__ so that starting the app costs nothing for a
        feature many sessions never touch.
        """
        if self.all_data is None:
            self.all_data = AllData()

        self.all_data.show_table(self.table, self.root)
        self.all_data.show()

        # raise_ and activateWindow together: the first lifts it above the other windows, the
        # second gives it the keyboard. Without both, a click on the button while the window is
        # already open behind this one appears to do nothing at all.
        self.all_data.raise_()
        self.all_data.activateWindow()

    def update_all_data(self):
        """Push fresh numbers into the window if it happens to be open.

        Called from refresh(), so a count landing mid-batch updates a window left open on a
        second screen. Silent when it is closed: this must never be the reason a refresh fails.
        """
        if self.all_data is not None and self.all_data.isVisible():
            self.all_data.show_table(self.table, self.root)

    # ---- reading the disk ----

    def refresh(self):
        """Re-read everything from disk, then redraw. The only way state enters this window.

        SLOW ON PURPOSE-ISH: 1-2 seconds here, because it stats every sidecar and reads every
        counts and areas file rather than trusting anything it remembers. That is the price of
        "the files are the state", and it is only paid when something has actually changed --
        at startup, on Refresh, after a folder change, and when a job has finished writing.
        Anything that merely needs the window redrawn calls draw_list() instead, which touches
        no disk and is instant.
        """
        # Notes are collected rather than printed, because find_sections' print() goes to a
        # terminal nobody launched this from. The log pane is where a person can read them.
        notes = []
        self.sections = find_sections(self.root, notes)
        self.table = build_table(self.sections)

        # After the files, before anything is drawn: a section being counted right now looks
        # identical on disk to one nobody has touched, and only the process list can tell them
        # apart. Done here rather than in find_sections because app_state is about files and
        # would have to stay about files even if this window did not exist.
        self.find_outside_counts()

        # A restored queue is a list of sections still WANTING a count, so anything that has since
        # got one stops being part of it -- counted in a terminal, counted by a job left running,
        # counted from the other app window. Without this the row would go on saying "in the queue
        # from last time" beside a section that is finished.
        # "COUNTED SINCE THE QUEUE WAS SAVED", not "has a count" -- the same test app_saved_queue
        # used to decide what to offer, so the two cannot disagree. A section she deliberately
        # queued for RE-counting already has an old count file, and judging on that would erase
        # her from the list for asking.
        by_stem = {section["stem"]: section for section in self.sections}
        self.saved_stems = [
            stem
            for stem in self.saved_stems
            if stem in by_stem
            and not app_saved_queue.counted_since(by_stem[stem], self.saved_at)
        ]

        for note in notes:
            self.log(note)

        self.draw_list()
        self.show_summary()
        self.section_changed(self.section_list.currentRow())
        self.update_all_data()

    def save_queue(self):
        """Write down what is being counted and what is waiting, so a quit cannot lose it.

        Alice, 2026-08-20: *"since when i reopened it i had to add all the images to queue
        again"*. Quitting always ended the queue -- see closeEvent, which can leave the running
        count alive but has no way to keep a list that lives in this window's memory.

        THE RUNNING SECTION IS INCLUDED, which looks wrong and is not. If the count is stopped on
        the way out it does need redoing, and if it is left running app_saved_queue drops it on
        the way back -- either because it is still going, or because its file is newer than the
        queue. Storing it costs nothing and leaving it out would lose it in the case that
        matters.

        SECTIONS RESTORED BUT NOT STARTED COUNT TOO. If she brings back a queue of eight, chooses
        "keep them", then counts one of them, the other seven are still hers and must stay written
        down -- an earlier version wrote only what the runner held and quietly shortened the queue
        to one.
        """
        running = [self.runner.stem] if self.runner.busy() and self.runner.stem else []
        waiting = [section["stem"] for section in self.runner.queue]

        if self.runner.kind != "count":
            running, waiting = [], []

        # dict.fromkeys rather than a set: it removes repeats and keeps the order, and the order is
        # the decision. A stem can be in two of these lists at once only by accident, but writing
        # one twice would offer it twice on the way back.
        stems = list(dict.fromkeys(running + waiting + list(self.saved_stems)))

        app_saved_queue.remember(self.root, stems)

    def offer_saved_queue(self):
        """On opening: bring back the queue that was waiting last time, and ask what to do.

        NOTHING STARTS BY ITSELF. Opening the app to look up one number must not commit the
        machine to five hours, so the sections are selected and the question is asked. The point
        of the feature is not to save the counting, it is to save the CHOOSING.

        Called from a timer rather than from __init__ so the window is painted before a dialog
        appears in front of it.
        """
        stems, saved_at = app_saved_queue.remembered(self.root)

        if not stems:
            return

        self.saved_at = saved_at

        to_offer, dropped = app_saved_queue.restorable(
            stems,
            self.sections,
            saved_at,
            self.outside_counts,
            self.runner.stem if self.runner.busy() else None,
        )

        self.log("")
        for line in app_saved_queue.describe(to_offer, dropped):
            self.log(line)

        if not to_offer:
            app_saved_queue.forget()

            return

        # MARKED AND SELECTED FIRST, so the queue is visible in the list behind the dialog. Being
        # asked about eight names in a message box is not the same as seeing which rows they are.
        self.saved_stems = [section["stem"] for section in to_offer]
        self.select_stems([section["stem"] for section in to_offer])
        self.draw_list()

        answer = self.ask_about_saved_queue(to_offer)

        if answer == "forget":
            app_saved_queue.forget()
            self.saved_stems = []
            self.draw_list()
            self.log("Forgotten. The sections are still selected if you want them.")

            return

        if answer == "later":
            # The marks stay. Alice, 2026-08-20: *"if i close and reopen can it still say in
            # queue"* -- the list is where she looks, and a selection alone disappears the moment
            # anything else is clicked.
            self.log("Left in the queue. Press Count cells when you want to start.")

            return

        self.log("")
        for line in app_queue.describe(*app_queue.plan(to_offer, self.outside_counts)):
            self.log(line)

        # The marks go now: from here the runner owns these sections, and its own "waiting in the
        # queue" is the truthful label -- something IS working towards them.
        self.saved_stems = []

        if not self.runner.run_many("count", to_offer):
            self.log("a job is already running")

    def ask_about_saved_queue(self, to_offer):
        """Start it, keep it selected, or forget it. Returns start/later/forget.

        A METHOD OF ITS OWN so the restore can be tested without a person clicking, the same
        arrangement as ask_about_running_count.

        THE START BUTTON DISAPPEARS WHEN SOMETHING ELSE IS COUNTING. Two counts at once exhaust
        this machine's memory and finish later than two run in turn -- that is why a batch is
        sequential (app_jobs.run_many has the measurements). A count started outside this window
        is outside the runner's reach, so the only way to keep that rule here is not to offer the
        button.

        DELIBERATELY STRICTER THAN THE COUNT BUTTON, which will still start a count beside an
        outside one if she asks it to. The difference is who raised the question: this dialog
        appears on its own, unasked, in front of someone who may only have opened the app to read
        a number, and an unprompted offer should not be the easy route to the one arrangement
        this machine cannot afford. The wording says the offer is withheld, not that the app
        refuses -- claiming a rule the button does not keep would be worse than the collision.
        """
        waiting = len(to_offer)

        box = QMessageBox(self)
        box.setWindowTitle("The queue from last time")
        box.setText(
            f"{waiting} section{'s' if waiting > 1 else ''} "
            f"{'were' if waiting > 1 else 'was'} still waiting to be counted when the app last "
            "closed."
        )

        outside = len(self.outside_counts)

        box.setInformativeText(
            f"They are selected in the list. Counting them would take "
            f"{app_queue.estimate(waiting)}.\n\n"
            + (
                f"{outside} count{'s' if outside > 1 else ''} started outside this window "
                f"{'are' if outside > 1 else 'is'} still running, so starting these is not "
                "offered here -- two counts at once exhaust this machine's memory and finish "
                "later than two run in turn. Wait for it to finish, then press Count cells."
                if outside
                else "Nothing is counting right now."
            )
        )

        start = None

        if not outside and not self.window_runner.busy():
            start = box.addButton("Count them now", QMessageBox.AcceptRole)

        later = box.addButton("Just keep them selected", QMessageBox.AcceptRole)
        box.addButton("Forget the queue", QMessageBox.DestructiveRole)

        # Keeping them selected is the default: it is the answer that decides nothing.
        box.setDefaultButton(later)
        box.exec()

        clicked = box.clickedButton()

        if start is not None and clicked is start:
            return "start"

        if clicked is later:
            return "later"

        return "forget"

    def select_stems(self, stems):
        """Select exactly these sections in the list, by name.

        EXACTLY THESE, so the old selection is cleared first. setCurrentRow does NOT do that in an
        ExtendedSelection list -- it adds the row to whatever was already highlighted (Qt selects
        the current index with SelectCurrent, which is Select|Current, not ClearAndSelect). Without
        the clear, restoring a queue of three onto a row someone had left selected would silently
        offer to count four, and the extra one is invisible unless the list is scrolled to it.
        Measured 2026-08-20 by check_app_queue.py section 8, which asked for one row and got three.

        CURRENT ROW BEFORE THE REST, because setCurrentRow moves the highlight and the rest of the
        selection has to survive it. The same trap draw_list documents.
        """
        wanted = set(stems)
        rows = [index for index, section in enumerate(self.sections) if section["stem"] in wanted]

        if not rows:
            return

        self.section_list.blockSignals(True)
        self.section_list.clearSelection()
        self.section_list.setCurrentRow(rows[0])

        for row in rows:
            self.section_list.item(row).setSelected(True)

        self.section_list.blockSignals(False)

        self.section_changed(rows[0])

    def find_outside_counts(self):
        """Look in the process list for counts this window did not start. True if that changed.

        The window's own count is taken out again. The runner already knows about that one and
        the list marks it its own way, and calling it "started outside this window" would be
        false in the one place someone is looking for the truth.

        Changes are LOGGED, both ways. A count appearing is news at startup -- it is the answer
        to "which one is still going" -- and a count disappearing is the moment its numbers
        exist, which is the only thing anyone was waiting for.
        """
        found = app_running.counting_now(self.sections)

        if self.runner.busy():
            found.pop(self.runner.stem, None)

        before = set(self.outside_counts)
        self.outside_counts = found

        if set(found) == before:
            return False

        for stem in sorted(before - set(found)):
            self.log(f"\nThe count of {stem}, which was started outside this window, has ended.")

        appeared = {stem: about for stem, about in found.items() if stem not in before}

        if appeared:
            self.log("")

        for line in app_running.describe(appeared):
            self.log(line)

        return True

    def check_outside_counts(self):
        """The 15-second poll. Cheap unless something actually changed.

        A finish and an appearance are handled differently on purpose. When an outside count
        ENDS it has just written its files, so the whole table has to be re-read -- that is the
        moment the numbers show up on their own, with nobody pressing Refresh. When one merely
        appears, no file has changed, so redrawing the list is enough and the 1-2 seconds of
        disk reading is not spent.
        """
        before = set(self.outside_counts)

        if not self.find_outside_counts():
            return

        if before - set(self.outside_counts):
            self.refresh()

            return

        self.draw_list()

        section = self.current_section()

        if section is not None:
            # So the Count button stops offering to start a second count of that section.
            self.update_buttons(section)

    def draw_list(self):
        """Redraw the section list from what is already in memory. Reads nothing.

        Separate from refresh() because the list has to be redrawn at moments when the disk has
        not changed at all -- when a batch moves to its next section, the queue markers move
        with it, and re-reading twenty sections' files to learn something the runner just told
        us would freeze the window for a second or two each time.

        Keeps the selection across the redraw, by stem rather than by row: statuses change as
        counts land, and losing the selection after every job would be maddening.

        THE WHOLE SELECTION, not just the current row, because a batch redraws after EVERY
        section. Keeping only the current row would empty a five-section selection as soon as
        the first one finished, and the person watching would have no way to tell whether the
        queue had been lost with it.

        The selection is read from the widget directly rather than through selected_sections(),
        whose fallback to the current row would invent a selection that was never made and then
        preserve it forever.
        """
        wanted = self.current_stem()
        selected = {
            self.sections[index.row()]["stem"]
            for index in self.section_list.selectedIndexes()
            if 0 <= index.row() < len(self.sections)
        }

        self.section_list.blockSignals(True)
        self.section_list.clear()

        # The queue, shown in the list itself. A batch is hours long and the log scrolls away,
        # so the list is the only place a person can check at any moment which sections are
        # still coming -- and notice, before the second hour, that the wrong ones were picked.
        running = self.runner.stem if self.runner.busy() else None
        queued = self.runner.queued_stems()

        # The section with a napari window open, and which kind. Both are needed: "drawing now"
        # and "reviewing now" mean different things to someone glancing at the list -- one is
        # about to change the outlines, the other cannot change anything.
        open_stem = self.window_runner.stem if self.window_runner.busy() else None
        open_kind = self.window_runner.kind if open_stem else None
        opened = "drawing" if open_kind == "draw" else "reviewing"

        for section in self.sections:
            stem = section["stem"]
            colour = STATUS_COLOURS.get(section["status"], "#000000")
            note = ""

            # The status WORD, which is normally the section's file-derived status. A count in
            # progress replaces it, because "ready to count" is actively misleading about a
            # section that is being counted -- it reads as an invitation to press the button.
            # Alice, 2026-08-20: *"can you change their status to counting"*.
            label = section["status"]

            # BOTH AT ONCE IS ITS OWN CASE, and the interesting one: drawing the section that is
            # being counted means the paused count will finish using the outlines it started
            # with, not the ones now on screen. The row says so rather than reporting only the
            # drawing, which would hide it. (The count that lands is then marked stale by
            # app_table, because the outlines are newer than it -- so nothing silently disagrees.)
            # REVIEWING THE SECTION BEING COUNTED IS NOW ALSO ITS OWN CASE. It used to change
            # nothing, so it needed no note; since review can add and delete cells, the count that
            # is about to rewrite the cell file will take any edits with it.
            if stem == open_stem and stem == running:
                colour = RUNNING_COLOUR
                label = "counting"
                note = (
                    "  <- drawing now; its paused count keeps the OLD outlines"
                    if open_kind == "draw"
                    else "  <- reviewing now; the count will overwrite any cell edits"
                )
            elif stem == open_stem:
                colour = RUNNING_COLOUR
                note = f"  <- {opened} now"
            elif stem == running:
                colour = RUNNING_COLOUR
                label = "counting"
                # Said in the list as well as the status line, because a count that is deliberately
                # held still looks exactly like one that has died, and the list is what someone
                # glances at.
                if not self.pause.paused():
                    note = "  <- counting now"
                elif open_kind == "draw":
                    note = "  <- counting, paused while you draw"
                else:
                    note = "  <- counting, paused while you review"
            elif stem in self.outside_counts:
                # A count nobody in this window started: a terminal, another app window, or an
                # app that has since quit and left its count running. Marked the same green as
                # this window's own count, because the section IS being counted -- how it got
                # started is the note's business, not the colour's.
                colour = RUNNING_COLOUR
                label = "counting"

                # Held by THIS window's napari, even though another window started it -- see
                # count_to_hold. Said here for the same reason as above: from the outside a stopped
                # count is indistinguishable from a dead one, and this row is the only place the
                # section appears.
                if self.pause.paused() and stem == self.held_stem:
                    note = "  <- counting, paused by this window until napari closes"
                else:
                    note = "  <- counting now, started outside this window"

                # No elapsed minutes here, on purpose. This list is redrawn only when something
                # changes, so a number in it would freeze at "17 min so far" for an hour. How
                # long it has been going is in the panel on the right, which is rebuilt every
                # time the section is clicked, and therefore cannot be stale when it is read.
            elif stem in queued:
                colour = QUEUED_COLOUR
                note = "  <- waiting in the queue"
            elif stem in self.saved_stems:
                # The queue from before the app closed, still a queue. Alice, 2026-08-20: *"if i
                # close and reopen can it still say in queue"*.
                #
                # WORDED DIFFERENTLY FROM THE LINE ABOVE, and the difference is the whole point.
                # "waiting in the queue" means a count is working towards it; this one means the
                # list survived but nothing is running, so it will sit there until she presses
                # the button. Reusing the same words would promise counting that is not
                # happening -- worse than losing the queue, because it looks fine.
                colour = QUEUED_COLOUR
                note = "  <- in the queue from last time -- press Count cells to start"

            # The status goes in the item's text after a newline, and its colour in a data role.
            # SectionDelegate splits them and draws the second line in italics. The indent is
            # the delegate's, not spaces, so it does not depend on the font being monospaced.
            item = QListWidgetItem(f"{stem}\n{label}{note}")
            item.setData(STATUS_COLOUR_ROLE, colour)
            self.section_list.addItem(item)

        # The current row FIRST and the rest of the selection after it. setCurrentRow replaces
        # the whole selection with that one row, so doing it second would wipe out everything
        # just restored -- which is exactly the bug this block exists to avoid.
        stems = [section["stem"] for section in self.sections]

        if wanted in stems:
            self.section_list.setCurrentRow(stems.index(wanted))
        elif self.sections:
            self.section_list.setCurrentRow(0)

        for index, stem in enumerate(stems):
            if stem in selected:
                self.section_list.item(index).setSelected(True)

        self.section_list.blockSignals(False)

    def current_stem(self):
        row = self.section_list.currentRow() if hasattr(self, "section_list") else -1

        return self.sections[row]["stem"] if 0 <= row < len(self.sections) else None

    def current_section(self):
        row = self.section_list.currentRow()

        return self.sections[row] if 0 <= row < len(self.sections) else None

    def selected_sections(self):
        """Every selected section, in list order, falling back to the current one.

        List order rather than click order, so a batch runs top to bottom as shown. Clicking
        order is invisible after the fact and would make the queue's order unpredictable.

        The fallback matters: clicking a row makes it both current and selected, so the two
        agree in normal use. They come apart after a refresh restores a selection, and an
        empty selection with a valid current row should still count that row rather than
        refusing with "nothing selected" while a section is plainly highlighted.
        """
        rows = sorted(index.row() for index in self.section_list.selectedIndexes())
        chosen = [self.sections[row] for row in rows if 0 <= row < len(self.sections)]

        if chosen:
            return chosen

        current = self.current_section()

        return [current] if current else []

    def selected_stems(self):
        return {section["stem"] for section in self.selected_sections()}

    def selection_changed(self):
        """Keep the count button honest about how many sections it will run."""
        section = self.current_section()

        if section is not None:
            self.update_buttons(section)

    def show_summary(self):
        """The one line worth quoting: mean density per region over current sections."""
        summary = summarise_by_region(self.table)

        if summary is None:
            self.summary.setText(
                "No current densities yet -- every counted section was counted before its "
                "outlines were last redrawn."
            )

            return

        parts = [
            f"{region} {row['mean']:.1f}/mm2 (n={int(row['count'])})"
            for region, row in summary.iterrows()
        ]

        self.summary.setText("mean density, current sections only:   " + "    ".join(parts))

    # ---- showing one section ----

    def section_changed(self, row):
        section = self.current_section()

        if section is None:
            self.heading.setText("no section selected")
            self.numbers.setRowCount(0)
            self.note.setText(
                "No images in this folder or any folder inside it. Use Choose folder to "
                "point the app at your .tif files."
            )

            return

        self.heading.setText(section["stem"])

        rows = self.table[self.table["stem"] == section["stem"]]
        stale = bool(rows["stale"].any()) if len(rows) else False

        self.note.setText(self.note_for(section, rows, stale))
        self.fill_numbers(rows, stale)
        self.update_buttons(section)

    def note_for(self, section, rows, stale):
        """The sentence under the heading: what this section needs next, in plain words.

        Written as guidance rather than a status word, because the status is already visible
        in the list and a person who has just clicked a section wants to know what to DO.

        A COUNT IN PROGRESS COMES FIRST, ahead of stale and ahead of everything the files say,
        because it is the only thing here that is about to change all of them. It goes in this
        panel rather than in the list because the panel is rebuilt on every click, so the
        elapsed minutes are read at the moment they are true.
        """
        about = self.outside_counts.get(section["stem"])

        if about:
            minutes = about["minutes"]
            left = ""

            if 0 <= minutes < app_queue.MINUTES_EACH:
                left = f", so roughly {app_queue.MINUTES_EACH - minutes} to go"

            return (
                "COUNTING NOW, by a job started outside this window"
                + (f" -- {minutes} minutes in{left}" if minutes >= 0 else "")
                + f" (process {about['pid']}). It will save its numbers when it finishes, "
                "whether this window is open or not. Do not count it again in the meantime."
            )

        if stale:
            return (
                "STALE -- these counts were made before the outlines were last redrawn, "
                "so the densities are wrong. Count again to bring them up to date."
            )

        if section["status"] == NEEDS_REGIONS:
            return "No outlines yet. Draw NCM and CMM, then click the dorsal marker."

        if section["status"] == NEEDS_DORSAL:
            return (
                "Outlines are drawn but there is no dorsal marker, so NCM cannot be split "
                "into dNCM and vNCM. Draw again and add the point."
            )

        if section["status"] == READY_TO_COUNT:
            missing = not areas_path(section["path"]).exists()

            return (
                "Ready to count."
                + (" Areas will be computed on the first count." if missing else "")
            )

        return "Counted. The numbers below are current."

    def fill_numbers(self, rows, stale):
        """One row per region. Blanks stay blank rather than becoming zero.

        pd.isna, not `is None`: pandas stores a missing number as NaN in a numeric column, so
        `is None` silently never matches and every empty cell would read as present. This
        exact bug was in app_table.py first.
        """
        self.numbers.setRowCount(len(ps6.REGION_NAMES))

        by_region = {row["region"]: row for _, row in rows.iterrows()}

        for index, name in enumerate(ps6.REGION_NAMES):
            row = by_region.get(name)

            def cell(value, formatter):
                text = "-" if row is None or pd.isna(value) else formatter(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                if stale:
                    item.setForeground(QColor(STALE_COLOUR))

                return item

            self.numbers.setItem(index, 0, QTableWidgetItem(name))
            self.numbers.setItem(
                index, 1, cell(row["cells"] if row is not None else None, lambda v: f"{int(v)}")
            )
            self.numbers.setItem(
                index,
                2,
                cell(row["area_mm2"] if row is not None else None, lambda v: f"{v:.3f}"),
            )
            self.numbers.setItem(
                index,
                3,
                cell(
                    row["density_per_mm2"] if row is not None else None,
                    lambda v: f"{v:.1f}",
                ),
            )

    def update_buttons(self, section):
        """Enable what is possible, and highlight what to do next.

        Disabling rather than letting a click fail: count_section would refuse with a
        readable message anyway, but a button that cannot work should not look like it can.
        """
        counting = self.runner.busy()
        window_open = self.window_runner.busy()
        busy = counting or window_open

        # WHAT THE ONE BUTTON DOES is decided by count_button_action, which the click asks too --
        # see there for the precedence between counting, adding and removing. Asked once and used
        # for both the label and the enabling, so a button cannot be greyed out for one reason
        # while its label describes another.
        chosen = self.selected_sections()
        action, targets = self.count_button_action(chosen)

        # The label says exactly what the click will do, because with a multi-select list
        # "Count cells" is ambiguous in the one case that costs hours if misread. Only the
        # sections that will be acted on are named: a selection of five where two lack outlines
        # reads "Count 3 sections", which is the truth and prompts a look at why.
        if action == "remove":
            # No "nothing to remove" case: this word only appears when there is something to take
            # out, otherwise the button is still offering to count.
            if len(targets) == 1:
                self.count_button.setText("Remove from queue")
            else:
                self.count_button.setText(f"Remove {len(targets)} from queue")
        elif action == "add":
            # "Count cells" would be a lie mid-batch -- nothing starts now, it starts in forty
            # minutes. The word is what tells her the queue is joinable at all.
            if not targets:
                self.count_button.setText("Nothing to add")
            elif len(targets) == 1:
                self.count_button.setText("Add to queue")
            else:
                self.count_button.setText(f"Add {len(targets)} to queue")
        elif len(chosen) <= 1:
            self.count_button.setText("Count cells")
        elif not targets:
            # "Count 0 sections" is technically true and reads as a bug. The button is disabled
            # either way; this says why in the same glance.
            self.count_button.setText("Nothing to count")
        else:
            self.count_button.setText(f"Count {len(targets)} sections")

        # DRAW AND REVIEW ARE THE BUTTONS A RUNNING COUNT DOES NOT DISABLE. Alice, 2026-08-20:
        # *"is it possible for me to draw outlines while cell counting is going on"* and *"why
        # can't i review cells for the images that are already counted"*. Both work because the
        # count is SUSPENDED for the duration rather than made to share the machine -- see
        # open_napari and app_pause.py. What is refused is a SECOND napari window, and only
        # because there is one napari and one pause to give it.
        self.draw_button.setEnabled(not window_open)

        # COUNT STAYS ON WHILE COUNTING, because there it means "add to the queue" and adding
        # starts nothing -- it appends to a list. What must stay off is starting a count into an
        # open drawing window with nothing already running: that is the memory collision from
        # the other end, and the count would be the thing swapped out that time. Adding to a
        # queue whose count is PAUSED for drawing is still safe, since the addition only becomes
        # a running process once the pause is over and the current section has finished.
        #
        # REMOVING IS NEVER REFUSED. It starts nothing, needs no memory and gives time back, so
        # neither an open drawing window nor a count elsewhere is a reason to withhold it -- and a
        # greyed-out "Remove from queue" would be the one state of this button that cannot be acted
        # on, which is what made it worth folding in rather than leaving beside.
        if action == "remove":
            self.count_button.setEnabled(True)
        else:
            self.count_button.setEnabled(bool(targets) and (counting or not window_open))

        # Reviewing needs counted cells to look at, not just outlines -- a different condition
        # from counting, so it gets its own test. Deliberately about the CURRENT section only,
        # not the selection: review opens a napari window, and eight of those is not a batch,
        # it is a mistake.
        #
        # `window_open`, NOT `busy`: a running count no longer blocks this. That it used to is what
        # made the button look like it was reporting missing counts.
        self.review_button.setEnabled(not window_open and section["has_counts"])
        self.refresh_button.setEnabled(not busy)

        # Disabled while a job runs for the same reason as the others: switching folders
        # mid-count would rebuild the list under the running job and leave the person looking
        # at a different section from the one being counted.
        self.folder_button.setEnabled(not busy)

        # View all data is deliberately absent from this method. It only reads the table already
        # in memory, so it is safe during a count -- and useful there: after two sections of
        # eight, looking at what has landed is a reasonable thing to want, and refresh() keeps
        # the window current as the rest arrive.

        suggested = NEXT_ACTION[section["status"]]

        # The suggestion comes from app_actions.NEXT_ACTION rather than being re-derived from
        # the status here, so there is one table saying what follows what.
        self.draw_button.setDefault(suggested == "draw")

        # Not while it says Remove: the default button is the one Enter presses, and "the obvious
        # next thing to do with this section" is never to un-queue it -- it is queued because
        # someone asked for it.
        self.count_button.setDefault(suggested == "count" and action != "remove")
        self.review_button.setDefault(suggested == "review")

    def count_button_action(self, chosen=None):
        """What the one button would do right now: ("count"|"add"|"remove", the sections).

        THE LABEL AND THE CLICK ARE THE SAME DECISION, so it is made once, here, and both
        update_buttons and start_count ask. A second copy of this rule in the handler is exactly
        how a button comes to read "Remove from queue" and start a five-hour count instead.

        ADDING WINS WHILE THERE IS ANYTHING TO ADD. That is the whole of the precedence, and it is
        what keeps a mixed selection unambiguous: two sections already waiting and one not reads
        "Add 1 to queue", and deselecting the odd one out turns it into "Remove 2 from queue".
        The button never has to guess which half of a selection was meant.

        WHAT THIS MEANS WHEN NOTHING IS RUNNING: a section restored from last time is countable,
        so the button offers to COUNT it -- which is what its own row in the list says to do --
        rather than to take it off the list. The launch dialog's "Forget them" is how that list is
        dropped wholesale. Once a count is running, plan_addition rules out whatever is already
        waiting, so removing is the only thing left to offer and the button says so.
        """
        chosen = self.selected_sections() if chosen is None else chosen

        # WHILE A COUNT RUNS the button adds to that queue instead of starting anything, so what
        # it can act on is a different and smaller list: not "could be counted" but "is not
        # already being counted". Worked out by the same module that will do it, so the label
        # cannot promise something the click then refuses.
        if self.runner.busy():
            action = "add"
            targets, _ = app_queue.plan_addition(
                chosen, self.runner.stem, self.runner.queued_stems(), self.outside_counts
            )
        else:
            action = "count"
            targets, _ = app_queue.plan(chosen, self.outside_counts)

        if targets:
            return action, targets

        # Nothing to count or add, so removing is what is left -- if any of the selection is in
        # fact waiting. queued_in covers both kinds of waiting and excludes the section being
        # counted now; see there.
        waiting = self.queued_in(chosen)

        if waiting:
            return "remove", waiting

        # Neither. The action still comes back, because the label for "nothing to add" and the
        # label for "nothing to count" are different sentences.
        return action, []

    # ---- running jobs ----

    def start_count(self):
        """Whatever the one button currently says: count these, add these, or un-queue these.

        The plan is shown and CONFIRMED before a batch of more than one starts. Not because
        clicking is untrustworthy, but because there is no cancel: once started, the only way
        out is quitting the app, and selecting five sections instead of one is a two-pixel
        mistake that costs three hours. One section starts immediately, as it always did --
        that is not a commitment worth a dialog.
        """
        chosen = self.selected_sections()

        # A count already running means this click ADDS rather than starts. Branching here, at
        # the top, rather than letting run_many refuse at the bottom: "a job is already running"
        # was a true message and a useless one -- the answer to "count this one too" is yes.
        # Asked again HERE, moments before anything starts, rather than trusting the 15-second
        # poll. The accident this prevents is small in time and large in cost: two app windows
        # open, a count started in one, and this button pressed in the other inside the same
        # quarter-minute. Both jobs would then write the same file.
        if self.find_outside_counts():
            self.draw_list()

        # Which of the three it is, decided AFTER that check rather than read off the label: a
        # count discovered in another window changes what this selection can do, and the label was
        # drawn up to fifteen seconds ago. count_button_action is the same call update_buttons
        # made, so the answer matches what she read unless the machine changed underneath it.
        if self.count_button_action(chosen)[0] == "remove":
            self.remove_from_queue()

            return

        if self.runner.busy():
            self.add_to_queue(chosen)

            return

        to_count, skipped = app_queue.plan(chosen, self.outside_counts)

        self.log("")
        for line in app_queue.describe(to_count, skipped):
            self.log(line)

        if not to_count:
            return

        if len(to_count) > 1 and not self.confirm_batch(to_count):
            self.log("Cancelled -- nothing started.")

            return

        # Handed over to the runner, so they stop being "the queue from last time" and become the
        # queue, full stop. See draw_list for why the two are worded differently.
        taken = {section["stem"] for section in to_count}
        self.saved_stems = [stem for stem in self.saved_stems if stem not in taken]

        if not self.runner.run_many("count", to_count):
            self.log("a job is already running")

    def add_to_queue(self, chosen):
        """Put more sections behind the one being counted.

        The drawing runner is not consulted: this cannot start anything, so an open drawing
        window is not a reason to refuse. See update_buttons.

        What gets LOGGED is what the runner took, not what was asked for. The runner drops a
        stem that is already queued, and a message built from the request would then name a
        section that is not in the queue -- the kind of small lie that makes someone stop
        trusting the log pane.
        """
        to_add, skipped = app_queue.plan_addition(
            chosen, self.runner.stem, self.runner.queued_stems(), self.outside_counts
        )

        if len(to_add) > 1 and not self.confirm_addition(to_add):
            self.log("Cancelled -- the queue is unchanged.")

            return

        added = self.runner.add(to_add)

        # Same handover as in start_count: once the runner has them, its own label is the true one.
        taken = {section["stem"] for section in added}
        self.saved_stems = [stem for stem in self.saved_stems if stem not in taken]

        self.log("")
        for line in app_queue.describe_addition(added, skipped, self.runner.remaining()):
            self.log(line)

        if not added:
            return

        # The status line's total came from the `started` signal, so it has to be told the batch
        # got longer or it goes on saying "1 of 1" while three sections wait.
        self.job_total = self.runner.total
        self.show_status()

        # Written down at once, not at closing time. A force-quit, a power cut and a kill all
        # skip closeEvent, and those are exactly the moments a queue is long.
        self.save_queue()

        # And the list has to be redrawn, or the sections just added are not marked as waiting
        # -- which is the only confirmation on screen that the click did anything at all.
        self.draw_list()

        section = self.current_section()

        if section is not None:
            self.update_buttons(section)

    def queued_in(self, chosen):
        """Which of these sections are waiting for a count. A list, in the order given.

        BOTH KINDS OF WAITING. `runner.queued_stems()` is the queue a count is working through;
        `saved_stems` is the one restored from the last time the app was open, which nothing is
        working through yet. In the list they read differently on purpose (see draw_list) but to
        a person they are the same fact -- "this is going to be counted" -- so removing has to
        cover both or the button would do nothing on half the rows that look queued.

        The section being counted RIGHT NOW is not included: it is not in either list, and
        stopping it is a different decision with a different cost. closeEvent owns that one.
        """
        waiting = self.runner.queued_stems()

        return [
            section for section in chosen
            if section["stem"] in waiting or section["stem"] in self.saved_stems
        ]

    def remove_from_queue(self):
        """Take the selected sections out of the queue. Counts nothing, stops nothing.

        Alice, 2026-08-20: *"can i also remove things from queue"*. Until now the queue only
        grew: a section added by mistake meant either forty wasted minutes or closing the app,
        which drops every other section waiting with it.

        REACHED FROM THE COUNT BUTTON, which turns into Remove from queue when the selection is
        already waiting and there is nothing left to add -- count_button_action decides, start_count
        dispatches. Kept as its own method rather than folded into that dispatch because it is a
        different kind of act with a different reason not to ask first, written below.

        NO CONFIRMATION, deliberately, and it is the opposite reasoning from confirm_addition.
        Adding is asked about because it commits hours of machine time that cannot be handed
        back. Removing gives time back and is undone by clicking Count again -- so a dialog here
        would only be in the way. What it does instead is SAY what it removed, by name, because
        a click that changes a list has to be visible somewhere.
        """
        chosen = self.selected_sections()
        waiting = self.queued_in(chosen)

        if not waiting:
            return

        stems = [section["stem"] for section in waiting]

        # The runner first, then the restored list. A stem can only be in one of them, but doing
        # both unconditionally means neither has to be a special case here.
        removed = self.runner.drop(stems)
        dismissed = [stem for stem in stems if stem in self.saved_stems]
        self.saved_stems = [stem for stem in self.saved_stems if stem not in stems]

        self.log("")

        for section in removed:
            self.log(f"took {section['stem']} out of the queue")

        for stem in dismissed:
            self.log(f"{stem} is no longer waiting from last time")

        if removed:
            left = self.runner.remaining()
            self.log(
                f"{left} still waiting -- {app_queue.estimate(left)} after the section being "
                f"counted now" if left else
                "nothing left in the queue; the section being counted now will finish"
            )

            # The status line divides by this, so it has to come down with the queue or the
            # progress reads "4 of 5" for the rest of the batch. See JobRunner.drop.
            self.job_total = self.runner.total
            self.show_status()

        # Saved straight away for the same reason adding is: a queue that is only written down at
        # closing time is lost by a force-quit, and a REMOVAL that is lost comes back as a
        # section counting itself overnight.
        self.save_queue()
        self.draw_list()

        section = self.current_section()

        if section is not None:
            self.update_buttons(section)

    def confirm_addition(self, to_add):
        """Ask before adding several. True if the person said go ahead.

        Same reasoning as confirm_batch -- hours, no cancel -- but the question is different
        enough to be its own: what matters here is how long the queue becomes, not how long the
        batch is, because part of it is already under way.

        ONE section is added without asking, deliberately. That is the whole gesture this
        feature exists for: notice the next section you want, click, walk away.
        """
        waiting = self.runner.remaining() + len(to_add)

        answer = QMessageBox.question(
            self,
            "Add to the queue?",
            f"Add {len(to_add)} sections to the queue, behind {self.runner.stem}?\n\n"
            f"That leaves {waiting} waiting -- {app_queue.estimate(waiting)} after the "
            f"section being counted now finishes.\n\n"
            f"Closing the app asks what to do with the running count and drops whatever is "
            f"still waiting. Anything already saved stays saved.\n\n"
            f"First: {to_add[0]['stem']}\n"
            f"Last:  {to_add[-1]['stem']}",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        return answer == QMessageBox.Yes

    def confirm_batch(self, to_count):
        """Ask before committing hours. True if the person said go ahead.

        The estimate and how to get out of it are both in the question, because those are the
        two facts that would change the answer.
        """
        answer = QMessageBox.question(
            self,
            "Count several sections?",
            f"Count {len(to_count)} sections one after another?\n\n"
            f"This takes {app_queue.estimate(len(to_count))}. To get out of it, close the "
            f"app -- it will ask whether to stop the section being counted, and the ones "
            f"already finished stay saved.\n\n"
            f"First: {to_count[0]['stem']}\n"
            f"Last:  {to_count[-1]['stem']}",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        return answer == QMessageBox.Yes

    def start(self, kind):
        section = self.current_section()

        if section is None:
            return

        # Both napari jobs go the same way. Nothing else reaches this method, but the branch is
        # written as a test rather than an assumption so a future job kind lands on the counting
        # runner by default instead of silently taking the pause with it.
        if kind in ("draw", "review"):
            self.open_napari(kind, section)

            return

        if not self.runner.run(kind, section):
            self.log("a job is already running")

    def count_to_hold(self):
        """The counting process a napari window should stop, as (process, whose section). Or None.

        OURS FIRST, THEN ANYBODY'S. Alice, 2026-08-20: *"why isnt my review cells opening"*. It was
        opening -- show_counts.py was alive and stuck in a disk wait -- while a count started in
        ANOTHER app window ran flat out and the machine sat on 10.1 GB of swap. This method used to
        be one expression, `self.runner.process()`, so a count from elsewhere was invisible to the
        pause and napari opened straight into the collision the pause exists to prevent. The
        machine has one memory; which window pressed the button is bookkeeping.

        THE PID IS RE-READ HERE, not taken from the last poll. find_outside_counts runs every 15
        seconds, and signalling a pid that old means signalling whatever inherited the number if
        that count has since finished. Read now, signalled a moment later. (app_pause.ByPid has the
        rest of the pid-reuse argument.)

        Only ONE is returned even if several counts are somehow running, because Pause holds one
        process -- and two concurrent counts is a state the queue exists to make impossible.
        """
        ours = self.runner.process()

        if ours is not None:
            return ours, self.runner.stem

        # A count found here is news the list should show, so a refresh that changes anything is
        # drawn before the window opens rather than after it closes.
        if self.find_outside_counts():
            self.draw_list()

        for stem, about in self.outside_counts.items():
            return app_pause.ByPid(about["pid"], stem), stem

        return None, None

    def open_napari(self, kind, section):
        """Open the drawing or review window, holding any running count still while it is open.

        ONE FUNCTION FOR BOTH, because the ordering below is the whole correctness argument and it
        is not worth having two copies of. Drawing and reviewing differ only in what they say.

        THE COUNT IS STOPPED BEFORE NAPARI STARTS, not after. Loading a section is the most
        memory-hungry moment either job has, so the order matters: suspend first and napari opens
        into a machine that is no longer being fought over.

        If the window fails to start, the count is let go again immediately. A pause with nothing
        to show for it is the one outcome worth being careful about here -- it would leave the
        count frozen with the buttons all back on and nothing saying why.

        THE COUNT IT STOPS NEED NOT BE THIS WINDOW'S -- see count_to_hold.
        """
        process, self.held_stem = self.count_to_hold()
        held = self.pause.suspend(process)

        if not self.window_runner.run(kind, section):
            if held:
                self.pause.resume()

            self.log("a napari window is already open")

            return

        if held:
            doing = "draw" if kind == "draw" else "look"

            self.log(
                f"\nThe count of {self.held_stem} is PAUSED while you {doing}. It is stopped\n"
                "where it stood and continues from exactly there when you close napari --\n"
                "nothing is lost and nothing is recounted. The elapsed clock leaves the\n"
                "paused time out, so it stays comparable with the usual 35-45 minutes."
            )

            if self.held_stem not in (None, self.runner.stem):
                # Worth saying, because this window shows no clock for a count it did not start and
                # its log has none of that count's output. Without this line the pause would be
                # invisible from here, and an invisible pause is the one that gets left behind.
                self.log(
                    "That count was started outside this window. It is held anyway -- one\n"
                    "machine, one count's worth of memory -- and released when napari closes\n"
                    "or when this window quits."
                )

            # THE SECTION BEING COUNTED IS THE ONE CASE WORTH A WARNING, and it says something
            # different for each job.
            #
            # Drawing: the paused count read its outlines when it started, so redrawing them now
            # cannot change the number it is going to produce. That count will show up marked stale
            # (its file is older than the outlines), which is correct but is much easier to
            # understand if it was expected.
            #
            # Reviewing: this used to need no warning, because reviewing changed nothing. It does
            # now -- edits go into the counted-cells file, and the running count REWRITES that file
            # when it finishes, which would throw the edits away without a word. Told rather than
            # prevented: the section can be looked at perfectly safely, and refusing to open it
            # would be a worse answer than saying which order to do things in.
            # `held_stem`, not `runner.stem`: the running count may be another window's, and a
            # count started elsewhere will overwrite these files just as surely as ours.
            if section["stem"] == self.held_stem:
                if kind == "draw":
                    self.log(
                        "NOTE: this is the section being counted. The count already read its\n"
                        "outlines, so it will finish using the OLD ones and land marked stale --\n"
                        "count it again afterwards to get numbers from what you draw now."
                    )
                else:
                    self.log(
                        "NOTE: this is the section being counted. Look all you like, but do NOT\n"
                        "add or delete cells yet -- the count will overwrite the cell file when\n"
                        "it finishes and your edits would go with it. Wait for it to finish,\n"
                        "then review it."
                    )

        self.log(
            "napari is opening in its own window. CLOSE IT WHEN YOU ARE DONE -- the other\n"
            "buttons here stay disabled until it exits. A napari window left open (or\n"
            "swapped out and invisible) reads as this window being broken; it is only\n"
            "waiting."
        )

        if kind == "review":
            self.log(
                "Start with the 'least confident' layer -- those are the calls just above the\n"
                "threshold, so they are where the model is most likely wrong in either\n"
                "direction.\n"
                "You can FIX what you find: Ctrl-A then click to add a cell it missed, Ctrl-E\n"
                "then click and press Delete to remove one that is not a cell. Yours are drawn\n"
                "filled, the model's are hollow rings. It saves when you close napari (or on\n"
                "Ctrl-S), and nothing is written unless you changed something. This window\n"
                "re-reads the numbers by itself when napari closes -- no Refresh needed."
            )

    def window_started(self, stem, _position, _total):
        """A napari window has opened. Its own handler, because it has no clock.

        Drawing and reviewing are not timed: someone is sitting in front of them, so elapsed time
        answers no question. The count's clock is the one that matters and it must not be
        restarted here, which is exactly what sharing job_started would have done.
        """
        self.open_stem = stem
        self.open_kind = self.window_runner.kind

        self.log(f"\n=== {self.open_kind} {stem} ===")
        self.draw_list()
        self.show_status()

        section = self.current_section()

        if section is not None:
            self.update_buttons(section)

    def window_finished(self, code, message):
        """napari has closed. Let the count go, then re-read the disk.

        RESUMED FIRST, before anything that could fail. refresh() reads every sidecar in the
        folder and the outlines that were just saved, and if any of that raised, a count left
        stopped would be the consequence.
        """
        self.open_stem = None
        self.open_kind = None

        if self.pause.resume():
            self.log(f"the count of {self.held_stem} has continued from where it was paused")

        self.held_stem = None

        if message:
            self.log(message)

        self.log(f"=== finished, exit code {code} ===")
        self.show_status()
        self.refresh()

    def show_status(self):
        """The one line that says the app is working. Called every second while a job runs.

        Facts only, no guesses: what is happening, which section of how many, and how long
        THIS section has been going. The elapsed time is the useful part -- a count prints
        nothing for its first several minutes while it proposes candidates, and without a
        clock that silence is indistinguishable from a hang.

        The clock is per section, not per batch, deliberately. Elapsed time for a whole batch
        answers no question a person has; time on the current section can be compared against
        the typical 35-45 minutes to judge whether something is wrong.

        TIME SPENT PAUSED IS SUBTRACTED, which is the whole reason app_pause.Pause keeps track
        of it. Draw for twenty minutes and an uncorrected clock would read 55m on a count that
        has done 35m of work -- and since the only question the clock answers is "is this
        wedged?", answered by comparing against 35-45 minutes, that is the wrong answer.
        """
        if self.job_start_time is None:
            # A napari window has no clock of its own -- someone is sitting in front of it -- but
            # the line still says what is open, so a blank status never means "a window is open
            # somewhere and this one is dead".
            opened = "Drawing" if self.open_kind == "draw" else "Review"
            line = f"{opened} window open -- {self.open_stem}" if self.open_stem else ""

            # No clock and yet a pause: the held count belongs to another window, so nothing else
            # here would mention it. It has to be said, because whoever left that count running is
            # entitled to know it is stopped -- and because this window is the one that stopped it.
            if line and self.pause.paused():
                line += f"\nthe count of {self.held_stem} is PAUSED until this window closes"

            self.status.setText(line)

            return

        seconds = int(time.monotonic() - self.job_start_time - self.pause.paused_for())
        elapsed = f"{seconds // 60}m {seconds % 60:02d}s"

        what = JOB_MESSAGES.get(self.job_kind, f"Running {self.job_kind}")
        position = f" ({self.job_position} of {self.job_total})" if self.job_total > 1 else ""
        line = f"{what}{position} -- {self.job_stem}   [{elapsed}]"

        if self.pause.paused():
            # A held count and a dead one look identical, so the status says which. The clock
            # above stops advancing on its own, since the paused total grows at the same rate.
            doing = "draw" if self.open_kind == "draw" else "review"
            line += (
                f"\nPAUSED while you {doing} {self.open_stem} -- "
                "it continues when napari closes"
            )

        self.status.setText(line)

    def job_started(self, stem, position, total):
        """A section has started. Restart the clock for it.

        Driven by the runner's signal rather than set in start_count, so every section of a
        batch gets its own clock without the window having to know when one ends and the next
        begins -- it is told.
        """
        self.job_kind = self.runner.kind
        self.job_stem = stem
        self.job_position = position
        self.job_total = total
        self.job_start_time = time.monotonic()

        # The paused total belongs to ONE section's elapsed time. Carried into the next section
        # of a batch it would subtract a pause that section never had, and read short.
        self.pause.forget()

        self.job_timer.start()
        self.show_status()

        # The queue, written down where a quit cannot take it. HERE rather than in start_count
        # because this signal is the only thing that knows a section has handed over to the next
        # one -- so what gets stored is always "the one running plus what is behind it", and a
        # section that has already finished is never offered again.
        self.save_queue()

        prefix = f"{position}/{total} " if total > 1 else ""
        self.log(f"\n=== {prefix}{self.job_kind} {stem} ===")

        # Said ONCE per batch, at the top, and only for counting. Alice, 2026-08-20: *"i added
        # images to queue to count overnight but they didn't count"* -- the Mac had slept all
        # night and then restarted. app_awake keeps it from idle-sleeping, but nothing can stop
        # the lid, so the one part she has to do is worth a line on screen. Repeating it every
        # section of an eight-section batch would train her to ignore it.
        if self.job_kind == "count" and position <= 1:
            for line in app_awake.describe(app_awake.on_battery()):
                self.log(line)

        # Redraw now, not at the end of this section: this is what disables the buttons and
        # marks the running and waiting sections in the list. Here rather than in start_count
        # so a batch's second and later sections get marked too -- they are announced by the
        # same signal, and nothing else knows when one section hands over to the next.
        #
        # draw_list, not refresh: nothing has been written yet, so there is nothing to re-read,
        # and refresh would freeze the window for a second or two at the start of every section.
        self.draw_list()

        section = self.current_section()

        if section is not None:
            self.update_buttons(section)

    def log(self, line):
        self.output.appendPlainText(line)

    def job_finished(self, code, message):
        """One section is done. There may be more coming.

        The clock is only stopped when the runner is idle. Mid-batch, `busy()` is still True
        and the next section's `started` signal is about to restart the clock anyway -- so
        clearing the status here would blank the label for an instant and then refill it,
        which looks like a glitch rather than progress.
        """
        if not self.runner.busy():
            self.job_timer.stop()
            self.job_start_time = None
            self.show_status()

        if message:
            self.log(message)

        self.log(f"=== finished, exit code {code} ===")

        # Re-read the disk rather than patching the table: the job wrote files, and the files
        # are the state. This also picks up the area cache that draw_section invalidated.
        # Done after every section of a batch, so numbers appear as they land rather than all
        # at the end of a five-hour queue.
        self.refresh()

    def ask_about_running_count(self):
        """Closing during a count: ask what should happen to it. Returns leave/stop/cancel.

        A METHOD OF ITS OWN so the decision can be tested without a person clicking. The two
        answers do very different things to forty minutes of work, so both paths need checking,
        and a modal dialog cannot be checked by a script -- it waits for a human. A test
        replaces this method and closeEvent below is exercised for real.

        WHY ASK AT ALL, rather than picking one. Both answers are reasonable and only she knows
        which: closing the window because the desk is needed is not the same intention as
        closing it because the count was started on the wrong section. Guessing gets one of
        those wrong every time, and the wrong guess costs either 40 minutes of computation or
        40 minutes of waiting for a result that was never wanted.
        """
        waiting = self.runner.remaining()

        box = QMessageBox(self)
        box.setWindowTitle("A count is still running")
        box.setText(f"{self.runner.stem} is still being counted.")
        box.setInformativeText(
            "Leave it running -- it carries on without this window and saves its numbers "
            "when it finishes. They will be waiting next time you open the app.\n\n"
            "Stop the count -- it ends now and this section stays uncounted. Nothing that "
            "has already been saved is affected."
            + (
                f"\n\nEither way, the {waiting} section{'s' if waiting > 1 else ''} still "
                "queued behind it will not be counted."
                if waiting
                else ""
            )
        )

        leave = box.addButton("Leave it running", QMessageBox.AcceptRole)
        stop = box.addButton("Stop the count", QMessageBox.DestructiveRole)
        box.addButton("Don't quit", QMessageBox.RejectRole)

        # Leaving it running is the default because it is the recoverable answer: a count left
        # running can still be stopped from Activity Monitor, while a stopped one cannot be
        # un-stopped. When an accidental keypress picks the answer, it should pick that one.
        box.setDefaultButton(leave)
        box.exec()

        clicked = box.clickedButton()

        if clicked is leave:
            return "leave"

        if clicked is stop:
            return "stop"

        return "cancel"

    def closeEvent(self, event):
        """Decide what happens to a running count, then let the window go.

        THE PAUSE IS THE ONE FAILURE WORTH GUARDING AGAINST UNCONDITIONALLY. A suspended
        process orphaned by a quit sits frozen forever holding its 1.5-2 GB and shows no sign
        of it -- in Activity Monitor a stopped process looks exactly like an idle one, at 0%
        CPU. So nothing here may leave a pause behind.

        ORDER MATTERS TWICE.

        Resume before terminating. SIGTERM is delivered to a process, but a stopped process is
        not running, so it cannot act on it until it starts again; terminating first and never
        resuming leaves it stopped with a pending signal, which is the frozen orphan above by
        another route.

        Resume AFTER the question, not before. If she answers "don't quit" the drawing window
        is still open, so the count must stay stopped -- resuming it there would undo the pause
        that drawing depends on while the app carries on running.

        WHAT QUITTING USED TO COST is worth writing down because it was unpredictable, which is
        the worst kind of answer. The count is a subprocess with its output piped here, so
        closing the window broke the pipe; whether that killed the count depended on whether it
        happened to write anything afterwards. Alice closed the app during Gre595 on 2026-08-20
        and lost the run. count_cells.ignore_a_lost_listener settled that: an unwatched count
        keeps going. This dialog is what makes it a choice instead of a side effect.

        THE PAUSED COUNT MAY BELONG TO ANOTHER WINDOW (count_to_hold), which makes the
        unconditional resume below matter more, not less: quitting must never leave somebody else's
        count frozen, and nothing outside this process would ever know to release it. The question
        above is still only asked about OUR count -- stopping a count this window did not start is
        not this window's decision.
        """
        if self.runner.busy() and self.runner.process() is not None:
            answer = self.ask_about_running_count()

            if answer == "cancel":
                event.ignore()

                return

            if self.pause.resume():
                self.log(f"released the paused count of {self.held_stem}")

            if answer == "stop":
                # Written down BEFORE the stop, because stop() empties the queue and the stem of
                # the section being killed is about to stop being the running one. Stopping means
                # "not now", not "never" -- none of these sections has been counted, so all of
                # them are worth offering back next time. self.quitting keeps queue_finished from
                # deleting the note when the terminated count reports in.
                waiting = ([self.runner.stem] if self.runner.stem else []) + [
                    section["stem"] for section in self.runner.queue
                ]
                self.quitting = True

                stopped, dropped = self.runner.stop()

                if stopped:
                    self.log(f"stopped the count of {self.runner.stem}")

                if dropped:
                    self.log(f"and dropped {dropped} queued section(s)")

                if waiting:
                    app_saved_queue.remember(self.root, waiting)
                    self.log(f"{len(waiting)} section(s) kept for next time")
        elif self.pause.resume():
            # The usual case here is now a count from ANOTHER window: ours would have been caught
            # above. Named, because that count outlives this app and its owner needs to know it was
            # touched at all.
            self.log(f"released the paused count of {self.held_stem} before quitting")

        super().closeEvent(event)

    def queue_finished(self, succeeded, failed):
        """The whole batch is over. Only worth saying anything if there was more than one."""
        # Nothing is waiting any more, so there is nothing to offer next time. Before the early
        # return below, because a batch of one still has a queue to clear.
        #
        # UNLESS THE APP IS QUITTING. Stopping a count on the way out empties the queue and this
        # signal follows, which would delete the note closeEvent just wrote -- the sections were
        # abandoned, not finished.
        if not self.quitting:
            app_saved_queue.forget()

        if succeeded + failed <= 1:
            return

        self.log(
            f"\n=== batch over: {succeeded} counted"
            + (f", {failed} FAILED -- see above for why" if failed else "")
            + " ==="
        )


def main():
    application = QApplication(sys.argv)

    window = CellCounter()
    window.show()

    ask_before_dying(window)

    sys.exit(application.exec())


def ask_before_dying(window):
    """Make Ctrl-C in the terminal close the window instead of killing the app outright.

    WHY THIS EXISTS. Alice, 2026-08-20: *"if i close the app while its counting i thought you
    said there would be a pop up that asked if user wanted it to run in the background?"*. The
    popup was there and it worked -- but it lives in CellCounter.closeEvent, and Qt only calls
    closeEvent when the WINDOW is closed. The app is started with `python app.py` from a
    terminal, and the obvious way to get that terminal back is Ctrl-C, which sends SIGINT and
    kills Python where it stands. No closeEvent, no question, and the count is orphaned by
    accident instead of on purpose.

    That is not a guess about what happened. At 01:38 the process list held PID 11062 counting
    Gre595_RH_3-1-4 with ppid 1 -- parent gone, count still running, nobody asked.

    THE TIMER IS NOT OPTIONAL. Python does not run a signal handler the instant the signal
    arrives; it sets a flag and runs the handler between bytecodes. An app sitting inside
    QApplication.exec() is inside C++ and executes no bytecodes at all, so without something
    that periodically returns to Python the handler waits for the next click -- which looks
    exactly like Ctrl-C being ignored. A do-nothing timer four times a second is the standard
    fix and costs nothing measurable.

    SIGTERM is handled the same way, so `kill` and a logout both get the question too. SIGKILL
    cannot be caught by anything, by design; that one still loses the dialog, and there is no
    version of this file that changes it.
    """
    # A list rather than a bool because a closure cannot rebind a name in its enclosing scope
    # without `nonlocal`, and this is clearer than reaching for it.
    asked = []

    def close_the_window():
        window.close()
        asked.clear()

    def wants_to_quit(signal_number, frame):
        # Two quick Ctrl-Cs would otherwise stack two dialogs on top of each other. Cleared by
        # close_the_window, so answering "Don't quit" leaves Ctrl-C working for next time.
        if asked:
            return

        asked.append(signal_number)

        # DEFERRED, not called here. This handler runs between two bytecodes of whatever the
        # main thread happened to be doing; opening a modal dialog from inside it would start a
        # nested event loop in the middle of an unrelated function. Handing it to the event loop
        # means the dialog opens from the same place every other dialog in this app opens from.
        QTimer.singleShot(0, close_the_window)

    signal.signal(signal.SIGINT, wants_to_quit)
    signal.signal(signal.SIGTERM, wants_to_quit)

    # Parented to the window, which is what keeps it alive after this function returns.
    ticker = QTimer(window)
    ticker.setInterval(250)
    ticker.timeout.connect(lambda: None)
    ticker.start()


if __name__ == "__main__":
    main()
