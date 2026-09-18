"""A second window showing every number in the folder at once, ready to copy or save.

WHY A SEPARATE WINDOW. The main window answers "what should I do next with this section"; one
section at a time is exactly right for that. This answers a different question -- "what did the
whole set come out at" -- and the two do not fit in one view without one of them being cramped.
Being a separate window also means it can be left open on a second screen while counting runs.

WHY THE COPY AND SAVE BUTTONS LIVE HERE. They used to sit in the main window, where they
exported the whole folder while the window showed one section -- so there was no way to see
what was about to be copied. Here, WHAT YOU COPY IS WHAT YOU ARE LOOKING AT: the buttons act on
the visible tab. That is the whole point of the window.

TWO TABS, TWO LEVELS. Every section is the raw record, one row per section per region. Average
per bird is one row per bird per region, which is the level most questions are actually asked at
-- a bird is the experimental unit and a section is a sample of it. app_export.by_bird holds the
two rules that view has to obey: means of densities rather than sums of counts, and stale
sections excluded from the averages but still visible in the raw tab.

NO SORTING BY CLICKING THE HEADERS, deliberately. Copying takes the rows in their canonical
order (bird, then section, then dNCM/vNCM/CMM); if the view could be re-sorted, the clipboard
would silently disagree with the screen. The rows are already in reading order.

Usage:
  python app/app_all_data.py           # open the window on its own, against the current folder
"""

import sys
from pathlib import Path

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# THE APP'S FOLDERS, BEFORE THE FIRST IMPORT THAT NEEDS THEM. This file can be run as its own
# process, and Python then puts only ITS folder on sys.path -- so `import ps6` at the root, or a
# sibling folder's module, would not be found. app_path.py explains the whole arrangement; the line
# before it is there because app_path is at the root, which is not on the path yet either. The
# condition also covers a flat copy of the app, where app_path sits right here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "app_path.py").exists() else _HERE.parent))

import app_path

app_path.setup()

import app_export

# Same red the main window uses for stale numbers, so the two windows mean the same thing by it.
STALE_COLOUR = "#b3261e"

# Columns whose values are numbers and should sit against the right edge, where digits line up
# and a long column can be read down. Named rather than detected from the dtype: "sections" is a
# count and belongs right, and a future text column must not have to opt out.
NUMBER_COLUMNS = {
    "cells",
    "area mm2",
    "cells per mm2",
    "sections",
    "mean cells per mm2",
    "sd",
    "mean area mm2",
}


class AllData(QWidget):
    """Every counted region in the folder, at two levels, with the two exports.

    Holds no state of its own beyond the two frames it was given. show_table() replaces both,
    so the main window can hand it fresh numbers after a count without this window having to
    know when a job finished or how to read a sidecar.
    """

    def __init__(self, parent=None):
        # No parent passed to QWidget on purpose: a parented QWidget is drawn INSIDE its parent
        # rather than as its own window. The reference is kept by the main window instead, which
        # is what stops this being garbage collected the moment it is shown.
        super().__init__()

        self.setWindowTitle("All data")
        self.resize(920, 620)

        self.rows = pd.DataFrame()
        self.birds = pd.DataFrame()

        # The frame the other two are derived from. Kept because describe_by_bird needs it to
        # say how many stale rows were left out, which cannot be seen from the averages alone.
        self.raw = pd.DataFrame()
        self.folder = ""

        self.build()

    # ---- layout ----

    def build(self):
        self.heading = QLabel("")
        self.heading.setFont(QFont("", 13, QFont.Bold))

        self.note = QLabel("")
        self.note.setWordWrap(True)

        self.rows_table = self.new_table()
        self.birds_table = self.new_table()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.rows_table, "Every section")
        self.tabs.addTab(self.birds_table, "Average per bird")

        # Which tab is showing changes what the buttons will copy, so the note under the
        # heading has to change with it.
        self.tabs.currentChanged.connect(self.show_note)

        self.copy_button = QPushButton("Copy for Sheets")
        self.copy_button.setToolTip(
            "Copy the table you are looking at, ready to paste into Google Sheets:\n"
            "click a cell there and press cmd-V, and it fills in the columns by itself."
        )
        self.copy_button.clicked.connect(self.copy_for_sheets)

        self.csv_button = QPushButton("Save CSV")
        self.csv_button.setToolTip(
            "Save the table you are looking at as a dated .csv file. Opens in Excel, or\n"
            "in Google Sheets with File > Import. A file is the version worth keeping."
        )
        self.csv_button.clicked.connect(self.save_csv)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.csv_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self.note)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(buttons)

    def new_table(self):
        """An empty read-only table. Both tabs are built the same way."""
        table = QTableWidget(0, 0)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        # Whole rows, so a copy of a section is a copy of everything about it. The tables are
        # read-only, so selection is only ever for reading and for the system's own copy.
        table.setSelectionBehavior(QTableWidget.SelectRows)

        return table

    # ---- filling it in ----

    def show_table(self, table, folder):
        """Rebuild both tabs from an app_table.build_table frame."""
        self.folder = str(folder)
        self.rows = app_export.for_export(table)
        self.birds = app_export.by_bird(table)
        self.raw = table

        self.heading.setText(f"All data -- {Path(self.folder).name or self.folder}")

        self.fill(self.rows_table, self.rows)
        self.fill(self.birds_table, self.birds)

        self.tabs.setTabText(0, f"Every section ({len(self.rows)})")
        self.tabs.setTabText(1, f"Average per bird ({len(self.birds)})")

        self.show_note()

    def fill(self, widget, frame):
        """Put a frame in a table widget. Missing numbers stay blank, stale rows go red."""
        widget.clear()
        widget.setColumnCount(len(frame.columns))
        widget.setRowCount(len(frame))
        widget.setHorizontalHeaderLabels(list(frame.columns))

        stale_column = "stale" if "stale" in frame.columns else None

        for row, (_, record) in enumerate(frame.iterrows()):
            stale = bool(stale_column and record[stale_column])

            for column, name in enumerate(frame.columns):
                value = record[name]

                # pd.isna, not `is None`: a column of numbers stores a missing value as NaN, so
                # `is None` never matches and every empty cell would print as "nan". This exact
                # bug has already been fixed twice in this project.
                item = QTableWidgetItem("" if pd.isna(value) else str(value))

                if name in NUMBER_COLUMNS:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                if stale:
                    item.setForeground(QColor(STALE_COLOUR))

                widget.setItem(row, column, item)

        header = widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # The section name is the one column worth giving the leftover width to: the stems are
        # long, and a truncated one cannot be told apart from its neighbour.
        if "section" in list(frame.columns):
            header.setSectionResizeMode(
                list(frame.columns).index("section"), QHeaderView.Stretch
            )

    def show_note(self):
        """The sentence under the heading, describing the tab that is showing."""
        if not len(self.rows) and not len(self.birds):
            self.note.setText(
                "Nothing counted in this folder yet. Count a section and come back."
            )

            return

        lines = (
            app_export.describe(self.rows)
            if self.showing_rows()
            else app_export.describe_by_bird(self.raw, self.birds)
        )

        # Worst news first is how both describes are ordered, so joining keeps that order. The
        # leading spaces mark a warning in the log pane and are noise in a label.
        self.note.setText("  ".join(line.strip() for line in lines))

    # ---- the exports ----

    def showing_rows(self):
        return self.tabs.currentIndex() == 0

    def visible_frame(self):
        """Whatever the visible tab is showing. The buttons must not export the other one."""
        return self.rows if self.showing_rows() else self.birds

    def visible_name(self):
        """A filename that says which of the two views it holds."""
        stem = app_export.file_name().removesuffix(".csv")

        return f"{stem}{'' if self.showing_rows() else ' by bird'}.csv"

    def copy_for_sheets(self):
        frame = self.visible_frame()

        if not len(frame):
            QMessageBox.information(self, "Nothing to copy", self.note.text())

            return

        QApplication.clipboard().setText(app_export.as_tsv(frame))

        # A dialog rather than a status line: this window has no log to write to, and a copy
        # that says nothing is indistinguishable from a copy that failed.
        QMessageBox.information(
            self,
            "Copied",
            f"{len(frame)} rows copied.\n\nIn Google Sheets, click the cell where the table "
            "should start and press cmd-V -- the columns fill in by themselves.",
        )

    def save_csv(self):
        frame = self.visible_frame()

        if not len(frame):
            QMessageBox.information(self, "Nothing to save", self.note.text())

            return

        suggested = str(Path(self.folder).resolve() / self.visible_name())

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save this table as a CSV file", suggested, "CSV files (*.csv)"
        )

        if not chosen:
            return

        try:
            Path(chosen).write_text(app_export.as_csv(frame), encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save", f"{chosen}\n\n{error}")

            return

        QMessageBox.information(
            self,
            "Saved",
            f"{len(frame)} rows written to\n{chosen}\n\nTo put it in Google Sheets: "
            "File > Import > Upload, and choose that file.",
        )


def main():
    """Open the window on its own, against the folder given or the current one.

    Worth having for the same reason every other module here has a __main__: the window can be
    looked at without launching the app, running a count, and clicking through to it.
    """
    from app_state import find_sections
    from app_table import build_table

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    application = QApplication([])

    window = AllData()
    window.show_table(build_table(find_sections(root)), root)
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
