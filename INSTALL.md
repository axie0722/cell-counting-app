# Cell counting — how to install and run it

This counts PS6-labelled cells in NCM and CMM on zebra finch brain sections, and reports a
density for each region. You outline the two regions on a section (the app proposes the outlines,
you correct them), and it does the counting.

**Read the two short warnings at the bottom before you use the numbers for anything.** One is
about the licence on the detector, the other is about what the counts are and are not.

---

## 1. What you need

* **A Mac, a Windows PC or a Linux machine.** Any of the three.
* **Python 3.11, 3.12 or 3.13.** Not the very newest version — see below.
* **About 3 GB of free disk space**, and an internet connection for the first run.
* **8 GB of memory or more.** A single section is a 40-megapixel image and counting one holds
  about 2 GB.

You do **not** need to install anything except Python. The launcher does the rest.

### Getting Python

* **macOS** — installers at <https://www.python.org/downloads/>. Take any 3.12.x.
* **Windows** — same page, and **tick "Add python.exe to PATH" on the first screen** of the
  installer. If you miss it, the launcher cannot find Python.
* **Linux** — `sudo apt install python3.12 python3.12-venv` (or your distribution's equivalent).
  The `-venv` part is separate on Debian and Ubuntu and the app needs it.

**Not the newest Python.** PyTorch, which the detector uses, has no downloads ready for a brand
new Python version for a few months after it appears, so "the latest" is usually the version where
the install fails. 3.12 is the safe answer.

---

## 2. Install and run

Unzip the folder somewhere you can find again — Documents is fine, the Desktop is fine. Then:

### macOS

Double-click **`Count cells.command`**.

The first time, macOS will probably refuse: *"cannot be opened because it is from an unidentified
developer"*. That is not about this app being faulty — it is about it not being signed with a paid
Apple developer account. **Right-click the file and choose Open** instead of double-clicking, and
the same box appears with an Open button on it. You only have to do this once.

### Windows

Double-click **`Count cells.bat`**.

Windows may show a blue "Windows protected your PC" box for the same reason as above. Click **More
info**, then **Run anyway**.

### Linux

Open a terminal in the folder and run:

```
bash count-cells.sh
```

### The first run takes 10–20 minutes

It builds the app its own private Python environment inside the folder (a `.venv` directory) and
downloads about 1 GB, most of which is PyTorch. Leave it alone until the window appears. **It only
happens once** — after that, starting the app takes about a second.

If the download is interrupted, just start the launcher again. Every step checks whether it has
already been done, so it picks up where it stopped rather than starting over.

If something goes wrong, the window stays open with the reason in it. That message is the useful
thing to send on.

---

## 3. Using it

**Press "Choose folder" first.** The app does not come with any images and does not assume they
are anywhere in particular. Point it at the folder holding your sections and it will search that
folder and everything inside it. It remembers the choice, so this is a first-run step only.

Your images stay where they are. Everything the app produces is written **next to each image** —
`<name> regions.csv`, `<name> counted cells.csv`, and so on — so a folder of sections carries its
own outlines and counts, and moving that folder moves the work with it.

### Two things your file names have to say

* **`_LH_` or `_RH_`** somewhere in the name, for the hemisphere: `Gre595_RH_3-1-4_NCM_Slide 1.TIF`.
  A hyphen works too (`Gre595_LH-1-1-10`). Without one of those the app cannot tell which
  hemisphere a section is and will say so rather than guess, because guessing wrong mirrors the
  brain and nothing downstream would complain.
* Sections must be **TIFs**.

**Scale.** The app reads microns-per-pixel out of the TIF's own metadata. If your scanner does not
write it, the app falls back to 0.3088 µm/px, which is this project's microscope. Densities from a
different microscope would then be wrong by whatever the ratio is, so check that first.

### The order of work, per section

1. **Draw outlines.** A napari window opens with NCM and CMM **already proposed** — drag the
   vertices that are wrong, or press Ctrl-R and draw it yourself. `Ctrl-S` saves; `Ctrl-F` flips
   which side of Field L is called NCM if the two are swapped (obvious on sight — cyan on the
   wrong region).
2. **Mark dorsal.** Press `4` and click once on the hippocampus side. This is still required even
   when the outlines look perfect: it is what splits NCM into dNCM and vNCM, and no outline on
   screen reveals it.
3. **Count cells.** 35–45 minutes per section. Select several sections and the button offers to
   count all of them, one after another — you can add more to the queue while it runs.
4. **Review cells.** Opens the counted cells on the image. You can add and delete cells here, and
   the app's numbers follow.
5. **View all data** gives you every counted region in the folder in one table, plus a per-bird
   average, with buttons to copy or save it for a spreadsheet.

### Leave it plugged in, and leave the lid open

A sleeping computer does not count — the job freezes and resumes when the machine wakes, so an
overnight run can produce twenty minutes of work. The app holds the machine awake for as long as a
count is running, on all three systems. **What no program can prevent is lid-close sleep.** If you
are leaving a batch overnight: plug it in, leave the lid open.

### One count at a time, on purpose

Two counts at once on an 8 GB machine finish **later** than the same two run back to back, because
they push each other's memory to disk. So the app queues them instead. For the same reason,
opening a drawing or review window during a count **pauses** the count where it stands and resumes
it from exactly there when you close the window; nothing is lost and nothing is repeated.

---

## 4. The licence on the detector — please read

`ps6_cnn_backbone.pt` is the trained detector, and it is built on **ImageNet pretrained weights,
which are licensed for NON-COMMERCIAL RESEARCH USE ONLY.**

That restriction travels with the file. It applies to this copy, to any copy you make, and to
anything you use it for. **Research and teaching: fine. Anything commercial: not permitted by that
licence.** If you need a commercially usable version, the backbone has to be retrained from
scratch on data that allows it.

## 5. What the numbers are

**It under-counts, and it knows by how much.** Accuracy was measured on windows where every cell
was clicked by hand on a blank screen, with no model output visible — so, unusually, "no click"
really means "no cell" and a miss is measurable. Against that:

| | it finds | of what it reports, this much is a cell |
| --- | --- | --- |
| tissue from a hemisphere it was trained on | about **66%** of the cells | about **86%** |
| a hemisphere it has never seen | about **57%** of the cells | about **90%** |

Three things follow, and they matter more than the exact numbers:

* **It misses cells; it does not invent them.** On unfamiliar tissue it gets *more* cautious, not
  wilder. That is the good failure mode for this kind of work.
* **Compare densities within a study, never quote a raw count as the truth.** A consistent
  one-third under-count cancels out of a comparison between two groups counted the same way. It
  does not cancel out of an absolute number, and it does not cancel out of a comparison against
  somebody's hand count.
* **A third of the missing cells are unreachable.** About 9% of real cells are never even offered
  to the classifier, so no setting anywhere in the app can find them.

And the human check is not optional: **look at your sections in Review cells** before using the
numbers. It takes a couple of minutes each and it is the only way to catch the things that go
wrong per-section rather than on average — a low-contrast slide, an outline in the wrong place, a
fold in the tissue. Cells near the edge of the tissue are the least reliable of all.

One more caveat that is easy to miss: these figures come from five birds in one lab, on one
microscope, with one staining protocol. On tissue that differs in any of those, the honest answer
is that nobody has measured it. Count a few sections by hand and compare before you trust a batch.

## 6. What is not in this package

No images, no counts, no outlines, no research data. This is the program and the trained model.
