#!/bin/bash
# Start the cell counting app on macOS or Linux.
#
# WHAT THIS IS FOR. The app is Python, and Python programs do not start by being double-clicked:
# they need an interpreter with thirteen packages installed, and that is a paragraph of terminal
# commands nobody should have to follow to look at a bird brain. So this file does it: finds a
# Python, builds the app its own environment the first time it runs, and starts the window. After
# the first run it takes about a second.
#
# ON WINDOWS use "Count cells.bat" instead. On macOS you can double-click "Count cells.command",
# which just runs this file -- see INSTALL.md, including what to do about the warning macOS shows
# the first time you open something you downloaded.
#
# IT IS SAFE TO RUN AGAIN AT ANY TIME. Every step below asks whether it is already done, so a run
# that was interrupted halfway through the download -- which is a gigabyte, so it happens -- is
# fixed by starting it again rather than by starting over. What was already fetched is not fetched
# twice: pip and uv both keep a cache in the home folder.

set -u

# WHERE THIS FILE IS, not where the terminal happens to be. Double-clicking on a Mac starts in the
# home folder, so without this the app would look for its own code in the wrong place. THE QUOTES
# MATTER: the folder is called "cell counting app", with a space in it, and an unquoted path would
# be read as two arguments -- which is the single most common way a launcher script fails.
cd "$(dirname "$0")" || exit 1

VENV=".venv"
PYTHON="$VENV/bin/python"

# WHICH INSTALLER DOES THE WORK. Empty until the setup below decides: "" means pip, "module" means a
# uv that was just fetched into the environment, and anything else is the path to a uv already on
# this computer. Declared here because `set -u` makes reading an unset variable an error.
UV=""

# CHECKED WITHOUT IMPORTING THEM. importlib.util.find_spec asks "is this installed" by looking on
# disk; actually importing torch and napari takes eight seconds and would be paid on every start.
NEEDED="napari torch torchvision qtpy psutil tifffile pandas scipy skimage PIL"

say() {
    printf '%s\n' "$*"
}

# STOPS WITH THE WINDOW STILL OPEN. A double-clicked script that fails and closes leaves nothing to
# read but a vanished window, and "it didn't work" is not a report anyone can act on.
stop() {
    say ""
    say "$*"
    say ""
    printf 'Press Return to close this window. '
    read -r _ || true
    exit 1
}

# 3.11, 3.12 or 3.13. NOT "the newest Python": torch has no wheels for a new Python version for
# some months after it appears, so the newest is usually the one where `pip install` fails with a
# wall of red. Written and run on 3.12.4.
usable() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)' \
        >/dev/null 2>&1
}

find_python() {
    for candidate in python3.12 python3.13 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && usable "$candidate"; then
            command -v "$candidate"

            return 0
        fi
    done

    return 1
}

ready() {
    # `python -c pass` first, because "the environment exists" is not the same question as "the
    # environment works". A .venv built by a Python that has since been upgraded or removed is left
    # behind as a folder full of dead symlinks, which is what a half-finished setup looks like.
    "$PYTHON" -c 'raise SystemExit(0)' >/dev/null 2>&1 || return 1

    # shellcheck disable=SC2086
    "$PYTHON" -c 'import importlib.util, sys
raise SystemExit(0 if all(importlib.util.find_spec(n) for n in sys.argv[1:]) else 1)' \
        $NEEDED >/dev/null 2>&1
}

# THE ONE PLACE THAT KNOWS HOW TO INSTALL, because it is called twice: once for torch from the CPU
# index on Linux, and once for requirements.txt.
#
# PREBUILT PACKAGES FIRST, SOURCE ONLY IF IT MUST. --only-binary says "download something already
# built, or fail" -- and failing in ten seconds is the kind thing to do, because a package that has
# to be compiled here can spend twenty minutes at it and then stop on a missing C compiler, which is
# not a thing to ask of somebody who wanted to count cells. It is an attempt, though, not a rule: if
# some dependency on some system genuinely ships only source, the second try lets it through rather
# than refusing to install an app that would have worked.
install_packages() {
    case "$UV" in
        "")     "$PYTHON" -m pip install --only-binary :all: "$@" && return 0 ;;
        module) "$PYTHON" -m uv pip install --python "$PYTHON" --only-binary :all: "$@" && return 0 ;;
        *)      "$UV" pip install --python "$PYTHON" --only-binary :all: "$@" && return 0 ;;
    esac

    say ""
    say "  something here has no prebuilt version -- trying again the slow way, which may take a while"

    case "$UV" in
        "")     "$PYTHON" -m pip install "$@" ;;
        module) "$PYTHON" -m uv pip install --python "$PYTHON" "$@" ;;
        *)      "$UV" pip install --python "$PYTHON" "$@" ;;
    esac
}

# uv IF IT CAN BE HAD, pip OTHERWISE. Both install the same pinned versions from requirements.txt;
# uv is that job with the waiting taken out. It fetches the packages in parallel instead of one after
# another, resolves the versions in seconds rather than minutes, and hard-links them out of a cache in
# the home folder instead of unpacking and byte-compiling every file. Measured on this machine with
# an empty cache, the same 1 GB over the same connection: uv 89 seconds against pip's 174.
#
# FETCHED WITH pip WHEN IT IS ABSENT -- one small wheel and a few seconds, paid back several times
# over. Deliberately NOT `curl ... | sh`, which is how uv's own site installs it: that asks somebody
# to run a script off the internet on the say-so of a program they were emailed, and the wheel on
# PyPI is the same tool arriving through a channel they already use.
#
# AND IF ANY OF THAT FAILS, pip does the work. uv is a speed-up, never a requirement: a pip too old
# to install it, a machine with no route to it, an unfamiliar platform -- all end in the same place,
# a working app, just later.
choose_installer() {
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
        say "  using uv, which is already installed here"

        return 0
    fi

    "$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1

    if "$PYTHON" -m pip install --quiet uv >/dev/null 2>&1; then
        UV="module"
        say "  fetched uv, a faster installer, to do it with"
    else
        say "  using pip (uv was not available here -- this works, it is only slower)"
    fi
}

if ! "$PYTHON" -c 'raise SystemExit(0)' >/dev/null 2>&1; then
    found="$(find_python)" || stop "\
This computer has no Python 3.11, 3.12 or 3.13, which the app needs.

  macOS:  install it from https://www.python.org/downloads/ (any 3.12.x), then run this again.
  Linux:  sudo apt install python3.12 python3.12-venv     (or your system's equivalent)

Nothing else has to be installed by hand -- this script does the rest."

    say "Setting up the app's own Python environment. This happens once."
    say "  using $found"

    # NOT --clear, which would empty the folder first. If .venv is here and broken it might still be
    # somebody's, and a launcher is not the right thing to be deleting folders. Building over it
    # replaces what is missing; if that is not enough, the message below says what to do.
    "$found" -m venv "$VENV" || stop "\
Could not create the environment in $VENV.

On Debian and Ubuntu this usually means the venv module is packaged separately:
  sudo apt install python3-venv

If $VENV already exists and is broken, delete that folder and run this again."
fi

if ! ready; then
    say ""
    say "Installing what the app needs: about 1 GB, mostly the detector's maths library."
    say "THIS HAPPENS ONCE. A few minutes on a fast connection, longer on a slow one -- most of it"
    say "is the download, so it goes at the speed of your internet."
    say ""

    choose_installer

    # ON LINUX, TORCH FIRST, FROM THE CPU INDEX. The default torch wheel for Linux is the NVIDIA
    # one: 2.5 GB, and useless without that card. Counting is memory-bound rather than
    # compute-bound here, so there is nothing to gain from it even on a machine that has one.
    # Installed before requirements.txt so the pinned version is already satisfied when the
    # installer reads that file, and it does not fetch the big one afterwards.
    if [ "$(uname -s)" = "Linux" ]; then
        say "  Linux: fetching the CPU build of torch (the default one is a 2.5 GB NVIDIA build)"
        install_packages --index-url https://download.pytorch.org/whl/cpu \
            torch==2.11.0 torchvision==0.26.0 || stop "\
Could not install torch. If this says something about disk space, the download needs about 1 GB
free; if it mentions a version, this computer's Python may be too new -- see the top of this file."
    fi

    install_packages -r requirements.txt || stop "\
Could not install the packages the app needs. The message above says which one and why; the usual
causes are no internet connection, or a Python version outside 3.11-3.13.

Nothing is broken -- run this script again when it is sorted out."
fi

if ! ready; then
    stop "\
The environment was built but is still missing something the app needs. Delete the $VENV folder
in this directory and run this script again; that rebuilds it from scratch."
fi

say ""
say "Starting the app. Its messages appear in this window."
say "Closing this window quits the app -- but a count already running carries on to the end,"
say "and its numbers are saved when it finishes."
say ""

# exec, so the app REPLACES this script rather than being its child: one process instead of two, and
# Ctrl-C in this window reaches the app itself, which is a normal way to quit it.
exec "$PYTHON" app.py
