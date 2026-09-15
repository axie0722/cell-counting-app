@echo off
rem Start the cell counting app on Windows. Double-click this file.
rem
rem The same job as count-cells.sh, which is the macOS and Linux version: find a Python, build the
rem app its own environment the first time, then start the window. After the first run it takes
rem about a second. It is safe to run again at any time -- every step checks whether it is already
rem done, so a setup interrupted halfway through the download is fixed by running it again.
rem
rem WHY THIS IS A SEPARATE FILE and not the same script: cmd.exe cannot read a shell script and bash
rem is not installed on Windows. There is no one file that starts a program on all three systems, so
rem all three go in the package and each person double-clicks the one for their computer.

rem WHERE THIS FILE IS, not where the command prompt was. /d because the folder may be on another
rem drive than the one the prompt started on, and plain `cd` will not change drives. THE QUOTES
rem MATTER: the folder is called "cell counting app", with a space in it.
cd /d "%~dp0"

set "VENV=.venv"
set "PYTHON=%VENV%\Scripts\python.exe"

rem Checked without importing them: find_spec looks on disk, where importing torch and napari would
rem cost eight seconds on every start.
set "NEEDED=napari torch torchvision qtpy psutil tifffile pandas scipy skimage PIL"

rem --- is there a working environment already? ---------------------------------------------------
rem "Exists" is not the same question as "works": an environment built by a Python that has since
rem been uninstalled or upgraded is left behind as a folder that cannot run anything.
"%PYTHON%" -c "raise SystemExit(0)" >nul 2>&1
if errorlevel 1 goto make_environment
goto check_packages

:make_environment
rem 3.11, 3.12 or 3.13 -- NOT "the newest Python", because torch has no wheels for a new version for
rem some months after it appears, and the newest is where `pip install` fails with a wall of red.
rem `py` is the launcher that comes with python.org installs and is the reliable way to ask for a
rem particular version; a bare `python` on Windows is often the Microsoft Store stub, which is why
rem it is tried last.
set "FOUND="

for %%V in (3.12 3.13 3.11) do (
    if not defined FOUND (
        py -%%V -c "raise SystemExit(0)" >nul 2>&1 && set "FOUND=py -%%V"
    )
)

if not defined FOUND (
    python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)" >nul 2>&1 && set "FOUND=python"
)

if not defined FOUND (
    echo.
    echo This computer has no Python 3.11, 3.12 or 3.13, which the app needs.
    echo.
    echo Install one from https://www.python.org/downloads/ -- any 3.12.x will do -- and TICK
    echo "Add python.exe to PATH" on the first page of the installer. Then run this again.
    echo.
    echo Nothing else has to be installed by hand; this file does the rest.
    goto failed
)

echo Setting up the app's own Python environment. This happens once.
echo   using %FOUND%

rem NOT --clear, which empties the folder first. A launcher is not the right thing to be deleting
rem folders with; building over what is there replaces whatever is missing.
%FOUND% -m venv "%VENV%"

if errorlevel 1 (
    echo.
    echo Could not create the environment in %VENV%.
    echo If that folder already exists and is broken, delete it and run this again.
    goto failed
)

:check_packages
"%PYTHON%" -c "import importlib.util, sys; raise SystemExit(0 if all(importlib.util.find_spec(n) for n in sys.argv[1:]) else 1)" %NEEDED% >nul 2>&1
if not errorlevel 1 goto start_app

echo.
echo Installing what the app needs. THE FIRST TIME THIS TAKES 10-20 MINUTES and downloads about
echo 1 GB, mostly the detector's maths library. It only happens once.
echo.

"%PYTHON%" -m pip install --upgrade pip >nul 2>&1
"%PYTHON%" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Could not install the packages the app needs. The message above says which one and why;
    echo the usual causes are no internet connection, or a Python outside 3.11-3.13.
    echo.
    echo Nothing is broken -- run this file again once that is sorted out.
    goto failed
)

"%PYTHON%" -c "import importlib.util, sys; raise SystemExit(0 if all(importlib.util.find_spec(n) for n in sys.argv[1:]) else 1)" %NEEDED% >nul 2>&1

if errorlevel 1 (
    echo.
    echo The environment was built but is still missing something the app needs. Delete the %VENV%
    echo folder in this directory and run this file again to rebuild it from scratch.
    goto failed
)

:start_app
echo.
echo Starting the app. Its messages appear in this window.
echo Closing this window quits the app -- but a count already running carries on to the end,
echo and its numbers are saved when it finishes.
echo.

"%PYTHON%" app.py

rem KEPT OPEN ONLY IF SOMETHING WENT WRONG. A console window that closes on a crash leaves nothing
rem to read, and "it just disappeared" is not a report anyone can act on. On a normal quit there is
rem nothing to say, so it closes.
if errorlevel 1 (
    echo.
    echo The app exited with an error. The lines above are what it said.
    goto failed
)

exit /b 0

:failed
echo.
pause
exit /b 1
