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
echo Installing what the app needs: about 1 GB, mostly the detector's maths library.
echo THIS HAPPENS ONCE. A few minutes on a fast connection, longer on a slow one -- most of it is
echo the download, so it goes at the speed of your internet.
echo.

rem --- WHO DOES THE INSTALLING: uv if it can be had, pip otherwise -------------------------------
rem Both install the same pinned versions from requirements.txt. uv is the same job with the waiting
rem taken out: it downloads the packages in parallel instead of one after another, resolves the
rem versions in seconds, and hard-links them from a cache instead of unpacking and byte-compiling
rem every file. Measured on a Mac with an empty cache, same 1 GB, same connection: uv 89 seconds
rem against pip's 174. If uv cannot be had, pip does the work and the app is just as good.
rem
rem NOT `curl ... | sh`, which is how uv's own site installs it: that asks somebody to run a script
rem off the internet on the say-so of a program they were emailed. The wheel on PyPI is the same
rem tool through a channel they already use.
set "UV="
where uv >nul 2>&1 && set "UV=uv"

if defined UV (
    echo   using uv, which is already installed here
    goto install_packages
)

"%PYTHON%" -m pip install --upgrade pip >nul 2>&1
"%PYTHON%" -m pip install --quiet uv >nul 2>&1

if not errorlevel 1 (
    set "UV=%PYTHON% -m uv"
    echo   fetched uv, a faster installer, to do it with
) else (
    echo   using pip ^(uv was not available here -- this works, it is only slower^)
)

:install_packages
rem PREBUILT PACKAGES FIRST, SOURCE ONLY IF IT MUST. --only-binary means "download something already
rem built, or fail" -- and failing in ten seconds is kinder than a twenty-minute compile that then
rem stops on a missing C compiler, which Windows does not have by default. The second attempt drops
rem the restriction, so a dependency that genuinely ships only source still gets in.
rem
rem WRITTEN WITH LABELS, NOT NESTED BRACKETS: inside a parenthesised block cmd.exe reads errorlevel
rem as it was BEFORE the block began, so "run this, then check whether it failed" has to be written
rem flat, one statement per line, to check the thing that actually just ran.
if not defined UV goto install_with_pip

%UV% pip install --python "%PYTHON%" --only-binary :all: -r requirements.txt
if not errorlevel 1 goto installed

echo.
echo   something here has no prebuilt version -- trying again the slow way
%UV% pip install --python "%PYTHON%" -r requirements.txt
if not errorlevel 1 goto installed
goto install_failed

:install_with_pip
"%PYTHON%" -m pip install --only-binary :all: -r requirements.txt
if not errorlevel 1 goto installed

echo.
echo   something here has no prebuilt version -- trying again the slow way
"%PYTHON%" -m pip install -r requirements.txt
if not errorlevel 1 goto installed

:install_failed
echo.
echo Could not install the packages the app needs. The message above says which one and why;
echo the usual causes are no internet connection, or a Python outside 3.11-3.13.
echo.
echo Nothing is broken -- run this file again once that is sorted out.
goto failed

:installed

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
