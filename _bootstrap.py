"""Put the app's own module folders on the import path.

WHY THIS EXISTS. The app is a flat set of modules that import each other by bare name --
`import ps6`, `from app_state import ...`. They used to all sit in one folder, which is the
only folder Python searches automatically (the directory of the script it was told to run).
They now live grouped under `src/` for the sake of anyone reading the tree, and Python does
not look inside subfolders on its own. This adds each of them to the search path so the bare
imports keep resolving exactly as before -- no import statement anywhere had to change.

WHO IMPORTS THIS. Every process the app starts is one of the entry scripts at the repository
root -- app.py and the four scripts it launches (count_cells.py, draw_regions.py,
show_counts.py, app_table.py). Each of them imports this module before its first local import.
A root script's own folder is the repository root, so `import _bootstrap` is found the same way
`import app` would be, and this runs before anything it enables is needed.

ANCHORED TO THIS FILE, not the working directory, for the same reason app_actions.HERE is:
a subprocess is started in the app's folder but the launcher may be run from anywhere, and the
one thing that cannot move relative to `src/` is this file sitting beside it.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")

# The order does not matter: every module basename is unique across these folders, so there is
# nothing for one to shadow in another. Inserted at the front so the app's own modules win over
# any same-named package that happens to be installed in the environment.
for _folder in ("app", "model", "regions", "boundaries", "review"):
    _path = os.path.join(_SRC, _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)
