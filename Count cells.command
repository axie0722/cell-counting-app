#!/bin/bash
# DOUBLE-CLICK THIS ON A MAC. Everything it does is in count-cells.sh next to it; this file exists
# because of what macOS will and will not open from the Finder.
#
# WHY A SEPARATE FILE AT ALL. Finder runs a .command by opening a Terminal window and executing it,
# and it will not do that for a .sh -- double-clicking one of those opens it in a text editor, which
# looks to anyone reasonable like the app is broken. Linux has the opposite convention, so rather
# than keep two copies of a sixty-line script in step, the logic lives in one file and this one
# calls it.
#
# THROUGH bash, NOT BY EXECUTING IT. `./count-cells.sh` would need that file to be marked
# executable, and the executable bit is the first thing lost when files travel: unzipped by some
# tools, copied through Windows, sent through email or a shared drive. Handing the file to bash as an
# argument needs no permission on it at all.
#
# THE FIRST TIME YOU OPEN THIS, macOS may say it "cannot be opened because it is from an
# unidentified developer". That is Gatekeeper, and it is not about this file being wrong -- it is
# about it not being signed by a paid Apple developer account. RIGHT-CLICK the file and choose Open
# instead of double-clicking, and the same box will offer an Open button. Once only. See INSTALL.md.

cd "$(dirname "$0")" || exit 1

exec /bin/bash "./count-cells.sh"
