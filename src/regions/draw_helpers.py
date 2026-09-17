"""Finding a section image by a fragment of its name.

Lifted out of `draw_regions.py` unchanged, because that file does its work at MODULE level -- importing it
opens napari -- so anything else that wants to resolve an image name cannot import it.
"""
from pathlib import Path


def resolve_image(wanted):
    """An exact path, a bare filename, or any fragment of one.

    The sections live in two places -- some in birds/, some in the project root -- so typing exact names means
    remembering which. A fragment like "Gre595_RH_3-1-4" is enough. An AMBIGUOUS fragment stops with the list
    rather than picking one: outlines are saved next to whichever image was opened, so a silent wrong guess
    writes NCM for one section into another section's file.
    """
    if Path(wanted).exists():
        return wanted

    if (Path("birds") / wanted).exists():
        return str(Path("birds") / wanted)

    matches = sorted(
        str(path)
        for folder in (Path("."), Path("birds"))
        for path in folder.glob("*.TIF")
        if wanted.lower() in path.name.lower()
    )

    if len(matches) == 1:
        print(f"  matched {Path(matches[0]).name}")
        return matches[0]

    if not matches:
        raise SystemExit(f"No image matching {wanted!r} in . or birds/")

    listed = "\n  ".join(matches)
    raise SystemExit(f"{wanted!r} matches {len(matches)} images -- be more specific:\n  {listed}")
