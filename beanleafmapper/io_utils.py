"""Filename parsing and image I/O helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


_FIELD_PATTERN = re.compile(
    r"^(?P<trial>[A-Za-z0-9]+)-B(?P<block>\d+)-S(?P<plot>\d+)"
    r"(?:-P(?P<plant>\d+))?-(?P<seq>\d+\.\d+)$"
)
_LAB_PATTERN = re.compile(
    r"^(?P<trial>[A-Za-z0-9]+)-B(?P<block>\d+)-S(?P<plot>\d+)"
    r"(?:-P(?P<plant>\d+))?-H(?P<leaf>\d+)-(?P<seq>\d+\.\d+)$"
)


@dataclass(frozen=True)
class ImageId:
    """Parsed components of a photo filename."""

    trial: str
    block: int
    plot: int
    sequence: str
    leaf_no: int | None  # set only for lab photos
    path: Path
    plant_no: int | None = None  # optional, present in some date folders (e.g. -P1-)

    @property
    def kind(self) -> str:
        return "lab" if self.leaf_no is not None else "field"

    @property
    def plot_key(self) -> str:
        """Identifies the plot (trial+block+plot+plant). Same value for the matching
        field photo and its lab counterparts within one capture date."""
        if self.plant_no is not None:
            return f"{self.trial}-B{self.block}-S{self.plot}-P{self.plant_no}"
        return f"{self.trial}-B{self.block}-S{self.plot}"

    @property
    def series_key(self) -> str:
        """Coarser identifier (trial+block) used for matching across capture dates,
        where plot numbers (S#) or plant numbers (P#) may drift."""
        return f"{self.trial}-B{self.block}"

    @property
    def stem(self) -> str:
        return self.path.stem


def parse_filename(path: str | Path) -> ImageId:
    """Parse a photo filename into an ImageId. Raises ValueError on unknown format."""
    path = Path(path)
    stem = path.stem
    if (m := _LAB_PATTERN.match(stem)) is not None:
        return ImageId(
            trial=m["trial"],
            block=int(m["block"]),
            plot=int(m["plot"]),
            sequence=m["seq"],
            leaf_no=int(m["leaf"]),
            plant_no=int(m["plant"]) if m["plant"] else None,
            path=path,
        )
    if (m := _FIELD_PATTERN.match(stem)) is not None:
        return ImageId(
            trial=m["trial"],
            block=int(m["block"]),
            plot=int(m["plot"]),
            sequence=m["seq"],
            leaf_no=None,
            plant_no=int(m["plant"]) if m["plant"] else None,
            path=path,
        )
    raise ValueError(f"Unrecognised filename pattern: {stem}")


def list_photos(photos_dir: str | Path, kind: str | None = None) -> list[ImageId]:
    """List parseable photos in a directory, optionally filtered by 'field' or 'lab'."""
    photos_dir = Path(photos_dir)
    images: list[ImageId] = []
    for p in sorted(photos_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            img = parse_filename(p)
        except ValueError:
            continue
        if kind is None or img.kind == kind:
            images.append(img)
    return images


def load_rgb(path: str | Path, downscale: int = 1) -> Image.Image:
    """Open an image as RGB, optionally downscaled by an integer factor."""
    img = Image.open(path).convert("RGB")
    if downscale and downscale > 1:
        img = img.resize((img.size[0] // downscale, img.size[1] // downscale))
    return img
