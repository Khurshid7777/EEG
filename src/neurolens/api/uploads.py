"""Uploaded recordings -> a local file the ingestors can read. A single file is saved as is; a
``.zip`` carries multi-file formats (BrainVision .vhdr/.vmrk/.eeg, EEGLAB .set/.fdt, WFDB .hea/.mat,
Nihon Kohden .EEG/.21E/.LOG/.PNT, Persyst .lay/.dat)
and is unpacked flat (names only - no paths from the archive are used) into a private directory.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

# the file the ingestor is pointed at; ".eeg" last: inside a BrainVision archive the .vhdr wins
PRIMARY = (".vhdr", ".set", ".hea", ".lay", ".edf", ".bdf", ".trc", ".eeg")
MAX_MEMBERS = 64
MAX_UNPACKED_BYTES = 8 << 30


class UploadError(ValueError):
    pass


def materialize(data: bytes, filename: str | None) -> tuple[Path, Path]:
    """(path to read, directory to remove afterwards)."""
    d = Path(tempfile.mkdtemp(prefix="neurolens_upload_"))
    name = Path(filename or "upload.edf").name
    if Path(name).suffix.lower() != ".zip":
        p = d / f"upload{Path(name).suffix or '.edf'}"
        p.write_bytes(data)
        return p, d
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        shutil.rmtree(d, ignore_errors=True)
        raise UploadError(f"not a zip archive: {exc}") from exc
    members = [m for m in z.infolist() if not m.is_dir() and not Path(m.filename).name.startswith(".")]
    if len(members) > MAX_MEMBERS or sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
        shutil.rmtree(d, ignore_errors=True)
        raise UploadError("archive too large")
    for m in members:
        out = d / Path(m.filename).name           # flat: never a path taken from the archive
        with z.open(m) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    for ext in PRIMARY:
        found = sorted(p for p in d.iterdir() if p.suffix.lower() == ext)
        if len(found) == 1:
            return found[0], d
        if len(found) > 1:
            shutil.rmtree(d, ignore_errors=True)
            raise UploadError(f"archive holds {len(found)} '{ext}' recordings; send one per request")
    shutil.rmtree(d, ignore_errors=True)
    raise UploadError(f"no recording in the archive (expected one of {', '.join(PRIMARY)})")
