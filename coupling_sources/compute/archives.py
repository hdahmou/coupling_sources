from __future__ import annotations

import tarfile
from pathlib import Path


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        try:
            for member in members:
                tarfile.data_filter(member, str(output_root))
            archive.extractall(output_root, members=members, filter="data")
        except tarfile.FilterError as exc:
            raise ValueError(
                f"unsafe archive member path: {exc.tarinfo.name}"
            ) from exc
