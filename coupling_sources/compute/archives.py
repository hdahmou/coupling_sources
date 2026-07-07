from __future__ import annotations

import tarfile
from pathlib import Path


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (output_root / member.name).resolve()
            if output_root != target and output_root not in target.parents:
                raise ValueError(f"unsafe archive member path: {member.name}")
        archive.extractall(output_root, members=members, filter="data")
