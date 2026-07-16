from __future__ import annotations

import io
import tarfile

import pytest

from coupling_sources.compute.archives import safe_extract_tar


def _write_archive(archive_path, members: dict[str, bytes]) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _write_symlink_archive(archive_path, name: str, target: str) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        archive.addfile(info)


def test_safe_extract_tar_extracts_safe_members(tmp_path) -> None:
    archive_path = tmp_path / "safe.tar.gz"
    output_dir = tmp_path / "out"
    _write_archive(archive_path, {"nested/result.txt": b"ok"})

    safe_extract_tar(archive_path, output_dir)

    assert (output_dir / "nested" / "result.txt").read_bytes() == b"ok"


def test_safe_extract_tar_rejects_path_traversal(tmp_path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    output_dir = tmp_path / "out"
    _write_archive(archive_path, {"../escape.txt": b"nope"})

    with pytest.raises(ValueError, match="unsafe archive member path"):
        safe_extract_tar(archive_path, output_dir)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_tar_rejects_link_outside_destination(tmp_path) -> None:
    archive_path = tmp_path / "unsafe-link.tar.gz"
    output_dir = tmp_path / "out"
    _write_symlink_archive(archive_path, "nested/link", "../../escape.txt")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        safe_extract_tar(archive_path, output_dir)

    assert not (output_dir / "nested" / "link").exists()
