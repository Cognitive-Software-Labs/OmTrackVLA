"""Checksums and never-overwrite output paths for recorded evidence."""
import hashlib
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_new_path(path):
    """Refuse to reuse an output path so a prior recording is never overwritten."""
    if Path(path).exists():
        raise FileExistsError(f'{path} already exists; choose a new output path')
