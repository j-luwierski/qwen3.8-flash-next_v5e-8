"""Manual Kaggle artifact exchange, including incomplete/resumable jobs."""

import os
import shutil
import zipfile
from pathlib import Path

from .io import atomic_write, digest, read_json, sha256, write_json
from .runtime import check, resumable_copy


EXTRACT_STATE = '.extract-state.json'


def unpack_cached(path, destination, control=None):
    path, destination = Path(path), Path(destination)
    check(control, 'artifact_extract')
    identity = sha256(path)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / EXTRACT_STATE
    state = read_json(metadata) if metadata.exists() else dict(archive=identity, files={})
    if state['archive'] != identity:
        raise ValueError('Extraction directory belongs to another archive')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [info.filename for info in entries]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive entries')
        for info in entries:
            target = destination / info.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive path')
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError('Archives cannot contain symbolic links')
            if info.filename in (EXTRACT_STATE, '.restore.json'):
                raise ValueError('Reserved archive metadata name')
        write_json(metadata, state)
        for info in entries:
            check(control, 'artifact_extract')
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if info.filename in state['files']:
                if not target.exists() or sha256(target) != state['files'][info.filename]:
                    raise ValueError('Corrupted extracted artifact')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.extracting')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > info.file_size:
                raise ValueError('Invalid partial extraction size')
            with archive.open(info) as source, partial.open('ab') as output:
                source.seek(offset)
                try:
                    while True:
                        check(control, 'artifact_extract')
                        block = source.read(8*1024**2)
                        if not block:
                            break
                        output.write(block)
                finally:
                    output.flush()
                    os.fsync(output.fileno())
            # Re-reading validates the zip CRC as well as a previously saved
            # partial prefix. Never trust a prefix merely because its size fits.
            expected_crc = 0
            import zlib
            with partial.open('rb') as file:
                for block in iter(lambda: file.read(8*1024**2), b''):
                    expected_crc = zlib.crc32(block, expected_crc)
            if partial.stat().st_size != info.file_size or expected_crc != info.CRC:
                raise ValueError('Extracted artifact CRC mismatch')
            os.replace(partial, target)
            state['files'][info.filename] = sha256(target)
            write_json(metadata, state)
    return destination


def snapshot_job(root, destination):
    """Fast stored ZIP; exclude input extraction caches and older ZIP copies."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root):
        raise ValueError('Resume archive must be outside the job directory')
    def write(file):
        with zipfile.ZipFile(file, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(root.rglob('*')):
                relative = path.relative_to(root)
                if not path.is_file() or path.is_symlink():
                    continue
                if path.name.endswith(('.tmp', '.lock', '.zip')) or str(relative) in (EXTRACT_STATE, '.restore.json'):
                    continue
                archive.write(path, str(relative))
    atomic_write(destination, write)
    write_json(destination.with_suffix('.sha256.json'), dict(sha256=sha256(destination)))
    return destination


def restore_job(source, root, control=None):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root:
        return root
    if source.is_dir() and (source.is_relative_to(root) or root.is_relative_to(source)):
        raise ValueError('Resume source and output must be separate directories')
    if source.is_file():
        identity = dict(archive=sha256(source))
    else:
        identity = dict(files={str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*'))
                               if p.is_file() and not p.is_symlink() and not p.name.endswith(('.lock', '.tmp'))
                               and str(p.relative_to(source)) not in (EXTRACT_STATE, '.restore.json')})
        if not identity['files']:
            raise ValueError('Empty or missing resume source')
    marker = root / '.restore.json'
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        saved = read_json(marker)
        if saved['source'] != identity:
            raise ValueError('Resume source changed during restoration')
        if saved['complete']:
            return root  # A newer checkpoint in the writable directory wins.
    elif any(root.iterdir()):
        raise ValueError('Output already has files: omit resume_from or use a fresh output')
    write_json(marker, dict(source=identity, complete=False))
    if source.is_file():
        unpack_cached(source, root, control)
    else:
        for name, checksum in identity['files'].items():
            check(control, 'restore')
            resumable_copy(source / name, root / name, checksum, control)
    write_json(marker, dict(source=identity, complete=True))
    return root
