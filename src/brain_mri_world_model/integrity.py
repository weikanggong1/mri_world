"""Hash once; reuse a digest while the file's filesystem fingerprint is unchanged."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import threading

_LOCK = threading.RLock()
_CONNECTION = None
_OWNER = None


def fingerprint(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def cache_path():
    explicit = os.environ.get('BWM_INTEGRITY_CACHE')
    if explicit:
        return Path(explicit).resolve()
    return Path(tempfile.gettempdir()) / f'bwm-integrity-{os.getuid()}' / 'digests.sqlite3'


def _connection():
    global _CONNECTION, _OWNER
    location = cache_path()
    owner = (os.getpid(), str(location))
    if _OWNER != owner:
        if _CONNECTION is not None:
            _CONNECTION.close()
        location.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _CONNECTION = sqlite3.connect(location, timeout=60, check_same_thread=False)
        _CONNECTION.execute('PRAGMA journal_mode=WAL')
        _CONNECTION.execute('PRAGMA synchronous=NORMAL')
        _CONNECTION.execute('CREATE TABLE IF NOT EXISTS digests(path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, sha256 TEXT NOT NULL)')
        _CONNECTION.commit()
        os.chmod(location, 0o600)
        _OWNER = owner
    return _CONNECTION


def record_verified_sha256(path, digest, observed_fingerprint=None):
    """Register a digest computed during this file's successful write/download."""
    path = Path(path).resolve()
    current = fingerprint(path)
    if observed_fingerprint is not None and tuple(observed_fingerprint) != current:
        raise ValueError('File changed after its digest was computed')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Expected a lowercase SHA-256 digest')
    with _LOCK:
        connection = _connection()
        connection.execute('INSERT OR REPLACE INTO digests VALUES(?,?,?)', (str(path), repr(current), digest))
        connection.commit()
    return digest


def file_sha256(path, *, force=False):
    """Read file contents only on a cache miss, changed fingerprint or explicit audit."""
    path = Path(path).resolve()
    before = fingerprint(path)
    if not force:
        with _LOCK:
            row = _connection().execute('SELECT fingerprint,sha256 FROM digests WHERE path=?', (str(path),)).fetchone()
        if row and row[0] == repr(before):
            return row[1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    if fingerprint(path) != before:
        raise ValueError('File changed while its initial digest was computed')
    return record_verified_sha256(path, digest.hexdigest(), before)
