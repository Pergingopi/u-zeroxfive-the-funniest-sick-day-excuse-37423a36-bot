#!/usr/bin/env python3
"""The Funniest Sick-Day Excuse — make.fun entry.

Writes exactly one line (1-200 characters, no trailing newline) to
``$MAKEFUN_OUTPUT_DIR/entry.txt``. Python 3.13 standard library only.
Output is fully deterministic: the same entry bytes on every run,
regardless of who runs it or where.

Write path is hardened for the grader. The entry is written to a
temporary file in the SAME directory as the target (``mkstemp`` with
``O_EXCL`` so no other process can guess or pre-create the name),
written in a loop until every byte lands, fsynced, chmodded 0644,
then atomically moved into place with ``os.replace``, chmodded 0644
again, and verified by read-back (exact bytes plus mode 0644).
The output directory itself is never deleted, and leftover temp
files from an interrupted earlier run are cleaned up first, with
a small retry, before the new write starts.

Validation is split into one small function per gate rule so a
failure names the exact rule that tripped, and every failure raises
an explicit ``ValueError`` — never an ``assert``, which could be
silently stripped with ``python -O``.
"""

import os
import sys
import tempfile
import time

ENTRY = (
    "Can't come in today \u2014 I slept on my arm wrong and it's still "
    "asleep, so technically half of me is already on sick leave."
)

OUTPUT_FILENAME = "entry.txt"
TMP_PREFIX = "entry.txt.tmp-"
MAX_ENTRY_CHARS = 200
DEFAULT_OUTPUT_DIR = "/output"
STALE_CLEANUP_ATTEMPTS = 3
STALE_CLEANUP_DELAY_SECONDS = 0.05


# ---------------------------------------------------------------------------
# Validation: one function per gate rule, orchestrated by validate_entry().
# ---------------------------------------------------------------------------

def _validate_not_blank(text: str) -> None:
    """Reject an empty entry or one that is only whitespace."""
    if not text or not text.strip():
        raise ValueError("entry text is empty or only whitespace")


def _validate_length(text: str) -> None:
    """Reject anything over the 200-character gate."""
    if len(text) > MAX_ENTRY_CHARS:
        raise ValueError(
            f"entry text is {len(text)} characters "
            f"(> {MAX_ENTRY_CHARS})"
        )


def _validate_single_line(text: str) -> None:
    """Reject embedded line breaks: the entry must be a single line."""
    if "\n" in text or "\r" in text:
        raise ValueError("entry text contains a line break; must be one line")


def _validate_no_control_chars(text: str) -> None:
    """Reject control characters (including DEL) anywhere in the text."""
    for ch in text:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            raise ValueError(
                f"entry text contains control character U+{code:04X}"
            )


def _validate_no_surrounding_whitespace(text: str) -> None:
    """Reject leading or trailing whitespace, including a trailing newline."""
    if text != text.strip():
        raise ValueError("entry text has leading or trailing whitespace")


def validate_entry(text: str) -> None:
    """Run every gate check; raise ValueError naming the first failure."""
    _validate_not_blank(text)
    _validate_length(text)
    _validate_single_line(text)
    _validate_no_control_chars(text)
    _validate_no_surrounding_whitespace(text)


# ---------------------------------------------------------------------------
# Output-directory resolution.
# ---------------------------------------------------------------------------

def _resolve_output_dir() -> str:
    """Return the output directory, falling back when the env var is
    unset, empty, or only whitespace (an empty value is not a directory
    and must not be treated as one)."""
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if raw is None:
        return DEFAULT_OUTPUT_DIR
    cleaned = raw.strip()
    if not cleaned:
        return DEFAULT_OUTPUT_DIR
    return cleaned


# ---------------------------------------------------------------------------
# Durable-write helpers.
# ---------------------------------------------------------------------------

def _fsync_dir(path: str) -> None:
    """Flush a directory entry to disk so the rename survives a crash."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_with_retry(path: str) -> bool:
    """Try to unlink ``path`` a few times; return True if it is gone.

    A stale temp file can be briefly busy on some filesystems, so one
    failed attempt is not proof it cannot be removed. After the last
    attempt the file is left alone — cleanup is best-effort and must
    never abort the real write.
    """
    for attempt in range(STALE_CLEANUP_ATTEMPTS):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 < STALE_CLEANUP_ATTEMPTS:
                time.sleep(STALE_CLEANUP_DELAY_SECONDS)
    return False


def _cleanup_stale_tmp(output_dir: str) -> int:
    """Best-effort removal of our own leftover temp files from prior
    runs. Returns how many were removed. Never raises: if the directory
    cannot even be listed, there is nothing safe to clean."""
    try:
        names = os.listdir(output_dir)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if name.startswith(TMP_PREFIX):
            if _remove_with_retry(os.path.join(output_dir, name)):
                removed += 1
    return removed


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte; ``os.write`` may legitimately write a prefix."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError("os.write wrote 0 bytes; refusing to spin forever")
        view = view[written:]


def _write_atomic(output_dir: str, data: bytes) -> str:
    """Write ``data`` to ``entry.txt`` in ``output_dir`` atomically and
    durably, returning the target path. The temp file is always removed
    on failure, and the file descriptor is always closed."""
    os.makedirs(output_dir, exist_ok=True)
    _cleanup_stale_tmp(output_dir)

    target = os.path.join(output_dir, OUTPUT_FILENAME)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=TMP_PREFIX, dir=output_dir)
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
            fd = None
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
        tmp_path = None
        os.chmod(target, 0o644)
        _fsync_dir(output_dir)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return target


def _verify_entry(target: str, data: bytes) -> None:
    """Read-back verification: exact bytes and mode 0644, or raise."""
    with open(target, "rb") as handle:
        actual = handle.read()
    if actual != data:
        raise ValueError("read-back mismatch: entry.txt does not match entry text")
    mode = os.stat(target).st_mode & 0o777
    if mode != 0o644:
        raise ValueError(f"entry.txt mode is {mode:03o}, expected 0644")


def write_entry(output_dir: str) -> str:
    """Validate, atomically write, and verify the entry; return its path."""
    validate_entry(ENTRY)
    data = ENTRY.encode("utf-8")
    target = _write_atomic(output_dir, data)
    _verify_entry(target, data)
    return target


def main() -> int:
    output_dir = _resolve_output_dir()
    try:
        target = write_entry(output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target} ({len(ENTRY)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
