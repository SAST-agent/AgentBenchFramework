"""Train-only, synthetic-only structured Replay Reading for 24_miracle.

The module parses approved canonical bytes into an ordered timeline. It never
starts a Judge, policy, Provider, runner, workspace, store, log, or session.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import ntpath
import os
import platform
import stat
import weakref
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

from agentbench_frame.eval.measurement import canonical_state_id
from agentbench_frame.games.miracle.research_protocol import (
    SEED_MAX,
    SEED_MIN,
    build_action_support,
    canonical_command,
    command_action_id,
    enumerate_legal_commands,
)


REPLAY_READING_MANIFEST_SCHEMA_VERSION = "24-miracle-replay-reading-manifest-v1"
SYNTHETIC_REPLAY_SCHEMA_VERSION = "24-miracle-synthetic-replay-v1"
RATIONALE_STATUS = "not_recorded"
MAX_REPLAY_MANIFEST_BYTES = 256 * 1024
MAX_SYNTHETIC_REPLAY_BYTES = 16 * 1024 * 1024

# Production is intentionally empty. Tests may temporarily monkeypatch it.
APPROVED_TRAINING_REPLAY_MANIFESTS: frozenset[str] = frozenset()
_SAFE_OPEN_BARRIER = None
_NATIVE_WINDOWS = os.name == "nt"

# Pure Win32/NT numeric values are platform-independent.  Keep them available
# for synthetic validation on POSIX without importing a DLL or touching a
# native Windows API.  Native structures and function bindings remain guarded
# below.
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_GENERIC_READ = 0x80000000
_FILE_LIST_DIRECTORY = 0x0001
_FILE_TRAVERSE = 0x0020
_FILE_READ_ATTRIBUTES = 0x0080
_SYNCHRONIZE = 0x00100000
_FILE_SHARE_READ = 0x0001
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x0010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_TYPE_DISK = 0x0001
_OBJ_CASE_INSENSITIVE = 0x00000040
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_OPEN = 0x00000001

if _NATIVE_WINDOWS:
    from ctypes import wintypes
    class _WinFileInfo(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _CreateFileW.restype = wintypes.HANDLE
    _GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_WinFileInfo),
    )
    _GetFileInformationByHandle.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _GetFileType = _kernel32.GetFileType
    _GetFileType.argtypes = (wintypes.HANDLE,)
    _GetFileType.restype = wintypes.DWORD
    _ReadFile = _kernel32.ReadFile
    _ReadFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    _ReadFile.restype = wintypes.BOOL
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = (wintypes.HANDLE,)
    _CloseHandle.restype = wintypes.BOOL
    class _WinUnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _WinObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_WinUnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class _WinIoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("Status", wintypes.LONG),
            ("Information", ctypes.c_size_t),
        ]

    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _NtCreateFile = _ntdll.NtCreateFile
    _NtCreateFile.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_WinObjectAttributes),
        ctypes.POINTER(_WinIoStatusBlock),
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _NtCreateFile.restype = wintypes.LONG


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


_OPENAT2_SYSCALL_BY_MACHINE = {
    "aarch64": 437,
    "x86_64": 437,
    "amd64": 437,
}
_RESOLVE_NO_XDEV = 0x01
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08


def _json_compatible(value: Any) -> Any:
    if type(value) in (dict, MappingProxyType):
        return {key: _json_compatible(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_compatible(item) for item in value]
    return value


def canonical_replay_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_compatible(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must be strict UTF-8 without BOM")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} must be strict canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        canonical = canonical_replay_json_bytes(value)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid JSON data") from exc
    if payload != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _strict_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_seed(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not SEED_MIN <= value <= SEED_MAX
    ):
        raise ValueError(f"{label} must be a strict frozen-range seed")
    return value


def _strict_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an int or float, not bool")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if type(value) is int and int(number) != value:
        raise ValueError(f"{label} integer must be exactly representable as a float")
    return number


def _relative_parts(relative: Any, label: str) -> tuple[str, ...]:
    if (
        type(relative) is not str
        or not relative
        or "\x00" in relative
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
    ):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a relative path without escape")
    return tuple(parts)
def _safe_open_barrier(label: str, relative: str, component_index: int) -> None:
    hook = _SAFE_OPEN_BARRIER
    if hook is not None:
        hook(label, relative, component_index)
def _read_descriptor(fd: int, limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = limit + 1
    while True:
        requested = min(1024 * 1024, remaining)
        chunk = os.read(fd, requested)
        if type(chunk) is not bytes:
            raise ValueError(f"{label} reader returned a non-bytes chunk")
        if len(chunk) > requested:
            raise ValueError(f"{label} reader exceeded its bounded request")
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
        if remaining == 0:
            raise ValueError(f"{label} exceeds its frozen byte limit")
def _strict_root_identity(value: Any) -> tuple[Any, ...]:
    if type(value) is not tuple or not value or type(value[0]) is not str:
        raise ValueError("approved replay root identity schema is invalid")
    expected_length = {"posix": 3, "windows": 4}.get(value[0])
    if expected_length is None or len(value) != expected_length:
        raise ValueError("approved replay root identity schema is invalid")
    if any(type(field) is not int for field in value[1:]):
        raise ValueError("approved replay root identity fields must be strict integers")
    return value
def _root_identities_match(actual: Any, expected: Any) -> bool:
    return _strict_root_identity(actual) == _strict_root_identity(expected)
class _ApprovedRoot:
    def __init__(
        self,
        path: str,
        identity: tuple[Any, ...],
        handle: Any,
        final_path: str | None = None,
    ) -> None:
        self.path = path
        self.identity = _strict_root_identity(identity)
        self.handle = handle
        self.final_path = final_path
    def __enter__(self) -> "_ApprovedRoot":
        return self
    def __exit__(self, *_: Any) -> None:
        self.close()
    def close(self) -> None:
        if self.handle is None:
            return
        if _NATIVE_WINDOWS:
            _CloseHandle(self.handle)
        else:
            os.close(self.handle)
        self.handle = None
    def read(self, relative: Any, label: str, limit: int) -> bytes:
        parts = _relative_parts(relative, label)
        if type(limit) is not int or limit <= 0:
            raise ValueError("safe read limit must be a positive strict integer")
        if _NATIVE_WINDOWS:
            return _read_relative_windows(self, parts, relative, label, limit)
        return _read_relative_posix(self, parts, relative, label, limit)


def _posix_open_component(parent_fd: int, component: str, flags: int) -> int:
    """Open one Linux path component without symlink or mount traversal."""

    if _NATIVE_WINDOWS or platform.system() != "Linux":
        raise NotImplementedError("Linux openat2 is required for approved roots")
    syscall_number = _OPENAT2_SYSCALL_BY_MACHINE.get(platform.machine().lower())
    if syscall_number is None:
        raise NotImplementedError("openat2 syscall number is unknown on this platform")
    if (
        type(component) is not str
        or not component
        or component in {".", ".."}
        or "/" in component
        or "\x00" in component
    ):
        raise ValueError("approved replay root component is unsafe")
    encoded = os.fsencode(component)
    how = _OpenHow(
        flags=flags,
        mode=0,
        resolve=_RESOLVE_NO_XDEV | _RESOLVE_NO_SYMLINKS | _RESOLVE_BENEATH,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    result = syscall(
        ctypes.c_long(syscall_number),
        ctypes.c_int(parent_fd),
        ctypes.c_char_p(encoded),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), component)
    return int(result)


def _open_approved_root(
    root: Path, expected_identity: tuple[Any, ...] | None = None
) -> _ApprovedRoot:
    raw_root = _bind_exact_path(root, "approved replay root", allow_string=False)
    root_path = (ntpath if _NATIVE_WINDOWS else os.path).abspath(raw_root)
    if _NATIVE_WINDOWS:
        approved = _open_windows_root(root_path, expected_identity)
    else:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory is None:
            raise ValueError("platform cannot prove safe approved-root traversal")
        flags = (
            os.O_RDONLY
            | nofollow
            | directory
            | getattr(os, "O_CLOEXEC", 0)
        )
        opened: list[int] = []
        try:
            anchor_fd = os.open("/", flags)
            opened.append(anchor_fd)
            info = os.fstat(anchor_fd)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("filesystem anchor must be a real directory")
            anchor_device = info.st_dev
            current_fd = anchor_fd
            components = tuple(part for part in root_path.split("/") if part)
            for component in components:
                fd = _posix_open_component(current_fd, component, flags)
                opened.append(fd)
                info = os.fstat(fd)
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError("approved replay root must be a real directory")
                if info.st_dev != anchor_device:
                    raise ValueError("approved replay root crosses a mount device")
                current_fd = fd
            final_fd = opened[-1]
            approved = _ApprovedRoot(
                root_path,
                ("posix", info.st_dev, info.st_ino),
                final_fd,
            )
            if expected_identity is not None and not _root_identities_match(
                approved.identity, expected_identity
            ):
                raise ValueError("approved replay root identity changed")
            opened.pop()
        except (OSError, NotImplementedError):
            raise ValueError("approved replay root is unsafe or unavailable") from None
        finally:
            for fd in reversed(opened):
                os.close(fd)
    return approved


def _reopen_current_posix_root(root: _ApprovedRoot) -> _ApprovedRoot:
    """Rebind a read to the current lexical root and its retained identity."""

    return _open_approved_root(Path(root.path), root.identity)


def _read_relative_posix(
    root: _ApprovedRoot,
    parts: tuple[str, ...],
    relative: str,
    label: str,
    limit: int,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ValueError("platform cannot prove safe relative-file traversal")
    try:
        with ExitStack() as ownership:
            current_root = ownership.enter_context(
                _reopen_current_posix_root(root)
            )
            current_fd = os.dup(current_root.handle)
            ownership.callback(os.close, current_fd)
            for index, part in enumerate(parts[:-1]):
                _safe_open_barrier(label, relative, index)
                next_fd = _posix_open_component(
                    current_fd,
                    part,
                    os.O_RDONLY
                    | nofollow
                    | directory
                    | getattr(os, "O_CLOEXEC", 0),
                )
                ownership.callback(os.close, next_fd)
                info = os.fstat(next_fd)
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(
                        f"{label} intermediate component is not a directory"
                    )
                if info.st_dev != root.identity[1]: raise ValueError(f"{label} crosses approved-root device")
                current_fd = next_fd
            _safe_open_barrier(label, relative, len(parts) - 1)
            file_fd = _posix_open_component(
                current_fd,
                parts[-1],
                os.O_RDONLY
                | nofollow
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            ownership.callback(os.close, file_fd)
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} final object is not a regular file")
            if before.st_dev != root.identity[1]: raise ValueError(f"{label} crosses approved-root device")
            if before.st_size > limit:
                raise ValueError(f"{label} exceeds its frozen byte limit")
            payload = _read_descriptor(file_fd, limit, label)
            after = os.fstat(file_fd)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity or len(payload) != before.st_size:
                raise ValueError(f"{label} changed while being read")
            return payload
    except OSError:
        raise ValueError(f"{label} is unsafe or unavailable") from None
def _win_information(handle: Any, label: str) -> Any:
    information = _WinFileInfo()
    if not _GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise ValueError(f"{label} handle metadata is unavailable")
    return information
def _win_identity(information: Any) -> tuple[Any, ...]:
    return (
        "windows",
        information.dwVolumeSerialNumber,
        information.nFileIndexHigh,
        information.nFileIndexLow,
    )
def _win_metadata(information: Any) -> tuple[int, ...]:
    return (
        information.dwFileAttributes,
        information.ftCreationTime.dwHighDateTime,
        information.ftCreationTime.dwLowDateTime,
        information.nFileSizeHigh,
        information.nFileSizeLow,
        information.nNumberOfLinks,
        information.ftLastWriteTime.dwHighDateTime,
        information.ftLastWriteTime.dwLowDateTime,
        information.dwVolumeSerialNumber,
        information.nFileIndexHigh,
        information.nFileIndexLow,
    )
def _win_final_path(handle: Any, label: str) -> str:
    size = _GetFinalPathNameByHandleW(handle, None, 0, 0)
    if not size:
        raise ValueError(f"{label} final handle path is unavailable")
    while True:
        buffer = ctypes.create_unicode_buffer(size + 1)
        written = _GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not written:
            raise ValueError(f"{label} final handle path is unavailable")
        if written < len(buffer):
            return _win_comparable_path(buffer.value)
        size = written
def _win_is_beneath(path: str, root: str) -> bool:
    try:
        normalized_root = _win_comparable_path(root)
        normalized_path = _win_comparable_path(path)
        return (
            normalized_path != normalized_root
            and ntpath.commonpath((normalized_root, normalized_path))
            == normalized_root
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _win_comparable_path(path: str) -> str:
    normalized = path
    if normalized[:8].lower() == "\\\\?\\unc\\":
        normalized = "\\\\" + normalized[8:]
    elif normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]
    return ntpath.normcase(ntpath.normpath(normalized))


def _win_paths_match(actual: str, expected: str) -> bool:
    return _win_comparable_path(actual) == _win_comparable_path(expected)


def _win_validate_opened_handle(
    handle: Any, information: Any, *, directory: bool, label: str
) -> None:
    attributes = information.dwFileAttributes
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"{label} contains a reparse-point component")
    is_directory = bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)
    if directory != is_directory:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} component is not a {kind}")
    if not directory and _GetFileType(handle) != _FILE_TYPE_DISK:
        raise ValueError(f"{label} final object is not a regular disk file")


def _win_open_checked(path: str, *, directory: bool, label: str) -> tuple[Any, Any]:
    access = _FILE_READ_ATTRIBUTES | (
        _FILE_LIST_DIRECTORY | _FILE_TRAVERSE if directory else _GENERIC_READ
    )
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    handle = _CreateFileW(
        path,
        access,
        _FILE_SHARE_READ,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle in (None, _INVALID_HANDLE_VALUE):
        raise ValueError(f"{label} is unsafe or unavailable")
    try:
        information = _win_information(handle, label)
        _win_validate_opened_handle(
            handle, information, directory=directory, label=label
        )
        return handle, information
    except Exception:
        _CloseHandle(handle)
        raise


def _win_open_relative_checked(
    parent_handle: Any, component: str, *, directory: bool, label: str
) -> tuple[Any, Any]:
    """Atomically open one exact child relative to an already checked handle."""

    if (
        not _NATIVE_WINDOWS
        or type(component) is not str
        or not component
        or component in {".", ".."}
        or any(character in component for character in "\\/:")
        or "\x00" in component
    ):
        raise ValueError(f"{label} relative component is unsafe")
    buffer = ctypes.create_unicode_buffer(component)
    encoded_length = len(component.encode("utf-16-le"))
    name = _WinUnicodeString(
        Length=encoded_length,
        MaximumLength=encoded_length + 2,
        Buffer=ctypes.cast(buffer, wintypes.LPWSTR),
    )
    attributes = _WinObjectAttributes(
        Length=ctypes.sizeof(_WinObjectAttributes),
        RootDirectory=parent_handle,
        ObjectName=ctypes.pointer(name),
        Attributes=_OBJ_CASE_INSENSITIVE,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = _WinIoStatusBlock()
    handle = wintypes.HANDLE()
    access = _SYNCHRONIZE | _FILE_READ_ATTRIBUTES | (
        _FILE_LIST_DIRECTORY | _FILE_TRAVERSE if directory else _GENERIC_READ
    )
    options = (
        _FILE_FLAG_OPEN_REPARSE_POINT
        | _FILE_SYNCHRONOUS_IO_NONALERT
        | (_FILE_DIRECTORY_FILE if directory else _FILE_NON_DIRECTORY_FILE)
    )
    status = _NtCreateFile(
        ctypes.byref(handle),
        access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0,
        _FILE_SHARE_READ,
        _FILE_OPEN,
        options,
        None,
        0,
    )
    if status < 0 or handle.value in (None, _INVALID_HANDLE_VALUE):
        raise ValueError(f"{label} is unsafe or unavailable")
    opened_handle = handle.value
    try:
        information = _win_information(opened_handle, label)
        _win_validate_opened_handle(
            opened_handle, information, directory=directory, label=label
        )
        return opened_handle, information
    except Exception:
        _CloseHandle(opened_handle)
        raise
def _open_windows_root(
    root_path: str, expected_identity: tuple[Any, ...] | None = None
) -> _ApprovedRoot:
    drive, tail = ntpath.splitdrive(root_path)
    if not drive or not tail.startswith(("\\", "/")):
        raise ValueError("approved replay root must have a trusted Windows anchor")
    anchor = drive + "\\"
    components = tuple(
        component for component in tail.replace("/", "\\").split("\\") if component
    )
    handles: list[Any] = []
    try:
        current_path = anchor
        handle, information = _win_open_checked(
            anchor, directory=True, label="approved replay root"
        )
        handles.append(handle)
        final_path = _win_final_path(handle, "approved replay root")
        if not _win_paths_match(final_path, anchor):
            raise ValueError("approved replay root anchor path changed")
        current_handle = handle
        for component in components:
            current_path = ntpath.join(current_path, component)
            handle, information = _win_open_relative_checked(
                current_handle,
                component,
                directory=True,
                label="approved replay root",
            )
            handles.append(handle)
            final_path = _win_final_path(handle, "approved replay root")
            if not _win_paths_match(final_path, current_path):
                raise ValueError("approved replay root component path changed")
            current_handle = handle
        final_handle = handles[-1]
        approved = _ApprovedRoot(
            root_path,
            _win_identity(information),
            final_handle,
            final_path,
        )
        if expected_identity is not None and not _root_identities_match(
            approved.identity, expected_identity
        ):
            raise ValueError("approved replay root identity changed")
        handles.pop()
        return approved
    finally:
        for handle in reversed(handles):
            _CloseHandle(handle)
def _win_read(handle: Any, label: str, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit + 1
    while True:
        requested = min(1024 * 1024, remaining)
        buffer = ctypes.create_string_buffer(requested)
        read = wintypes.DWORD()
        if not _ReadFile(handle, buffer, requested, ctypes.byref(read), None):
            raise ValueError(f"{label} could not be read from its safe handle")
        if read.value > requested:
            raise ValueError(f"{label} reader exceeded its bounded request")
        if read.value == 0:
            return b"".join(chunks)
        chunks.append(buffer.raw[: read.value])
        remaining -= read.value
        if remaining == 0:
            raise ValueError(f"{label} exceeds its frozen byte limit")


def _win_revalidate_root(root: _ApprovedRoot) -> None:
    """Reopen every lexical-root component and prove it matches the retained root."""

    retained_information = _win_information(root.handle, "approved replay root")
    if not _root_identities_match(
        _win_identity(retained_information), root.identity
    ):
        raise ValueError("approved replay root identity changed")
    if not _win_paths_match(
        _win_final_path(root.handle, "approved replay root"), root.final_path
    ):
        raise ValueError("approved replay root path changed")
    with _open_windows_root(root.path, root.identity) as current_root:
        if not _win_paths_match(
            current_root.final_path, root.final_path
        ):
            raise ValueError("approved replay root path changed")


def _read_relative_windows(
    root: _ApprovedRoot,
    parts: tuple[str, ...],
    relative: str,
    label: str,
    limit: int,
) -> bytes:
    directory_handles: list[Any] = []
    file_handle: Any = None
    try:
        _win_revalidate_root(root)
        current_handle = root.handle
        for index, part in enumerate(parts[:-1]):
            _safe_open_barrier(label, relative, index)
            handle, information = _win_open_relative_checked(
                current_handle, part, directory=True, label=label
            )
            try:
                handle_path = _win_final_path(handle, label)
            except Exception:
                _CloseHandle(handle)
                raise
            if not _win_is_beneath(handle_path, root.final_path):
                _CloseHandle(handle)
                raise ValueError(f"{label} escapes approved root")
            if information.dwVolumeSerialNumber != root.identity[1]:
                _CloseHandle(handle)
                raise ValueError(f"{label} crosses approved-root volume")
            directory_handles.append(handle)
            current_handle = handle
        _safe_open_barrier(label, relative, len(parts) - 1)
        file_handle, before = _win_open_relative_checked(
            current_handle, parts[-1], directory=False, label=label
        )
        if not _win_is_beneath(_win_final_path(file_handle, label), root.final_path):
            raise ValueError(f"{label} escapes approved root")
        if before.dwVolumeSerialNumber != root.identity[1]:
            raise ValueError(f"{label} crosses approved-root volume")
        file_size = (before.nFileSizeHigh << 32) | before.nFileSizeLow
        if file_size > limit:
            raise ValueError(f"{label} exceeds its frozen byte limit")
        payload = _win_read(file_handle, label, limit)
        after = _win_information(file_handle, label)
        if _win_metadata(before) != _win_metadata(after) or len(payload) != file_size:
            raise ValueError(f"{label} changed while being read")
        return payload
    finally:
        if file_handle not in (None, _INVALID_HANDLE_VALUE):
            _CloseHandle(file_handle)
        for handle in reversed(directory_handles):
            _CloseHandle(handle)
def _validate_lexical_filesystem_path(value: str, label: str) -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} lexical path is invalid")
    path_module = ntpath if _NATIVE_WINDOWS else os.path
    if _NATIVE_WINDOWS and value.startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")): raise ValueError(f"{label} uses an unsupported Windows device namespace")
    if _NATIVE_WINDOWS and "/" in value and "\\" in value: raise ValueError(f"{label} lexical path mixes separators")
    drive, tail = path_module.splitdrive(value)
    if drive and not path_module.isabs(value):
        raise ValueError(f"{label} lexical path is drive-relative")
    if _NATIVE_WINDOWS:
        separator = "\\" if "\\" in tail else "/"
    else:
        if "\\" in tail:
            raise ValueError(f"{label} lexical path contains a backslash")
        separator = "/"
    if tail.startswith(separator):
        tail = tail[1:]
    parts = tail.split(separator) if tail else ()
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ValueError(f"{label} lexical components are unsafe")
def _bind_exact_path(value: Any, label: str, *, allow_string: bool) -> str:
    if allow_string and type(value) is str:
        return value
    if type(value) is not type(Path()):
        raise ValueError(f"{label} must be an exact concrete Path" + " or exact string" * allow_string)
    raw_path = os.fspath(value)
    if type(raw_path) is not str:
        raise ValueError(f"{label} filesystem representation must be an exact string")
    return raw_path
def _manifest_location(root: Path, path: Path | str) -> tuple[Path, str]:
    raw_root = _bind_exact_path(root, "approved replay root", allow_string=False)
    raw_path = _bind_exact_path(path, "manifest path", allow_string=True)
    _validate_lexical_filesystem_path(raw_root, "approved replay root")
    _validate_lexical_filesystem_path(raw_path, "manifest path")
    path_module = ntpath if _NATIVE_WINDOWS else os.path
    if not path_module.isabs(raw_root): raise ValueError("approved replay root must be absolute")
    if not path_module.isabs(raw_path): raise ValueError("manifest path must be absolute")
    absolute_root_text = path_module.abspath(raw_root)
    absolute_path_text = path_module.abspath(raw_path)
    if _NATIVE_WINDOWS:
        try:
            common = ntpath.commonpath((absolute_root_text, absolute_path_text))
        except ValueError as exc:
            raise ValueError("manifest path escapes approved root") from exc
        if ntpath.normcase(common) != ntpath.normcase(absolute_root_text):
            raise ValueError("manifest path escapes approved root")
        relative = ntpath.relpath(
            absolute_path_text, absolute_root_text
        ).replace("\\", "/")
    else:
        absolute_path = Path(absolute_path_text)
        try:
            relative = absolute_path.relative_to(Path(absolute_root_text)).as_posix()
        except ValueError as exc:
            raise ValueError("manifest path escapes approved root") from exc
    _relative_parts(relative, "manifest path")
    return Path(absolute_root_text), relative


def _validate_case(value: Any) -> dict[str, Any]:
    fields = {
        "case_id", "opponent", "evaluated_agent_camp", "map_type", "day_time", "repeat"
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("case identity fields are incomplete or extra")
    result = copy.deepcopy(dict(value))
    _strict_text(result["case_id"], "case ID")
    _strict_text(result["opponent"], "opponent")
    for field in ("evaluated_agent_camp", "map_type", "day_time"):
        if not isinstance(result[field], int) or isinstance(result[field], bool) or result[field] not in (0, 1):
            raise ValueError(f"case {field} must be strict integer 0 or 1")
    if not isinstance(result["repeat"], int) or isinstance(result["repeat"], bool) or result["repeat"] not in (1, 2, 3):
        raise ValueError("case repeat must be strict integer 1, 2, or 3")
    return result


def _validate_seeds(value: Any) -> dict[str, int]:
    fields = ("logic_seed", "evaluated_agent_seed", "opponent_seed")
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError("seed bundle is incomplete or extra")
    return {field: _strict_seed(value[field], field) for field in fields}


def _validate_policy(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"version", "source_sha256"}:
        raise ValueError("acting policy identity is incomplete or extra")
    return {
        "version": _strict_text(value["version"], "policy version"),
        "source_sha256": _strict_sha(value["source_sha256"], "policy source"),
    }


def _validate_champion(value: Any) -> dict[str, str]:
    fields = {"logical_id", "version", "descriptor_sha256", "artifact_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("champion identity is incomplete or extra")
    return {
        "logical_id": _strict_text(value["logical_id"], "champion logical ID"),
        "version": _strict_text(value["version"], "champion version"),
        "descriptor_sha256": _strict_sha(value["descriptor_sha256"], "champion descriptor"),
        "artifact_sha256": _strict_sha(value["artifact_sha256"], "champion artifact"),
    }


def _validate_case_champion_binding(
    case: Mapping[str, Any], champion: Mapping[str, str]
) -> None:
    """Bind the declared case opponent to the pinned champion identity."""

    if case["opponent"] != champion["logical_id"]:
        raise ValueError("case opponent does not match champion logical ID")


def _validate_state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"canonical_state_id", "observation"}:
        raise ValueError(f"{label} must contain state identity and observation")
    observation = value["observation"]
    if not isinstance(observation, Mapping):
        raise ValueError(f"{label} observation must be an object")
    if value["canonical_state_id"] != canonical_state_id(observation):
        raise ValueError(f"{label} canonical state identity mismatch")
    return copy.deepcopy(dict(value))


def _validate_evaluated_camp(
    observation: Mapping[str, Any], expected_camp: int, label: str
) -> None:
    camp = observation.get("camp")
    if type(camp) is not int or camp not in (0, 1):
        raise ValueError(f"{label} observation camp must be strict integer 0 or 1")
    if camp != expected_camp:
        raise ValueError(f"{label} observation camp does not match evaluated agent camp")


@dataclass(frozen=True)
class DecisionFrame:
    decision_step: int
    state_before: Mapping[str, Any]
    action_support: Mapping[str, Any]
    chosen_action: Mapping[str, Any]
    acting_identity_refs: Mapping[str, str]
    state_after: Mapping[str, Any]
    reward: float
    outcome: str
    terminated: bool
    truncated: bool
    case_identity_ref: str
    seed_bundle_ref: str
    rationale_status: str = RATIONALE_STATUS


@dataclass(frozen=True)
class ReplayPacket:
    manifest_sha256: str
    replay_sha256: str
    match_plan_sha256: str
    case_identity: Mapping[str, Any]
    role: str
    seeds: Mapping[str, int]
    acting_policy: Mapping[str, str]
    champion: Mapping[str, str]
    decision_frames: tuple[DecisionFrame, ...]
    terminal: Mapping[str, Any]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class ReplayReadingContext:
    """Issuer-only immutable preflight proof; it never stores a ReplayPacket."""

    _approved_root: str
    _approved_root_identity: tuple[Any, ...]
    _manifest_relative_path: str
    _manifest_bytes: bytes
    _manifest_sha256: str

    def __new__(cls):
        raise TypeError("ReplayReadingContext can only be issued by trusted preflight")

    @property
    def manifest(self) -> Mapping[str, Any]:
        return _deep_freeze(_strict_object_bytes(self._manifest_bytes, "context manifest"))


def _identity_refs(policy: Mapping[str, str], champion: Mapping[str, str]) -> dict[str, str]:
    return {
        "acting_policy_version": policy["version"],
        "policy_source_sha256": policy["source_sha256"],
        "champion_logical_id": champion["logical_id"],
        "champion_version": champion["version"],
        "champion_descriptor_sha256": champion["descriptor_sha256"],
        "champion_artifact_sha256": champion["artifact_sha256"],
    }


def _validate_frame(value: Any, expected_step: int, *, case, seeds, policy, champion) -> DecisionFrame:
    fields = {
        "decision_step", "state_before", "action_support", "chosen_action",
        "acting_identity_refs", "state_after", "reward", "outcome", "terminated",
        "truncated", "case_identity_ref", "seed_bundle_ref", "rationale_status",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DecisionFrame fields are incomplete or extra")
    step = value["decision_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step != expected_step:
        raise ValueError("decision_step must be strict integers 1..N")
    before = _validate_state(value["state_before"], "state_before")
    after = _validate_state(value["state_after"], "state_after")
    expected_camp = case["evaluated_agent_camp"]
    _validate_evaluated_camp(
        before["observation"], expected_camp, "state_before"
    )
    _validate_evaluated_camp(
        after["observation"], expected_camp, "state_after"
    )
    support = build_action_support(enumerate_legal_commands(before["observation"]))
    actions = [{"action_id": item.action_id, "command": item.action} for item in support.actions]
    supplied = value["action_support"]
    trusted = {"schema_version": support.schema_version, "support_id": support.support_id, "actions": actions}
    if (
        type(supplied) is not dict
        or set(supplied) != set(trusted)
        or canonical_replay_json_bytes(supplied)
        != canonical_replay_json_bytes(trusted)
    ):
        raise ValueError("DecisionFrame does not contain trusted ActionSupport")
    chosen = value["chosen_action"]
    if not isinstance(chosen, Mapping) or set(chosen) != {"action_id", "command"}:
        raise ValueError("chosen action is incomplete")
    command = canonical_command(chosen["command"])
    by_id = {item["action_id"]: item["command"] for item in actions}
    if command_action_id(command) != chosen["action_id"] or by_id.get(chosen["action_id"]) != command:
        raise ValueError("chosen action is outside trusted ActionSupport")
    if value["acting_identity_refs"] != _identity_refs(policy, champion):
        raise ValueError("acting identity refs mismatch")
    if value["case_identity_ref"] != case["case_id"]:
        raise ValueError("DecisionFrame case identity ref mismatch")
    seed_ref = hashlib.sha256(canonical_replay_json_bytes(dict(seeds))).hexdigest()
    if value["seed_bundle_ref"] != seed_ref:
        raise ValueError("DecisionFrame seed bundle ref mismatch")
    if value["rationale_status"] != RATIONALE_STATUS:
        raise ValueError("rationale must remain not_recorded")
    if not isinstance(value["terminated"], bool) or not isinstance(value["truncated"], bool):
        raise ValueError("terminal flags must be booleans")
    if value["outcome"] not in {"win", "loss", "draw", "ongoing"}:
        raise ValueError("DecisionFrame outcome is invalid")
    return DecisionFrame(
        step, _deep_freeze(before), _deep_freeze(dict(supplied)),
        _deep_freeze(dict(chosen)),
        _deep_freeze(dict(value["acting_identity_refs"])), _deep_freeze(after),
        _strict_number(value["reward"], "DecisionFrame reward"), value["outcome"],
        value["terminated"], value["truncated"], value["case_identity_ref"],
        value["seed_bundle_ref"],
    )


def _validate_replay(value: Mapping[str, Any], manifest: Mapping[str, Any], manifest_sha: str) -> ReplayPacket:
    fields = {
        "schema_version", "match_plan_sha256", "case_identity", "role", "seeds",
        "acting_policy", "champion", "rationale_status", "decision_count",
        "terminal_recorded", "decision_frames", "terminal",
    }
    if set(value) != fields or value.get("schema_version") != SYNTHETIC_REPLAY_SCHEMA_VERSION:
        raise ValueError("synthetic replay schema or fields mismatch")
    case = _validate_case(value["case_identity"])
    seeds = _validate_seeds(value["seeds"])
    policy = _validate_policy(value["acting_policy"])
    champion = _validate_champion(value["champion"])
    _validate_case_champion_binding(case, champion)
    if any((
        value["match_plan_sha256"] != manifest["match_plan_sha256"],
        case != manifest["case_identity"], value["role"] != manifest["role"],
        seeds != manifest["seeds"], policy != manifest["acting_policy"],
        champion != manifest["champion"],
    )):
        raise ValueError("replay and approved manifest identity mismatch")
    frames = value["decision_frames"]
    count = value["decision_count"]
    if not isinstance(frames, list) or not frames or not isinstance(count, int) or isinstance(count, bool) or count != len(frames):
        raise ValueError("synthetic replay decision count mismatch")
    if value["terminal_recorded"] is not True or value["rationale_status"] != RATIONALE_STATUS:
        raise ValueError("synthetic replay terminal/rationale marker is invalid")
    parsed = tuple(
        _validate_frame(frame, index, case=case, seeds=seeds, policy=policy, champion=champion)
        for index, frame in enumerate(frames, start=1)
    )
    for left, right in zip(parsed, parsed[1:]):
        if left.state_after != right.state_before:
            raise ValueError("DecisionFrame state chain is not continuous")
        if left.terminated or left.truncated or left.outcome != "ongoing":
            raise ValueError("only the last DecisionFrame may be terminal")
    terminal = value["terminal"]
    terminal_fields = {"outcome", "reward", "termination_reason", "terminated", "truncated"}
    if not isinstance(terminal, Mapping) or set(terminal) != terminal_fields:
        raise ValueError("terminal record is incomplete or extra")
    terminal_reward = _strict_number(terminal["reward"], "terminal reward")
    termination_reason = _strict_text(
        terminal["termination_reason"], "termination reason"
    )
    if terminal["outcome"] not in {"win", "loss", "draw"} or not isinstance(terminal["terminated"], bool) or not isinstance(terminal["truncated"], bool):
        raise ValueError("terminal record is invalid")
    last = parsed[-1]
    if not (last.terminated or last.truncated) or (last.outcome, last.reward, last.terminated, last.truncated) != (terminal["outcome"], terminal_reward, terminal["terminated"], terminal["truncated"]):
        raise ValueError("terminal record disagrees with the last DecisionFrame")
    normalized_terminal = {
        "outcome": terminal["outcome"],
        "reward": terminal_reward,
        "termination_reason": termination_reason,
        "terminated": terminal["terminated"],
        "truncated": terminal["truncated"],
    }
    return ReplayPacket(
        manifest_sha, manifest["replay_sha256"], manifest["match_plan_sha256"],
        _deep_freeze(case), "train", _deep_freeze(seeds), _deep_freeze(policy),
        _deep_freeze(champion), parsed, _deep_freeze(normalized_terminal),
    )


def _read_and_validate(
    root: Path,
    manifest_relative: str,
    expected_bytes: bytes | None = None,
    expected_root_identity: tuple[Any, ...] | None = None,
) -> tuple[ReplayPacket, bytes, str, tuple[Any, ...]]:
    with _open_approved_root(root, expected_root_identity) as approved:
        manifest_bytes = approved.read(
            manifest_relative, "manifest path", MAX_REPLAY_MANIFEST_BYTES
        )
        if expected_bytes is not None and manifest_bytes != expected_bytes:
            raise ValueError("manifest bytes changed after preflight")
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        if digest not in APPROVED_TRAINING_REPLAY_MANIFESTS:
            raise ValueError("replay manifest is not independently approved")
        manifest = _strict_object_bytes(manifest_bytes, "replay manifest")
        fields = {
            "schema_version", "replay_artifact_path", "replay_sha256", "match_plan_sha256",
            "case_identity", "role", "seeds", "acting_policy", "champion",
        }
        if set(manifest) != fields or manifest.get("schema_version") != REPLAY_READING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("replay manifest schema or fields mismatch")
        manifest["replay_sha256"] = _strict_sha(manifest["replay_sha256"], "replay")
        manifest["match_plan_sha256"] = _strict_sha(manifest["match_plan_sha256"], "match plan")
        manifest["case_identity"] = _validate_case(manifest["case_identity"])
        manifest["seeds"] = _validate_seeds(manifest["seeds"])
        manifest["acting_policy"] = _validate_policy(manifest["acting_policy"])
        manifest["champion"] = _validate_champion(manifest["champion"])
        _validate_case_champion_binding(
            manifest["case_identity"], manifest["champion"]
        )
        if manifest["role"] != "train":
            raise ValueError("Replay Reading accepts only independently approved role=train")
        replay_bytes = approved.read(
            manifest["replay_artifact_path"],
            "replay path",
            MAX_SYNTHETIC_REPLAY_BYTES,
        )
        if hashlib.sha256(replay_bytes).hexdigest() != manifest["replay_sha256"]:
            raise ValueError("replay artifact SHA mismatch")
        document = _strict_object_bytes(replay_bytes, "synthetic replay")
        packet = _validate_replay(document, manifest, digest)
        return packet, manifest_bytes, digest, approved.identity
def _context_snapshot(context: ReplayReadingContext) -> tuple[Any, ...]:
    if type(context._approved_root) is not str:
        raise ValueError("approved replay root context field is invalid")
    identity = _strict_root_identity(context._approved_root_identity)
    relative = "/".join(
        _relative_parts(context._manifest_relative_path, "context manifest path")
    )
    if type(context._manifest_bytes) is not bytes:
        raise ValueError("context manifest bytes are invalid")
    digest = _strict_sha(context._manifest_sha256, "context manifest")
    return (
        context._approved_root,
        identity,
        relative,
        context._manifest_bytes,
        digest,
    )


def _build_context_authority():
    registry: weakref.WeakKeyDictionary[ReplayReadingContext, tuple[Any, ...]] = (
        weakref.WeakKeyDictionary()
    )
    def preflight(
        manifest_path: Path | str, *, approved_root: Path
    ) -> ReplayReadingContext:
        root, relative = _manifest_location(approved_root, manifest_path)
        packet, payload, digest, root_identity = _read_and_validate(root, relative)
        if packet.role != "train":
            raise ValueError("replay manifest role changed during preflight")
        context = object.__new__(ReplayReadingContext)
        object.__setattr__(
            context,
            "_approved_root",
            (ntpath if _NATIVE_WINDOWS else os.path).abspath(os.fspath(root)),
        )
        object.__setattr__(
            context, "_approved_root_identity", _strict_root_identity(root_identity)
        )
        object.__setattr__(context, "_manifest_relative_path", relative)
        object.__setattr__(context, "_manifest_bytes", bytes(payload))
        object.__setattr__(context, "_manifest_sha256", digest)
        registry[context] = _context_snapshot(context)
        return context
    def is_issued(context: Any) -> bool:
        if type(context) is not ReplayReadingContext:
            return False
        try:
            registered = registry.get(context)
            return registered is not None and _context_snapshot(context) == registered
        except (AttributeError, TypeError, ValueError):
            return False
    return preflight, is_issued
preflight_replay_reading, _is_issued_context = _build_context_authority()
del _build_context_authority
def open_replay_reading(context: ReplayReadingContext) -> ReplayPacket:
    if not _is_issued_context(context):
        raise ValueError("trusted issued Replay Reading context is required")
    _strict_object_bytes(context._manifest_bytes, "context manifest")
    if context._manifest_sha256 not in APPROVED_TRAINING_REPLAY_MANIFESTS:
        raise ValueError("context manifest digest is no longer approved")
    root = Path(context._approved_root)
    packet, payload, digest, root_identity = _read_and_validate(
        root,
        context._manifest_relative_path,
        context._manifest_bytes,
        context._approved_root_identity,
    )
    if (
        not _root_identities_match(root_identity, context._approved_root_identity)
        or payload != context._manifest_bytes
        or digest != context._manifest_sha256
        or packet.role != "train"
    ):
        raise ValueError("context manifest bytes, digest, or role changed")
    return packet


def render_replay_timeline(context: ReplayReadingContext) -> str:
    """Reopen issuer-created context and render only the revalidated packet."""

    if type(context) is not ReplayReadingContext:
        raise TypeError("renderer requires an issued Replay Reading context")
    packet = open_replay_reading(context)
    case = packet.case_identity
    seeds = packet.seeds

    def encoded_text(value: Any, label: str) -> str:
        return quote(_strict_text(value, label), safe="-._~")

    def encoded_command(value: Mapping[str, Any]) -> str:
        canonical = canonical_replay_json_bytes(value).decode("utf-8")
        return quote(canonical, safe="-._~")

    lines = [
        " | ".join((
            f"case={encoded_text(case['case_id'], 'case ID')}",
            f"role={encoded_text(packet.role, 'role')}",
            f"logic_seed={seeds['logic_seed']}",
            f"evaluated_agent_seed={seeds['evaluated_agent_seed']}",
            f"opponent_seed={seeds['opponent_seed']}",
        ))
    ]
    for frame in packet.decision_frames:
        command = encoded_command(frame.chosen_action["command"])
        lines.append(" | ".join((
            f"step={frame.decision_step}",
            f"state={encoded_text(frame.state_before['canonical_state_id'], 'state ID')}",
            f"legal_actions={len(frame.action_support['actions'])}",
            f"chosen={encoded_text(frame.chosen_action['action_id'], 'action ID')}:{command}",
            "policy="
            f"{encoded_text(packet.acting_policy['version'], 'policy version')}@"
            f"{encoded_text(packet.acting_policy['source_sha256'], 'policy source')}",
            "champion="
            f"{encoded_text(packet.champion['logical_id'], 'champion logical ID')}@"
            f"{encoded_text(packet.champion['version'], 'champion version')}",
            f"reward={frame.reward}",
            f"outcome={encoded_text(frame.outcome, 'frame outcome')}",
            "rationale_status="
            f"{encoded_text(frame.rationale_status, 'rationale status')}",
        )))
    terminal = packet.terminal
    lines.append(" | ".join((
        "terminal",
        f"outcome={encoded_text(terminal['outcome'], 'terminal outcome')}",
        f"reward={terminal['reward']}",
        f"terminated={str(terminal['terminated']).lower()}",
        f"truncated={str(terminal['truncated']).lower()}",
        "termination_reason="
        f"{encoded_text(terminal['termination_reason'], 'termination reason')}",
    )))
    return "\n".join(lines) + "\n"


__all__ = [
    "APPROVED_TRAINING_REPLAY_MANIFESTS",
    "MAX_REPLAY_MANIFEST_BYTES",
    "MAX_SYNTHETIC_REPLAY_BYTES",
    "RATIONALE_STATUS",
    "REPLAY_READING_MANIFEST_SCHEMA_VERSION",
    "SYNTHETIC_REPLAY_SCHEMA_VERSION",
    "DecisionFrame",
    "ReplayPacket",
    "ReplayReadingContext",
    "canonical_replay_json_bytes",
    "preflight_replay_reading",
    "open_replay_reading",
    "render_replay_timeline",
]
