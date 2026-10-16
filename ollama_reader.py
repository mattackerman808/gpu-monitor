"""Ollama stats reader: polls the Ollama HTTP API for loaded models."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OllamaModel:
    name: str
    size_bytes: int  # total size resident in memory
    size_vram_bytes: int  # portion placed on the GPU(s)
    expires_at: float  # epoch seconds (0 = unknown)

    @property
    def gpu_fraction(self) -> float:
        """Share of the model resident in GPU VRAM (0.0 - 1.0)."""
        if self.size_bytes <= 0:
            return 0.0
        return max(0.0, min(1.0, self.size_vram_bytes / self.size_bytes))


def _default_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in host:
        host = "http://" + host
    # OLLAMA_HOST may be set to 0.0.0.0 / :: so the *server* binds to the
    # whole network, but that wildcard can't be *connected to* (Windows
    # raises WinError 10049 "address not valid in its context"). For a
    # local connection, use the loopback instead.
    for wildcard in ("0.0.0.0", "[::]", "::"):
        prefix = "http://" + wildcard
        if host.startswith(prefix):
            host = host.replace(wildcard, "127.0.0.1", 1)
            break
    return host.rstrip("/")


def _parse_expires(value: str) -> float:
    """Parse an RFC3339 timestamp (may have nanoseconds and/or a UTC offset)."""
    try:
        v = value.strip()
        if v.endswith(("Z", "z")):
            v = v[:-1] + "+00:00"
        date_part, _, time_part = v.partition("T")
        if "." in time_part:
            whole, frac = time_part.split(".", 1)
            m = re.match(r"(\d+)(.*)$", frac)
            if m:
                digits, suffix = m.group(1), m.group(2)
                time_part = whole + "." + (digits + "000000")[:6] + suffix
        return datetime.fromisoformat(date_part + "T" + time_part).timestamp()
    except ValueError:
        return 0.0


def read_ollama(host: str | None = None, timeout: float = 1.0) -> list[OllamaModel]:
    """Return models currently loaded in Ollama (GET /api/ps).

    Raises RuntimeError (or urllib's errors) if the server is unreachable
    or the response is unparseable.
    """
    base = host or _default_host()
    req = urllib.request.Request(base + "/api/ps")
    # Bypass system/VPN proxies: Ollama is a local service, and a stale
    # proxy (e.g. from a VPN that isn't connected) makes urllib try to
    # route 127.0.0.1 through a dead address.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        payload = json.load(resp)

    models: list[OllamaModel] = []
    for entry in payload.get("models", []):
        if not isinstance(entry, dict):
            continue
        size = int(entry.get("size", 0))
        models.append(
            OllamaModel(
                name=str(entry.get("name", "unknown")),
                size_bytes=size,
                size_vram_bytes=int(entry.get("size_vram", size)),
                expires_at=_parse_expires(str(entry.get("expires_at", ""))),
            )
        )
    return models


_PROMPT_RE = re.compile(
    r"slot print_timing: id \s*(\d+) \| task (\d+) \| "
    r"prompt eval time =\s+[\d.]+ ms /\s*(\d+) tokens "
    r"\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)"
)
_EVAL_RE = re.compile(
    r"slot print_timing: id \s*(\d+) \| task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s*(\d+) tokens "
    r"\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)"
)


def find_server_log() -> str | None:
    """Locate the Ollama server log (OLLAMA_LOG_FILE overrides)."""
    env = os.environ.get("OLLAMA_LOG_FILE")
    if env:
        return env
    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(os.path.join(local, "Ollama", "server.log"))
    candidates.append(os.path.join(os.path.expanduser("~"), ".ollama", "server.log"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


@dataclass
class OllamaGenStats:
    tokens: int
    tok_per_s: float
    prompt_tokens: int = 0
    prompt_tok_per_s: float = 0.0
    observed_at: float = 0.0


class OllamaLogTail:
    """Follows the Ollama server log and reports completed generations.

    Each poll() returns an OllamaGenStats when a new 'eval time'
    print_timing line appears in the log, else None.
    """

    _INIT_SCAN_BYTES = 2 * 1024 * 1024

    def __init__(self, log_path: str | None) -> None:
        self._path = log_path
        self._offset = 0
        self._buffer = ""
        self._pending_prompt: dict[int, tuple[int, float]] = {}
        self._last_seen: OllamaGenStats | None = None
        if log_path and os.path.isfile(log_path):
            self._init_scan()

    @property
    def last_seen(self) -> OllamaGenStats | None:
        return self._last_seen

    def _init_scan(self) -> None:
        """Seed with the most recent generation already in the log."""
        try:
            size = os.path.getsize(self._path)
            with open(self._path, "rb") as fh:
                fh.seek(max(0, size - self._INIT_SCAN_BYTES))
                chunk = fh.read().decode("utf-8", errors="replace")
            self._offset = size
            for line in chunk.splitlines():
                self._consume(line)
        except OSError:
            pass

    def _consume(self, line: str) -> None:
        m = _PROMPT_RE.search(line)
        if m:
            self._pending_prompt[int(m.group(1))] = (
                int(m.group(3)),
                float(m.group(4)),
            )
            return
        m = _EVAL_RE.search(line)
        if m:
            prompt_tokens, prompt_rate = self._pending_prompt.pop(int(m.group(1)), (0, 0.0))
            self._last_seen = OllamaGenStats(
                tokens=int(m.group(3)),
                tok_per_s=float(m.group(4)),
                prompt_tokens=prompt_tokens,
                prompt_tok_per_s=prompt_rate,
                observed_at=time.time(),
            )

    def poll(self) -> OllamaGenStats | None:
        """Read new log content; return stats if a generation just completed."""
        if not self._path:
            return None
        try:
            size = os.path.getsize(self._path)
        except OSError:
            return None
        if size < self._offset:  # log rotated or truncated
            self._offset = 0
            self._buffer = ""
        if size == self._offset:
            return None
        try:
            with open(self._path, "rb") as fh:
                fh.seek(self._offset)
                delta = fh.read()
        except OSError:
            return None
        self._offset = size
        lines = (self._buffer + delta.decode("utf-8", errors="replace")).split("\n")
        self._buffer = lines.pop()  # keep the possibly-partial last line
        completed = False
        for line in lines:
            before = self._last_seen
            self._consume(line)
            if self._last_seen is not before:
                completed = True
        return self._last_seen if completed else None


def _main() -> None:
    """Standalone smoke test: print one poll to the console."""
    models = read_ollama()
    if not models:
        print("ollama reachable, no models loaded")
    for m in models:
        print(f"{m.name}: {m.size_bytes / 2**30:.2f} GiB | "
              f"{int(round(m.gpu_fraction * 100))}% GPU | until {m.expires_at:.0f}")
    log_path = find_server_log()
    if not log_path:
        print("no ollama server log found")
        return
    tail = OllamaLogTail(log_path)
    last = tail.last_seen
    if last is None:
        print(f"log at {log_path}: no generation timings found yet")
    else:
        print(f"log at {log_path}: last gen {last.tok_per_s:.2f} tok/s, "
              f"{last.tokens} tokens (prompt {last.prompt_tok_per_s:.2f} tok/s)")


if __name__ == "__main__":
    _main()
