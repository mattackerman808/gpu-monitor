"""GPU data reader: polls nvidia-smi and returns parsed per-GPU stats."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)

_QUERY = ",".join(NVIDIA_SMI_FIELDS)
_QUERY_CMD = ["--query-gpu", _QUERY, "--format=csv,noheader,nounits"]


@dataclass
class GPUInfo:
    index: int
    name: str
    utilization: int  # percent 0-100
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int
    power_w: float

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used_mib / 1024.0

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total_mib / 1024.0


def find_nvidia_smi() -> str | None:
    """Locate the nvidia-smi executable, or None if not installed."""
    return shutil.which("nvidia-smi")


def read_gpus(smi_path: str | None = None) -> list[GPUInfo]:
    """Poll nvidia-smi once and return a GPUInfo per GPU.

    Raises RuntimeError if the query fails or the output is unparseable.
    """
    if smi_path is None:
        smi_path = find_nvidia_smi()
    if smi_path is None:
        raise RuntimeError("nvidia-smi not found; are NVIDIA drivers installed?")

    # Keep nvidia-smi from popping a console window when we run as a
    # console-less (windowed) app, e.g. the --noconsole PyInstaller exe.
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [smi_path, *_QUERY_CMD],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=creation_flags,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"nvidia-smi query failed: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()}")

    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(NVIDIA_SMI_FIELDS):
            continue  # skip malformed lines
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    utilization=int(parts[2]),
                    mem_used_mib=int(parts[3]),
                    mem_total_mib=int(parts[4]),
                    temp_c=int(parts[5]),
                    power_w=float(parts[6]),
                )
            )
        except ValueError:
            continue

    if not gpus:
        raise RuntimeError(f"no GPU data in nvidia-smi output: {proc.stdout.strip()!r}")
    return gpus


def _main() -> None:
    """Standalone smoke test: print one poll to the console."""
    gpus = read_gpus()
    for g in gpus:
        print(
            f"[GPU {g.index}] {g.name}: {g.utilization}% | "
            f"{g.mem_used_gb:.1f}/{g.mem_total_gb:.1f} GiB | "
            f"{g.temp_c}\u00b0C | {g.power_w:.1f} W"
        )


if __name__ == "__main__":
    _main()
