# GPU Monitor

A small, always-on-top Windows desktop widget that shows live per-GPU stats
while you run AI tasks. Polls `nvidia-smi` every 1.5 s and displays, for each
NVIDIA GPU:

- GPU name
- Utilization bar (green → amber → red) + percentage
- VRAM used / total
- Temperature and power draw

Idle GPUs are dimmed so the busy one stands out.

It also polls the local Ollama server (`GET /api/ps`, default
`http://127.0.0.1:11434`, overridable with the `OLLAMA_HOST` env var) and
shows each loaded model with its in-memory size and unload countdown.

Additionally it tails the Ollama server log
(`%LOCALAPPDATA%\Ollama\server.log` on Windows, override with
`OLLAMA_LOG_FILE`) and reports the throughput of the last completed
generation (tok/s + token count). Ollama doesn't expose a live generation
rate via its API, so this updates when a run finishes, not while it runs.

## Requirements

- Windows 10/11 with NVIDIA drivers (`nvidia-smi` on PATH)
- Python 3.10+

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python main.py
```

## Usage

- **Drag** anywhere on the widget to move it (position is remembered in
  `~/.gpu-monitor.json`).
- **Right-click → Quit** to close it.
- Standalone smoke test of the data layer (prints one poll):

  ```powershell
  .venv\Scripts\python gpu_reader.py
  ```

## Build a standalone exe

No Python needed on the target machine after building:

```powershell
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --onefile --noconsole --name GpuMonitor main.py
```

Produces `dist\GpuMonitor.exe` (~35 MB) — a single self-contained file you can
copy anywhere (e.g. `%LOCALAPPDATA%\Programs`). First launch is slightly
slower because the onefile bundle unpacks itself to a temp folder.

## Start at login (optional)

1. `Win+R` → `shell:startup`
2. Create a shortcut there pointing to the built exe, e.g.

   ```
   C:\Users\matt\source\gpu-monitor\dist\GpuMonitor.exe
   ```

   (Or, without building, point it at
   `C:\Users\matt\source\gpu-monitor\.venv\Scripts\pythonw.exe` with argument
   `C:\Users\matt\source\gpu-monitor\main.py` — `pythonw.exe` starts it
   without a console window.)

## Project layout

| File               | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `main.py`          | Entry point; background poll thread, app wiring    |
| `widget.py`        | Frameless always-on-top overlay UI (PyQt6)         |
| `gpu_reader.py`    | `nvidia-smi` polling and CSV parsing, `GPUInfo`    |
| `ollama_reader.py` | Ollama `/api/ps` polling, `OllamaModel`; server-log tailing for last-gen tok/s, `OllamaLogTail` |
