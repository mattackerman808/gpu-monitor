"""Entry point: starts the polling worker and shows the GPU widget."""

from __future__ import annotations

import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

import gpu_reader
import ollama_reader
from ollama_reader import OllamaLogTail, find_server_log
from widget import GPUWidget

POLL_INTERVAL_SEC = 1.5


class GPUWorker(QThread):
    """Polls nvidia-smi in the background and emits parsed GPU stats."""

    data_ready = pyqtSignal(list)
    read_failed = pyqtSignal(str)
    ollama_ready = pyqtSignal(object)  # list[OllamaModel] or None (offline)
    ollama_gen = pyqtSignal(object)  # OllamaGenStats or None (no new generation)

    def __init__(self) -> None:
        super().__init__()
        self._running = True
        self._smi = gpu_reader.find_nvidia_smi()
        self._log_tail = OllamaLogTail(find_server_log())

    def run(self) -> None:  # noqa: D102 (QThread API)
        while self._running:
            try:
                gpus = gpu_reader.read_gpus(self._smi)
                self.data_ready.emit(list(gpus))
            except RuntimeError as exc:
                self.read_failed.emit(str(exc))
            try:
                self.ollama_ready.emit(ollama_reader.read_ollama())
            except Exception:  # server down, timeout, bad JSON -> "offline"
                self.ollama_ready.emit(None)
            self.ollama_gen.emit(self._log_tail.poll())
            self.msleep(int(POLL_INTERVAL_SEC * 1000))

    def stop(self) -> None:
        self._running = False
        self.wait(3000)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("GPU Monitor")
    app.setQuitOnLastWindowClosed(True)

    widget = GPUWidget()
    worker = GPUWorker()
    worker.data_ready.connect(widget.set_data)
    worker.read_failed.connect(widget.set_error)
    worker.ollama_ready.connect(widget.set_ollama)
    worker.ollama_gen.connect(widget.set_gen_stats)
    widget.finished.connect(worker.stop)

    widget.show()
    worker.start()

    rc = app.exec()
    worker.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
