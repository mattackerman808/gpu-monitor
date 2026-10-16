"""The desktop widget UI: a small, frameless, always-on-top overlay."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QProgressBar,
    QVBoxLayout,
)

from gpu_reader import GPUInfo
from ollama_reader import OllamaGenStats, OllamaModel

POS_FILE = Path.home() / ".gpu-monitor.json"

WIDTH = 300
RADIUS = 12
PADDING = 12

BG = QColor(18, 20, 26, 225)
BORDER = QColor(255, 255, 255, 28)
TEXT = "#e5e7eb"
MUTED = "#8b93a1"
IDLE = "#565d6b"
ERROR = "#f87171"

GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"


def _util_color(pct: int) -> str:
    if pct < 60:
        return GREEN
    if pct < 85:
        return AMBER
    return RED


def _bar_style(color: str) -> str:
    return f"""
        QProgressBar {{
            border: none;
            background: #262a33;
            border-radius: 4px;
            max-height: 8px;
            min-height: 8px;
        }}
        QProgressBar::chunk {{
            border-radius: 4px;
            background: {color};
        }}
    """


def _split_style() -> str:
    """Two-tone bar: green chunk = share in GPU VRAM, slate base = CPU share."""
    return f"""
        QProgressBar {{
            border: none;
            background: #4a5568;
            border-radius: 4px;
            max-height: 8px;
            min-height: 8px;
        }}
        QProgressBar::chunk {{
            border-radius: 4px;
            background: {GREEN};
        }}
    """


def _fmt_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _fmt_split(gpu_fraction: float) -> str:
    pct = int(round(gpu_fraction * 100))
    if pct >= 99:
        return "100% GPU"
    if pct <= 1:
        return "100% CPU"
    return f"{pct}% GPU \u00b7 {100 - pct}% CPU"


class _GPURow(QFrame):
    """One compact row per GPU: name, utilization bar, stats line."""

    def __init__(self, parent: QFrame | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(52)

        self.name_label = QLabel("Loading\u2026")
        self.name_label.setStyleSheet(f"color: {MUTED}; font-weight: 600; font-size: 12px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(_bar_style(IDLE))

        self.stats_label = QLabel("\u2014")
        self.stats_label.setStyleSheet(f"color: {IDLE}; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.name_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.stats_label)

    def update_gpu(self, gpu: GPUInfo) -> None:
        color = _util_color(gpu.utilization)

        self.name_label.setText(gpu.name)
        self.name_label.setStyleSheet(
            f"color: {TEXT}; font-weight: 600; font-size: 12px;"
        )
        self.bar.setValue(gpu.utilization)
        self.bar.setStyleSheet(_bar_style(color))
        self.stats_label.setText(
            f"{gpu.utilization}%   "
            f"{gpu.mem_used_gb:.1f}/{gpu.mem_total_gb:.1f} GiB   "
            f"{gpu.temp_c}\u00b0C   {gpu.power_w:.0f} W"
        )
        self.stats_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")


class _OllamaRow(QFrame):
    """One compact row per loaded Ollama model, mirroring the GPU row layout:
    name, GPU/CPU split bar, stats line."""

    def __init__(self, parent: QFrame | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(52)

        self.name_label = QLabel()
        self.name_label.setStyleSheet(f"color: {TEXT}; font-weight: 600; font-size: 12px;")

        self.split_bar = QProgressBar()
        self.split_bar.setRange(0, 100)
        self.split_bar.setTextVisible(False)
        self.split_bar.setStyleSheet(_split_style())

        self.info_label = QLabel()
        self.info_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.name_label)
        layout.addWidget(self.split_bar)
        layout.addWidget(self.info_label)

    def update_model(self, model: OllamaModel, now: float) -> None:
        self.name_label.setText(model.name)
        self.split_bar.setValue(int(round(model.gpu_fraction * 100)))
        bits = [f"{model.size_bytes / 2**30:.1f} GiB", _fmt_split(model.gpu_fraction)]
        if math.isinf(model.expires_at):
            bits.append("until never")
        elif model.expires_at > 0:
            bits.append(f"until {_fmt_mmss(model.expires_at - now)}")
        self.info_label.setText(" \u00b7 ".join(bits))


class GPUWidget(QFrame):
    """Always-on-top draggable widget hosting one row per GPU."""

    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(WIDTH)
        self.setMouseTracking(True)

        self.rows: list[_GPURow] = []
        self.ollama_rows: list[_OllamaRow] = []
        self._drag_offset: QPointF | None = None

        self.header = QLabel("GPU MONITOR")
        self.header.setStyleSheet(f"color: {MUTED}; font-size: 10px; letter-spacing: 2px;")
        self.ollama_header = QLabel("OLLAMA")
        self.ollama_header.setStyleSheet(f"color: {MUTED}; font-size: 10px; letter-spacing: 2px;")
        self.ollama_status = QLabel()
        self.ollama_status.setStyleSheet(f"color: {IDLE}; font-size: 11px;")
        self.gen_label = QLabel()
        self.gen_label.setWordWrap(True)
        self.gen_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.gen_label.hide()
        self._gen_stats: OllamaGenStats | None = None
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {ERROR}; font-size: 11px;")
        self.error_label.hide()

        self.gpu_layout = QVBoxLayout()
        self.gpu_layout.setSpacing(10)
        self.ollama_layout = QVBoxLayout()
        self.ollama_layout.setSpacing(10)

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        self.body.addWidget(self.header)
        self.body.addLayout(self.gpu_layout)
        self.body.addWidget(self.ollama_header)
        self.body.addLayout(self.ollama_layout)
        self.body.addWidget(self.gen_label)
        self.body.addWidget(self.ollama_status)
        self.body.addWidget(self.error_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        outer.addLayout(self.body)

        self._load_pos()

    # -- data updates -------------------------------------------------------

    def set_data(self, gpus: list[GPUInfo]) -> None:
        self.error_label.hide()
        for gpu in gpus:
            row = self._ensure_row(gpu.index)
            row.update_gpu(gpu)
        self._refresh_gen_line()
        self._fit_height()

    def set_error(self, message: str) -> None:
        for row in self.rows:
            row.setVisible(False)
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self._fit_height()

    def set_ollama(self, models: list[OllamaModel] | None) -> None:
        """Update the Ollama section. None = server unreachable."""
        now = time.time()
        if models is None:
            for row in self.ollama_rows:
                row.setVisible(False)
            self.ollama_status.setText("server offline")
            self.ollama_status.setVisible(True)
            self._fit_height()
            return

        if not models:
            for row in self.ollama_rows:
                row.setVisible(False)
            self.ollama_status.setText("no models loaded")
            self.ollama_status.setVisible(True)
            self._fit_height()
            return

        self.ollama_status.setVisible(False)
        for i, model in enumerate(models):
            row = self._ensure_ollama_row(i)
            row.update_model(model, now)
        self._fit_height()

    def _ensure_ollama_row(self, index: int) -> _OllamaRow:
        while index >= len(self.ollama_rows):
            row = _OllamaRow(self)
            self.ollama_layout.addWidget(row)
            self.ollama_rows.append(row)
        self.ollama_rows[index].setVisible(True)
        return self.ollama_rows[index]

    def set_gen_stats(self, stats: OllamaGenStats | None) -> None:
        """Record the latest completed generation (from the Ollama log)."""
        if stats is not None:
            self._gen_stats = stats
        self._refresh_gen_line()
        self._fit_height()

    def _refresh_gen_line(self) -> None:
        """Keep the 'last gen' age fresh; called on every data tick."""
        if self._gen_stats is None:
            if self.gen_label.isVisible():
                self.gen_label.hide()
            return
        s = self._gen_stats
        age = max(0, int(time.time() - s.observed_at))
        if age < 60:
            ago = f"{age}s ago"
        elif age < 3600:
            ago = f"{age // 60}m {age % 60}s ago"
        else:
            ago = f"{age // 3600}h {age % 3600 // 60}m ago"
        self.gen_label.setTextFormat(Qt.TextFormat.RichText)
        self.gen_label.setText(
            f'<span style="color: {GREEN}; font-size: 13px; font-weight: 600;">'
            f'{s.tok_per_s:.1f} tok/s</span>'
            f'<span style="color: {TEXT}; font-size: 11px;">'
            f' \u00b7 {s.tokens} tokens \u00b7 {ago}</span>'
        )
        if not self.gen_label.isVisible():
            self.gen_label.show()

    def _ensure_row(self, index: int) -> _GPURow:
        while index >= len(self.rows):
            row = _GPURow(self)
            self.gpu_layout.addWidget(row)
            self.rows.append(row)
        self.rows[index].setVisible(True)
        return self.rows[index]

    def _fit_height(self) -> None:
        self.adjustSize()
        self.setFixedHeight(self.sizeHint().height())

    # -- painting (rounded translucent card) --------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().toRectF(), RADIUS, RADIUS)
        painter.fillPath(path, QBrush(BG))
        painter.setPen(QPen(BORDER, 1))
        painter.drawPath(path)

    # -- dragging ------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self._save_pos()
            event.accept()

    # -- right-click menu ------------------------------------------------------

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1c1f27; color: "
            + TEXT
            + "; border: 1px solid #333844; }"
            "QMenu::item:selected { background: #2c3140; }"
        )
        quit_action = menu.addAction("Quit")
        chosen = menu.exec(event.globalPos())
        if chosen == quit_action:
            self.finished.emit()
            self.close()

    # -- position persistence ---------------------------------------------------

    def _load_pos(self) -> None:
        pos: tuple[int, int] | None = None
        try:
            data = json.loads(POS_FILE.read_text(encoding="utf-8"))
            pos = (int(data["x"]), int(data["y"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass

        if pos is None:
            screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
            if screen is None:
                self.move(0, 0)
                return
            geo = screen.availableGeometry()
            pos = (geo.right() - WIDTH - 16, geo.top() + 16)

        # Keep the widget on a visible screen in case monitors changed.
        on_screen = any(
            screen.availableGeometry().adjusted(0, 0, -WIDTH, -80).contains(*pos)
            for screen in QApplication.screens()
        )
        x, y = pos if on_screen else (0, 0)
        self.move(x, y)

    def _save_pos(self) -> None:
        try:
            POS_FILE.write_text(
                json.dumps({"x": self.x(), "y": self.y()}), encoding="utf-8"
            )
        except OSError:
            pass

