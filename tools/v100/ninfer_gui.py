#!/usr/bin/env python3
"""PyQt control panel for the NInfer HTTP server.

The GUI only owns the child ``ninfer-serve`` process.  It does not alter CUDA, driver,
IOMMU, or desktop services.  All values are forwarded as explicit command-line options,
and the generated command is shown before/while the server is running.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from .gui_cache import CacheMonitor
    from .launcher_sampling import validate_sampling
    from .launcher_configs import ConfigurationLibrary
    from .launcher_model import ModelControl
    from .launcher_resources import GIB, LoadStage, MemoryOptions, estimate_memory, query_gpu_memory
else:
    from gui_cache import CacheMonitor
    from launcher_sampling import validate_sampling
    from launcher_configs import ConfigurationLibrary
    from launcher_model import ModelControl
    from launcher_resources import GIB, LoadStage, MemoryOptions, estimate_memory, query_gpu_memory


def _help() -> None:
    print(
        "usage: ninfer_gui.py [--repo PATH] [--binary PATH] [--artifact PATH]\n"
        "\n"
        "Open a PyQt control panel for ninfer-serve. The GUI starts a local child process;\n"
        "it does not configure remote hosts or system services.\n"
        "\n"
        "Environment overrides: NINFER_GUI_REPO, NINFER_GUI_BINARY, NINFER_GUI_ARTIFACT."
    )


try:  # PyQt6 is preferred; PyQt5 is a supported fallback for customer images.
    from PyQt6.QtCore import QProcess, QSettings, QTimer, QUrl, Qt
    from PyQt6.QtGui import QColor, QDesktopServices, QFontDatabase, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QInputDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    QT6 = True
except ImportError:  # selected by the customer's installed Qt binding.
    from PyQt5.QtCore import QProcess, QSettings, QTimer, QUrl, Qt
    from PyQt5.QtGui import QColor, QDesktopServices, QFontDatabase, QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QInputDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    QT6 = False


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


class NInferGui(QWidget):
    def __init__(self, repo: Path, binary: str = "", artifact: str = "", *,
                 settings: QSettings | None = None) -> None:
        super().__init__()
        self.repo = repo
        self.binary_override = binary or os.environ.get("NINFER_GUI_BINARY", "")
        self.artifact_override = artifact or os.environ.get("NINFER_GUI_ARTIFACT", "")
        self.process: QProcess | None = None
        self.health_in_flight = False
        self.model_control: ModelControl | None = None
        self.model_view_state = 'unknown'
        self.health_results: queue.Queue[tuple[str, object, bool, str]] = queue.Queue()
        self.settings = settings or QSettings("NInfer", "V100X2Server")
        self.output_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.request_log_directory = tempfile.TemporaryDirectory(prefix="ninfer-gui-")
        self.cache_monitor = CacheMonitor(Path(self.request_log_directory.name) / "requests-0.jsonl")
        self.server_starts = 0
        self.server_listening = False
        self.server_stopping = False
        self.restart_requested = False
        self.output_tail = ''
        self.load_stage = LoadStage()
        self.gpu_memory = {}
        self.gpu_query_running = False
        self.resource_results = queue.Queue()
        self.config_library = ConfigurationLibrary(Path(self.settings.fileName()).with_suffix('.profiles.json'))
        self.active_configuration_name = str(self.settings.value('active_configuration', ''))

        self.setWindowTitle("NInfer")
        self.resize(1180, 820)
        self.setMinimumSize(960, 680)
        self._apply_style()
        self._build_widgets()
        self.estimate_timer = QTimer(self)
        self.estimate_timer.setSingleShot(True)
        self.estimate_timer.setInterval(180)
        self.estimate_timer.timeout.connect(self._refresh_memory_estimate)
        self._load_settings()
        self._refresh_command()
        self._reload_configurations()

        self.health_timer = QTimer(self)
        self.health_timer.setInterval(2000)
        self.health_timer.timeout.connect(self._poll_health)
        self.health_timer.start()
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(100)
        self.queue_timer.timeout.connect(self._drain_health_results)
        self.queue_timer.timeout.connect(self._poll_cache)
        self.queue_timer.timeout.connect(self._drain_resource_results)
        self.queue_timer.timeout.connect(self._poll_model)
        self.queue_timer.start()
        self.resource_timer = QTimer(self)
        self.resource_timer.setInterval(5000)
        self.resource_timer.timeout.connect(self._query_resources)
        self.resource_timer.start()
        self._refresh_memory_estimate()
        self._query_resources()

    def _apply_style(self) -> None:
        # Dialogs are separate windows and do not inherit the main window's palette.
        # Own the theme at application scope so their text and backgrounds agree.
        app = QApplication.instance()
        app.setStyle('Fusion')
        palette = QPalette()
        roles = QPalette.ColorRole if QT6 else QPalette
        groups = QPalette.ColorGroup if QT6 else QPalette
        for role, color in {
            'Window': '#f6f7f9', 'WindowText': '#252b35',
            'Base': '#ffffff', 'AlternateBase': '#f0f3f7',
            'Text': '#252b35', 'Button': '#ffffff', 'ButtonText': '#252b35',
            'Highlight': '#34465c', 'HighlightedText': '#ffffff',
            'Light': '#ffffff', 'Midlight': '#e9edf2', 'Mid': '#a1a7b0',
            'Dark': '#737b88', 'Shadow': '#252b35',
            'ToolTipBase': '#ffffff', 'ToolTipText': '#252b35',
        }.items():
            palette.setColor(getattr(roles, role), QColor(color))
        for role in ('WindowText', 'Text', 'ButtonText'):
            palette.setColor(groups.Disabled, getattr(roles, role), QColor('#8a929e'))
        app.setPalette(palette)
        app.setStyleSheet('''
            QWidget { color: #252b35; font-size: 13px; }
            NInferGui, QDialog { background: #f6f7f9; }
            QToolTip { background: white; color: #252b35; border: 1px solid #dce1e7; padding: 5px; }
            QLabel#title { font-size: 22px; font-weight: 600; }
            QLabel#muted { color: #737b88; }
            QGroupBox { background: white; border: 1px solid #e3e6eb; border-radius: 9px;
                        margin-top: 10px; padding: 16px 12px 12px; font-weight: 500; }
            QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 4px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: white; border: 1px solid #dce1e7; border-radius: 5px; padding: 6px;
                selection-background-color: #dce4ee; selection-color: #17202b;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #778498; }
            QComboBox::drop-down { border: 0; width: 24px; }
            QComboBox QAbstractItemView {
                background: white; color: #252b35; border: 1px solid #dce1e7;
                selection-background-color: #34465c; selection-color: white;
                outline: 0;
            }
            QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled { background: #f4f5f7; color: #a1a7b0; }
            QPushButton { background: white; border: 1px solid #dce1e7; border-radius: 5px; padding: 7px 12px; }
            QPushButton:hover { background: #eef1f5; }
            QPushButton:pressed { background: #e3e8ee; }
            QPushButton:disabled { color: #a7adb6; background: #f6f7f9; border-color: #e7e9ed; }
            QPushButton#primary { background: #293442; color: white; border-color: #293442; }
            QPushButton#primary:hover { background: #3b4859; }
            QPushButton#primary:disabled { background: #e5e8ed; color: #a1a7b0; border-color: #e5e8ed; }
            QToolButton#kvSettings { color: #737b88; background: transparent; border: 0; padding: 6px; }
            QToolButton#kvSettings:hover { background: #eef1f5; border-radius: 4px; }
            QToolButton#kvSettings:checked { color: #252b35; }
            QToolButton#kvSettings:disabled { color: #a1a7b0; }
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: transparent; padding: 9px 12px; color: #77808e; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected { color: #202a37; border-bottom-color: #54657a; }
            QScrollArea { border: 0; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QProgressBar { border: 0; border-radius: 3px; background: #e9edf2; min-height: 6px; max-height: 6px; }
            QProgressBar::chunk { border-radius: 3px; background: #7c8fa5; }
            QSplitter::handle { background: transparent; width: 14px; }
            QCheckBox { spacing: 7px; padding: 3px 0; }
            QCheckBox:disabled { color: #8a929e; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border: 1px solid #8a95a4;
                border-radius: 3px; background: white;
            }
            QCheckBox::indicator:unchecked:hover { border-color: #34465c; background: #edf2f7; }
            QCheckBox::indicator:checked {
                background: #34465c; border-color: #34465c;
                image: url("CHECK_ICON");
            }
            QCheckBox::indicator:checked:hover { background: #445a75; border-color: #445a75; }
            QCheckBox::indicator:focus { border: 2px solid #617d9f; width: 14px; height: 14px; }
            QCheckBox::indicator:unchecked:disabled { background: #edf0f4; border-color: #b9c1cc; }
            QCheckBox::indicator:checked:disabled { background: #8b98aa; border-color: #8b98aa; }
        '''.replace('CHECK_ICON', (Path(__file__).resolve().parent / 'icons' / 'check.svg').as_posix()))

    def _build_widgets(self) -> None:
        self.binary = QLineEdit(str(self._default_binary()))
        self.artifact = QLineEdit(str(self._quasar_artifact()))
        self.host = QLineEdit("127.0.0.1")
        self.port = self._spin(6200, 1, 65535, 1)
        self.tp = QComboBox()
        self.tp.addItems(["2", "1", "4"])
        self.devices = QLineEdit("0,1")
        self.pp = QComboBox()
        self.pp.addItems(["1", "2"])
        self.pp.setToolTip("PP2 为实验性分层流水线：TP1、两张卡、并发1、纯文本 None/MTP；未确立速度收益。")
        self.pipeline_chunks = self._spin(8, 1, 8, 1)
        self.max_context = self._spin(180000, 128, 262144, 128)
        self.kv_capacity = QLineEdit()
        self.kv_capacity.setPlaceholderText("留空跟随上下文，或填 auto")
        self.kv_capacity.setToolTip("auto：按启动时可用显存选择；留空：跟随上下文容量；整数：不得小于上下文容量。")
        self.kv_capacity_toggle = QToolButton()
        self.kv_capacity_toggle.setObjectName('kvSettings')
        self.kv_capacity_toggle.setText("独立 KV")
        self.kv_capacity_toggle.setCheckable(True)
        self.kv_capacity_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon if QT6 else Qt.ToolButtonTextBesideIcon)
        self.kv_capacity_toggle.setArrowType(Qt.ArrowType.RightArrow if QT6 else Qt.RightArrow)
        self.kv_capacity_toggle.setToolTip("展开后可单独设置 KV；收起时清除独立设置，跟随上下文容量。")
        self.prefill_chunk = self._spin(2560, 128, 1048576, 128)
        self.kv_dtype = QComboBox()
        self.kv_dtype.addItems(["int8", "bf16"])
        self.spec = QComboBox()
        self.spec.addItems(["mtp", "none", "dflash"])
        self.draft_tokens = self._spin(4, 0, 15, 1)
        self.lm_head_draft = QCheckBox("使用优化草稿头")
        self.lm_head_draft.setChecked(True)
        self.cuda_graph = QCheckBox("启用 CUDA Graph")
        self.cuda_graph.setChecked(True)
        self.prefix_reuse = QCheckBox("启用前缀缓存")
        self.prefix_reuse.setChecked(True)
        self.prefix_cache_tokens = self._spin(384000, 0, 100000000, 10000)
        self.prefix_cache_tokens.setToolTip(
            "历史分枝的主机 RAM KV 预算，推荐 384,000 tokens；0 仅保留当前 GPU 前缀。\n"
            "按唯一 64-token 页计费，共享树干不重复计数，超额淘汰最旧分枝。\n"
            "27B INT8 主 KV 加 MTP 约 12.8 GiB / 384,000 tokens，另有至多 2 GiB 历史状态。\n"
            "完整 GPU KV 的纯文本 None/MTP 支持；恢复旧枝需 RAM→GPU 传输。")
        self.vision = QCheckBox("开启视觉（图片 / 视频）")
        self.vision.setToolTip("对应 --vision；默认关闭，需重启 API。关闭时不加载视觉 GPU 权重。\n"
                               "开启时关闭快速数值路径和前缀复用（可手动启用缓存）；视觉预算与文本容量独立。")
        self.vision_device = QComboBox()
        self.vision_device.addItems(["gpu", "cpu"])
        self.vision_device.setToolTip("GPU：较快；CPU：视觉塔仅驻主机 RAM，图片/视频编码较慢，文本推理仍在 GPU。\n"
                                      "CPU 使用 FP32 计算并保留 BF16 激活边界，输出可能有数值差异，未做 GPQA。\n"
                                      "CPU 模式将 API 准备/排队超时设为 10 分钟，以容纳慢速媒体编码。")
        self.vision_max_tokens = self._spin(1024, 1, 32768, 128)
        self.vision_max_tokens.setToolTip("每个请求的图片/视频合计视觉 token 上限；增加会占用更多显存。")
        self.volta_fast_decode = QCheckBox("实验：NVFP4 快速数值路径（MTP / DFlash）")
        self.volta_long_attention = QCheckBox("实验：完整历史长注意力（文本 / 图片 / 视频）")
        self.ram_kv = QCheckBox("实验：RAM-KV 近似历史检索")
        self.ram_window = self._spin(32768, 64, 262144, 64)
        self.ram_budget = QLineEdit("32000000000")
        self.ram_kv.setToolTip("默认关闭；prefill 和 decode 都可能只关注检索到的历史，可能降低长上下文质量。\n"
                               "本配置无确定速度/GPQA数据；RAM archive 预算不等于进程 RSS 限制。\n"
                               "KV 容量启用时自动等于 GPU window。与视觉、DFlash、TP4、长注意力互斥。")
        self.volta_fast_decode.setToolTip(
            "对应 --volta-fast-decode；QUASAR / SM70 / TP2/TP4 / 纯文本 None/MTP，DFlash 仅 TP2。\n"
            "NVFP4 FP32 末尾缩放、MTP W8 重排与归约；可能改变 logits、草稿接受和输出。\n"
            "DFlash7：85K 122.65→128.11 tok/s（+4.45%）；长注意力组合 168.15（+37.10%）。\n"
            "DFlash 仅改目标 NVFP4，保留 W8/BF16 drafter；HumanEval 64 次有 2 次输出变化，质量未测。\n"
            "85K / MTP4 组合：本机 TP2 约 198.1 tok/s；远端 TP4 / 220 W 约 267.5 tok/s。\n"
            "TP4 比 231.3 组合约 +12.46%；prefill 约 2337 tok/s，Split-D 可能改变输出。\n"
            "TP4 普通、前缀及实际 85K Graph/eager 状态检查均通过；不代表质量无损。\n"
            "不是单开关独立增幅。前序组合25点最大 KL 0.001903、最小 cosine 0.998831；\n"
            "前序近无损界限仅9/25点通过；当前组合GPQA未跑，智力影响未知。\n"
            "默认关闭，需重启 API。")
        self.volta_long_attention.setToolTip(
            "对应 --volta-long-attention；SM70 27B / TP2/TP4 / INT8 KV / 并发1–8；视觉和 DFlash 仅 TP2。\n"
            "保留全部历史 KV，改变 softmax/PV 累加顺序；可能改变输出。\n"
            "DFlash7：85K 原始代码前缀 122.65→167.49 tok/s（+36.57%），输出改变，质量未测。\n"
            "DFlash 与快速数值组合 168.15 tok/s；HumanEval 短输入约 301→302，非质量分数。\n"
            "纯文本与快速数值路径组合约 198.1 tok/s，不代表视觉或此开关单项收益；当前组合 GPQA 未测。\n"
            "默认关闭，需重启 API；不能与 RAM-KV 一起使用；DFlash 数值/接受率可能改变，质量未评估。")
        self.max_concurrency = self._spin(1, 1, 8, 1)
        self.default_max_tokens = self._spin(65536, 1, 1048576, 1024)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password if QT6 else QLineEdit.Password)
        self.temperature = QLineEdit()
        self.top_p = QLineEdit()
        self.top_k = QLineEdit()
        self.min_p = QLineEdit()
        self.presence_penalty = QLineEdit()
        self.frequency_penalty = QLineEdit()
        self.seed = QLineEdit()
        for field in (self.temperature, self.top_p, self.top_k, self.min_p,
                      self.presence_penalty, self.frequency_penalty, self.seed):
            field.setPlaceholderText("留空 = 模型默认 / 由请求设置")
        self.no_thinking = QCheckBox("默认关闭 thinking")
        self.preserve_thinking = QCheckBox("保留 thinking")
        self.greedy = QCheckBox("强制 greedy")
        self.cors = QCheckBox("允许 CORS")

        self.status = QLabel("未启动")
        self.status.setStyleSheet("color: #777; font-weight: bold")
        self.command = QLineEdit()
        self.command.setReadOnly(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        browse_binary = QPushButton("浏览")
        browse_artifact = QPushButton("浏览")
        self.path_buttons = (browse_binary, browse_artifact)
        browse_binary.clicked.connect(lambda: self._browse(self.binary, executable=True))
        browse_artifact.clicked.connect(lambda: self._browse(self.artifact, executable=False))

        model = QFormLayout()
        model.addRow("模型文件", self._with_button(self.artifact, browse_artifact))
        model.addRow("服务程序", self._with_button(self.binary, browse_binary))
        model.addRow("张量并行 TP", self.tp)
        model.addRow("设备 IDs", self.devices)
        model.addRow("Prefill chunk", self.prefill_chunk)
        inference = QFormLayout()
        inference.addRow("上下文容量", self._with_button(self.max_context, self.kv_capacity_toggle))
        self.kv_capacity_panel = QWidget()
        kv_layout = QFormLayout(self.kv_capacity_panel)
        kv_layout.setContentsMargins(12, 0, 0, 0)
        kv_layout.addRow("KV 容量 / auto", self.kv_capacity)
        self.kv_capacity_hint = QLabel()
        self.kv_capacity_hint.setObjectName('muted')
        self.kv_capacity_hint.setWordWrap(True)
        kv_layout.addRow(self.kv_capacity_hint)
        inference.addRow(self.kv_capacity_panel)
        self.kv_capacity_panel.hide()
        self.max_context.setToolTip("上下文容量也是默认 KV 容量；需要单独设置时展开“独立 KV”。容量不是当前输入长度。")
        for label, widget in (("KV 格式", self.kv_dtype),
                              ("推理后端", self.spec), ("草稿 token 数", self.draft_tokens),
                              ("草稿头", self.lm_head_draft), ("CUDA Graph", self.cuda_graph),
                              ("前缀缓存", self.prefix_reuse), ("历史缓存预算 / tokens", self.prefix_cache_tokens),
                              ("视觉", self.vision),
                              ("视觉推理设备", self.vision_device),
                              ("视觉 token 预算", self.vision_max_tokens),
                              ("默认最大输出", self.default_max_tokens)):
            inference.addRow(label, widget)

        service = QFormLayout()
        for label, widget in (("Host", self.host), ("Port", self.port), ("API key", self.api_key),
                              ("最大并发请求", self.max_concurrency)):
            service.addRow(label, widget)
        for check in (self.no_thinking, self.preserve_thinking, self.greedy, self.cors):
            service.addRow(check)

        sampling = QFormLayout()
        for label, field in (("Temperature", self.temperature), ("Top P", self.top_p),
                             ("Top K", self.top_k), ("Min P", self.min_p),
                             ("Presence penalty", self.presence_penalty),
                             ("Frequency penalty", self.frequency_penalty), ("Seed", self.seed)):
            sampling.addRow(label, field)
        sampling_note = QLabel("留空使用模型默认值；请求参数可覆盖。")
        sampling_note.setObjectName('muted')
        sampling_note.setWordWrap(True)
        sampling.addRow(sampling_note)

        experimental = QFormLayout()
        experimental.addRow(self.volta_fast_decode)
        experimental.addRow(self.volta_long_attention)
        experimental.addRow(self.ram_kv)
        experimental.addRow("RAM-KV GPU tokens", self.ram_window)
        experimental.addRow("RAM archive bytes", self.ram_budget)
        experimental.addRow("流水线 PP", self.pp)
        experimental.addRow("PP prefill chunks", self.pipeline_chunks)
        attention_note = QLabel("长注意力支持 TP2 图片/视频；快速数值路径仅限文本。数值实验可能改变输出；RAM-KV 使用近似注意力。")
        attention_note.setObjectName('muted')
        attention_note.setWordWrap(True)
        experimental.addRow(attention_note)
        impact_button = QPushButton("数值影响与测量说明")
        impact_button.clicked.connect(self.show_experimental_details)
        experimental.addRow(impact_button)

        self.controls = QTabWidget()
        self.controls.setMinimumWidth(410)
        for title, form in (("模型", model), ("推理", inference), ("采样", sampling),
                            ("服务", service), ("实验", experimental)):
            form.setSpacing(12)
            page = QWidget()
            column = QVBoxLayout(page)
            column.setContentsMargins(8, 18, 8, 8)
            column.addLayout(form)
            column.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.controls.addTab(scroll, title)

        title = QLabel("NInfer")
        title.setObjectName('title')
        self.start_button = QPushButton("启动")
        self.start_button.setObjectName('primary')
        self.stop_button = QPushButton("停止")
        self.restart_button = QPushButton("重启")
        self.offload_button = QPushButton("卸载到内存")
        self.load_button = QPushButton("加载回显存")
        self.offload_button.setToolTip("保留原始模型在系统内存，释放显存；KV/前缀缓存会清空")
        self.load_button.setToolTip("从系统内存恢复模型到显存；重新创建推理缓存和 CUDA Graph")
        self.offload_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.stop_server)
        self.restart_button.clicked.connect(self.restart_server)
        self.offload_button.clicked.connect(lambda: self._model_action('offload'))
        self.load_button.clicked.connect(lambda: self._model_action('load'))
        header = QHBoxLayout()
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(self.status)
        header.addStretch(1)
        for button in (self.start_button, self.stop_button, self.restart_button,
                       self.offload_button, self.load_button):
            header.addWidget(button)

        self.config_selector = QComboBox()
        self.config_selector.setMinimumWidth(180)
        if QT6:
            self.config_selector.setPlaceholderText("选择已保存配置")
        self.config_load = QPushButton("加载")
        self.config_save = QPushButton("保存为")
        self.config_delete = QPushButton("删除")
        self.config_import = QPushButton("导入")
        self.config_export = QPushButton("导出")
        self.config_load.clicked.connect(self.load_configuration)
        self.config_save.clicked.connect(self.save_configuration)
        self.config_delete.clicked.connect(self.delete_configuration)
        self.config_import.clicked.connect(self.import_configuration)
        self.config_export.clicked.connect(self.export_configuration)
        bank = QHBoxLayout()
        bank.addWidget(QLabel("配置"))
        bank.addWidget(self.config_selector, 1)
        for button in (self.config_load, self.config_save, self.config_delete,
                       self.config_import, self.config_export):
            bank.addWidget(button)
        self.config_notice = QLabel("")
        self.config_notice.setObjectName('muted')

        self.memory_note = QLabel("正在读取模型目录…")
        self.memory_note.setObjectName('muted')
        self.memory_note.setWordWrap(True)
        self.memory_rows = []
        memory_grid = QGridLayout()
        memory_grid.setSpacing(14)
        for index in range(4):
            block = QWidget()
            box = QVBoxLayout(block)
            box.setContentsMargins(0, 0, 0, 0)
            heading, detail = QLabel(), QLabel()
            detail.setObjectName('muted')
            detail.setWordWrap(True)
            bar = QProgressBar()
            bar.setTextVisible(False)
            box.addWidget(heading)
            box.addWidget(bar)
            box.addWidget(detail)
            memory_grid.addWidget(block, index // 2, index % 2)
            self.memory_rows.append((block, heading, detail, bar))
            block.hide()
        memory_layout = QVBoxLayout()
        memory_layout.addLayout(memory_grid)
        memory_layout.addWidget(self.memory_note)
        memory_card = self._group("显存预估", memory_layout)

        self.load_label = QLabel("未启动")
        self.load_progress = QProgressBar()
        self.load_progress.setTextVisible(False)
        self.load_progress.setRange(0, 1000)
        self.load_progress.setValue(0)
        loading = QVBoxLayout()
        loading.addWidget(self.load_label)
        loading.addWidget(self.load_progress)
        self.load_card = self._group("加载进度", loading)

        self.cache_latest = QLabel("最近请求：—")
        self.cache_total = QLabel("本次启动累计：—")
        self.draft_live = QLabel("实时草稿接受率：—")
        self.throughput_live = QLabel("实时吞吐：—")
        self.request_last = QLabel("上次请求：—")
        metrics = QVBoxLayout()
        metrics.setContentsMargins(12, 18, 12, 12)
        metrics.setSpacing(16)
        for label in (self.throughput_live, self.draft_live, self.cache_latest,
                      self.cache_total, self.request_last):
            label.setWordWrap(True)
            metrics.addWidget(label)
        metrics.addStretch(1)
        metrics_page = QWidget()
        metrics_page.setLayout(metrics)

        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont if QT6 else QFontDatabase.FixedFont))
        self.save_log_button = QPushButton("保存日志")
        self.save_log_button.clicked.connect(self.save_log)
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(0, 12, 0, 0)
        log_layout.addWidget(self.log, 1)
        log_layout.addWidget(self.save_log_button)

        self.health_button = QPushButton("检查连接")
        self.open_button = QPushButton("打开 API 地址")
        self.copy_button = QPushButton("复制命令")
        self.health_button.clicked.connect(lambda: self._check_health(force=True))
        self.open_button.clicked.connect(self.open_api)
        self.copy_button.clicked.connect(self.copy_command)
        command_page = QWidget()
        command_layout = QVBoxLayout(command_page)
        command_layout.setContentsMargins(0, 12, 0, 0)
        command_layout.addWidget(self.command)
        for button in (self.copy_button, self.health_button, self.open_button):
            command_layout.addWidget(button)
        command_layout.addStretch(1)
        self.activity_tabs = QTabWidget()
        for label, page in (("指标", metrics_page), ("日志", log_page), ("命令", command_page)):
            self.activity_tabs.addTab(page, label)

        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(12)
        monitor_layout.addWidget(memory_card)
        monitor_layout.addWidget(self.load_card)
        monitor_layout.addWidget(self.activity_tabs, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(monitor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 650])
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        root.addLayout(header)
        root.addLayout(bank)
        root.addWidget(self.config_notice)
        root.addWidget(splitter, 1)
        self.config_notice.hide()
        for widget in self.findChildren(QLineEdit):
            if widget is not self.command:
                widget.textChanged.connect(self._refresh_command)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._refresh_command)
        for widget in (self.tp, self.pp, self.spec, self.kv_dtype):
            widget.currentTextChanged.connect(self._refresh_command)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._refresh_command)
        self.spec.currentTextChanged.connect(self._spec_changed)
        self.vision.toggled.connect(self._vision_changed)
        self.ram_kv.toggled.connect(self._update_feature_controls)
        self.pp.currentTextChanged.connect(self._update_feature_controls)
        self.kv_capacity_toggle.toggled.connect(self._toggle_kv_capacity)
        self.config_selector.currentIndexChanged.connect(
            lambda _: self._set_configuration_lock(self.process is not None))

    @staticmethod
    def _spin(value: int, minimum: int, maximum: int, step: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons if QT6 else QSpinBox.NoButtons)
        return widget

    @staticmethod
    def _with_button(widget: QWidget, button: QPushButton | QToolButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _group(title: str, layout) -> QGroupBox:
        group = QGroupBox(title)
        group.setLayout(layout)
        return group

    def _browse(self, target: QLineEdit, executable: bool) -> None:
        if executable:
            path, _ = QFileDialog.getOpenFileName(self, "选择 ninfer-serve", str(self.repo))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择模型 artifact", str(self.repo), "NInfer artifact (*.ninfer);;All files (*)"
            )
        if path:
            target.setText(path)

    def _default_binary(self) -> Path:
        if self.binary_override:
            return self._local_path(self.binary_override)
        return _first_existing([
            self.repo / 'build-v100/apps/ninfer-serve',
            self.repo / 'build-v100-tp4/apps/ninfer-serve',
            self.repo / 'build/apps/ninfer-serve',
        ])

    def _quasar_artifact(self) -> Path:
        if self.artifact_override:
            return self._local_path(self.artifact_override)
        return _first_existing([self.repo / "models/quasar-v3/qwen3_8_27b_nvfp4.ninfer",
                               Path("/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer")])

    def _option_widgets(self) -> dict:
        names = ("binary artifact host devices kv_capacity ram_budget temperature top_p top_k min_p "
                 "presence_penalty frequency_penalty seed port max_context prefill_chunk draft_tokens "
                 "vision_max_tokens pipeline_chunks ram_window max_concurrency default_max_tokens "
                 "tp pp kv_dtype spec lm_head_draft cuda_graph prefix_reuse prefix_cache_tokens volta_long_attention "
                 "volta_fast_decode vision vision_device ram_kv no_thinking preserve_thinking greedy cors")
        return {name: getattr(self, name) for name in names.split()}

    def configuration(self) -> dict:
        values = {}
        for name, widget in self._option_widgets().items():
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            else:
                values[name] = widget.text()
        return values

    def _validate_configuration(self, values: dict) -> None:
        if isinstance(values, dict):
            values = {"vision_device": "gpu", **values}
        widgets = self._option_widgets()
        if not isinstance(values, dict) or set(values) != set(widgets):
            raise ValueError("配置项不完整或包含未知字段")
        for name, widget in widgets.items():
            value = values[name]
            if isinstance(widget, QCheckBox):
                valid = type(value) is bool
            elif isinstance(widget, QSpinBox):
                valid = type(value) is int and widget.minimum() <= value <= widget.maximum()
            elif isinstance(widget, QComboBox):
                valid = type(value) is str and value in [widget.itemText(i) for i in range(widget.count())]
            else:
                valid = type(value) is str
            if not valid:
                raise ValueError(f"配置项无效：{name}")
        for name, _ in self._sampling_fields():
            validate_sampling(name, values[name])

    def apply_configuration(self, values: dict) -> None:
        values = {"vision_device": "gpu", **values}
        if self.process is not None:
            raise ValueError("请先停止服务再加载配置")
        self._validate_configuration(values)
        widgets = self._option_widgets()
        blocked = [(widget, widget.blockSignals(True)) for widget in widgets.values()]
        try:
            for name, widget in widgets.items():
                value = values[name]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(value)
                elif isinstance(widget, QSpinBox):
                    widget.setValue(value)
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(value)
                else:
                    widget.setText(value)
        finally:
            for widget, previous in blocked:
                widget.blockSignals(previous)
        self.kv_capacity_toggle.setChecked(bool(self.kv_capacity.text().strip()))
        self._update_feature_controls()
        self.draft_tokens.setEnabled(self.spec.currentText() != 'none')
        self.lm_head_draft.setEnabled(self.spec.currentText() != 'none')
        self._refresh_command()

    def _load_settings(self) -> None:
        values = self.configuration()
        for name, widget in self._option_widgets().items():
            value = self.settings.value(name)
            if value is not None:
                if isinstance(widget, QCheckBox):
                    value = str(value).lower() in {"1", "true", "yes"}
                elif isinstance(widget, QSpinBox):
                    value = max(widget.minimum(), min(widget.maximum(), int(value)))
                else:
                    value = str(value)
                values[name] = value
        if self.binary_override:
            values['binary'] = self.binary_override
        if self.artifact_override:
            values['artifact'] = self.artifact_override
        self.apply_configuration(values)

    def _save_settings(self) -> None:
        for name, value in self.configuration().items():
            self.settings.setValue(name, value)
        self.settings.remove("api_key")
        self.settings.setValue('active_configuration', self.active_configuration_name)
        self.settings.sync()

    def _reload_configurations(self, selected: str = '') -> None:
        self.config_selector.clear()
        try:
            self.config_library.initialize(Path(__file__).resolve().parent / 'configs' / 'defaults.json')
            self.config_selector.addItems(sorted(self.config_library.entries()))
            name = selected or self.active_configuration_name
            self.config_selector.setCurrentIndex(self.config_selector.findText(name) if name else -1)
        except (OSError, ValueError) as error:
            self.config_notice.setText(f"配置库读取失败：{error}")
            self.config_notice.show()
        self._set_configuration_lock(self.process is not None)

    def save_named_configuration(self, name: str) -> None:
        values = self.configuration()
        self._validate_configuration(values)
        self.config_library.save(name, values)
        self.active_configuration_name = name.strip()
        self._reload_configurations(name.strip())

    def load_named_configuration(self, name: str) -> None:
        values = self.config_library.entries()[name]
        self.apply_configuration(values)
        self.active_configuration_name = name
        self.config_selector.setCurrentText(name)
        self._save_settings()

    def _confirm_replace_configuration(self, name: str) -> bool:
        if name not in self.config_library.entries():
            return True
        yes = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
        no = QMessageBox.StandardButton.No if QT6 else QMessageBox.No
        return QMessageBox.question(self, "覆盖配置", f"覆盖“{name}”？", yes | no, no) == yes

    def save_configuration(self) -> None:
        name, ok = QInputDialog.getText(self, "保存配置", "名称", text=self.config_selector.currentText())
        if not ok or not name.strip():
            return
        try:
            if self._confirm_replace_configuration(name.strip()):
                self.save_named_configuration(name.strip())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "配置保存失败", str(error))

    def load_configuration(self) -> None:
        name = self.config_selector.currentText()
        if not name:
            return
        try:
            self.load_named_configuration(name)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "配置加载失败", str(error))

    def delete_configuration(self) -> None:
        name = self.config_selector.currentText()
        if not name:
            return
        yes = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
        no = QMessageBox.StandardButton.No if QT6 else QMessageBox.No
        if QMessageBox.question(self, "删除配置", f"删除“{name}”？", yes | no, no) != yes:
            return
        try:
            self.config_library.delete(name)
            if self.active_configuration_name == name:
                self.active_configuration_name = ''
            self._reload_configurations()
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "配置删除失败", str(error))

    def import_configuration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入配置", str(self.repo), "JSON (*.json)")
        if not path:
            return
        try:
            name, values = self.config_library.read_export(Path(path))
            self._validate_configuration(values)
            if self._confirm_replace_configuration(name):
                self.config_library.save(name, values)
                self.apply_configuration(values)
                self.active_configuration_name = name
                self._reload_configurations(name)
                self._save_settings()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "配置导入失败", str(error))

    def export_configuration(self) -> None:
        name = self.config_selector.currentText() or "自定义"
        path, _ = QFileDialog.getSaveFileName(self, "导出当前配置", str(self.repo / 'ninfer-config.json'), "JSON (*.json)")
        if path:
            try:
                self.config_library.export(Path(path), name, self.configuration())
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "配置导出失败", str(error))

    def show_experimental_details(self) -> None:
        QMessageBox.information(self, "实验说明",
            "完整历史长注意力支持 TP2 文本（None/MTP/DFlash）、图片和视频（None/MTP）、完整 INT8 KV。\n"
            "GPU/CPU 视觉塔均可用。DFlash 快速数值路径只修改目标模型，保留原生 W8/BF16 drafter。\n"
            "开启视觉会保留长注意力；快速数值路径仍仅限文本。两项实验默认关闭，修改后需重启 API。\n\n"
            "DFlash7 / TP2：85K 原始代码前缀普通 122.65、长注意力 167.49、快速数值 128.11、组合 168.15 tok/s。\n"
            "组合 +37.10%；HumanEval 32题×2 约 301.21→302.38，非 pass@1。\n"
            "组合改变 2/64 个短请求输出及全部 85K 请求输出；质量影响未知，未测 GPQA。\n\n"
            "纯文本数值路径 + 长注意力：85K 输入、MTP4、greedy 下组合约 198.1 tok/s；末步 W8 约 +0.84%。\n"
            "不是各开关的独立增幅。前序组合 25 点 argmax 一致，最小 cosine 0.998831，最大 KL 0.001903；\n"
            "near-lossless 仅 9/25 通过，当前组合未测 GPQA。可能改变输出和草稿接受位置。\n\n"
            "RAM-KV 在长 prefill 和 decode 中使用检索历史，可能降低长上下文质量；收益未测。\n"
            "RAM archive 预算不是进程 RSS 上限。PP2 无确定速度增幅。\n"
            "实验默认关闭；模型与硬件准入仍由引擎检查。")

    def _local_path(self, text: str) -> Path:
        path = Path(text.strip()).expanduser()
        return path if path.is_absolute() else self.repo / path

    def _validate(self) -> str | None:
        artifact = self._local_path(self.artifact.text())
        binary = self._local_path(self.binary.text())
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return f"找不到可执行 ninfer-serve：{binary}"
        if not artifact.is_file() or artifact.suffix != ".ninfer":
            return f"找不到模型 artifact：{artifact}"
        devices = [item.strip() for item in self.devices.text().split(",")]
        tp = int(self.tp.currentText())
        pp = int(self.pp.currentText())
        if (len(devices) != tp * pp or not all(x.isdecimal() for x in devices) or
                len(set(map(int, devices))) != len(devices)):
            return f"TP{tp} / PP{pp} 必须填写 {tp * pp} 个不同的设备 ID"
        spec = self.spec.currentText()
        if pp == 2 and (tp != 1 or self.max_concurrency.value() != 1 or self.vision.isChecked() or
                        spec == "dflash" or self.volta_fast_decode.isChecked() or self.volta_long_attention.isChecked()):
            return "实验 PP2 要求 TP1、两卡、并发1、NVFP4/QUASAR 纯文本 None/MTP；关闭两个数值实验"
        if self.volta_long_attention.isChecked() and (
                tp not in (2, 4) or self.kv_dtype.currentText() != "int8" or
                self.ram_kv.isChecked() or (spec == "dflash" and tp != 2)):
            return ("实验性长上下文注意力要求 TP2/TP4、INT8 KV、并发 1–8、视觉和 DFlash 仅支持 TP2；"
                    "请调整配置或取消勾选该开关。硬件和模型还须为 SM70 27B。")
        if self.volta_fast_decode.isChecked() and (tp not in (2, 4) or self.vision.isChecked() or
                                                    (spec == "dflash" and tp != 2)):
            return "实验性快速数值路径要求 QUASAR / SM70 / TP2/TP4 / 纯文本；DFlash 仅 TP2"
        if self.vision.isChecked() and (tp != 2 or spec == "dflash"):
            return "视觉要求 TP2，后端为 none 或 mtp；不能与 DFlash 一起使用"
        if self.vision.isChecked() and artifact.name in {"qwen3_8_27b_q4_k_m.ninfer", "qwen3_8_27b_fp8.ninfer"}:
            return "该 Q4_K_M / 原生 FP8 产物不支持视觉；请选择 QUASAR NVFP4"
        if tp == 4 and spec == "dflash":
            return "当前产品契约不支持 TP4 DFlash；请使用 TP2"
        if spec == "none" and self.lm_head_draft.isChecked():
            return "优化草稿头只能与 MTP/DFlash 一起使用"
        if spec == "mtp" and not 1 <= self.draft_tokens.value() <= 5:
            return "MTP 最大草稿数必须为 1..5（实际接受 0 个仍有效）"
        if spec == "dflash" and not 1 <= self.draft_tokens.value() <= 7:
            return "27B DFlash 草稿数必须为 1..7"
        if not self.host.text().strip():
            return "Host 不能为空"
        if self.prefill_chunk.value() % 128:
            return "Prefill chunk 必须为 128 的正整数倍"
        capacity = self.kv_capacity.text().strip()
        if not self.ram_kv.isChecked() and capacity and capacity != "auto" and (not capacity.isdecimal() or int(capacity) < self.max_context.value()):
            return (f"KV 容量“{capacity}”无效，当前最大上下文为 {self.max_context.value()}。\n"
                    f"请填 auto，或不小于 {self.max_context.value()} 的整数；清空则跟随最大上下文。")
        if self.ram_kv.isChecked():
            if self.vision.isChecked() or spec == "dflash" or tp == 4 or self.max_concurrency.value() != 1:
                return "实验 RAM-KV 要求 TP1/TP2/PP2、并发1、纯文本 None/MTP"
            minimum = self.prefill_chunk.value() * (self.pipeline_chunks.value() if pp == 2 else 1) + 8192
            window = self.ram_window.value()
            if window % 64 or not minimum <= window <= self.max_context.value():
                return f"RAM-KV 窗口须为64倍数，且位于 {minimum}..{self.max_context.value()}"
            budget = self.ram_budget.text().strip()
            if not budget.isdecimal() or not 0 < int(budget) <= 18446744073709551615:
                return "RAM archive 预算须为正 uint64 字节数，不是进程 RSS 上限"
        if artifact.name == "qwen3_8_27b_fp8.ninfer" and tp != 4:
            return "原生 FP8 产物要求 TP4；请设置 TP4 和四个设备 ID"
        for name, field in self._sampling_fields():
            try:
                validate_sampling(name, field.text())
            except ValueError:
                return f"采样参数 {name} 的值或范围无效"
        return None

    def _args(self) -> list[str]:
        args = [str(self._local_path(self.artifact.text())), "--host", self.host.text().strip(),
                "--port", str(self.port.value()), "--tp", self.tp.currentText(),
                "--devices", self.devices.text().strip(), "--max-context", str(self.max_context.value()),
                "--kv-dtype", self.kv_dtype.currentText(),
                "--prefill-chunk", str(self.prefill_chunk.value()), "--max-concurrency",
                str(self.max_concurrency.value()), "--default-max-tokens", str(self.default_max_tokens.value()),
                "--request-log-jsonl", str(self.cache_monitor.path)]
        args += ["--log-stats-interval-ms", "1000"]
        spec = self.spec.currentText()
        if self.pp.currentText() != "1":
            args += ["--pp", self.pp.currentText(), "--pipeline-chunks", str(self.pipeline_chunks.value())]
        if self.ram_kv.isChecked():
            args += ["--ram-kv-window", str(self.ram_window.value()), "--kv-capacity", str(self.ram_window.value()),
                     "--ram-kv-budget-bytes", self.ram_budget.text().strip()]
        elif self.kv_capacity.text().strip():
            args += ["--kv-capacity", self.kv_capacity.text().strip()]
        if spec != "none":
            args += ["--spec", spec, "--draft-tokens", str(self.draft_tokens.value())]
        if spec != "none" and self.lm_head_draft.isChecked():
            args.append("--lm-head-draft")
        if not self.cuda_graph.isChecked():
            args.append("--no-cuda-graph")
        args += ["--prefix-cache-tokens", str(self.prefix_cache_tokens.value())]
        if not self.prefix_reuse.isChecked():
            args.append("--no-prefix-reuse")
        if self.volta_long_attention.isChecked():
            args.append("--volta-long-attention")
        if self.volta_fast_decode.isChecked():
            args.append("--volta-fast-decode")
        if self.vision.isChecked():
            args += ["--vision", "--vision-device", self.vision_device.currentText(),
                     "--vision-max-tokens", str(self.vision_max_tokens.value())]
            if self.vision_device.currentText() == "cpu":
                args += ["--pending-timeout-ms", "600000"]
        if self.api_key.text():
            args += ["--api-key", self.api_key.text()]
        if self.no_thinking.isChecked():
            args.append("--no-thinking")
        if self.preserve_thinking.isChecked():
            args.append("--preserve-thinking")
        if self.greedy.isChecked():
            args.append("--greedy")
        if self.cors.isChecked():
            args.append("--cors")
        for name, field in self._sampling_fields():
            if field.text().strip():
                args += ["--" + name.replace("_", "-"), field.text().strip()]
        return args

    def _command(self) -> list[str]:
        binary = self._local_path(self.binary.text())
        return [str(binary), *self._args()]

    def _set_command(self, command: list[str]) -> None:
        redacted = list(command)
        if "--api-key" in redacted:
            redacted[redacted.index("--api-key") + 1] = "<API_KEY>"
        self.command.setText(shlex.join(redacted))
        self.command.setCursorPosition(0)

    def _sampling_fields(self):
        return [(name, getattr(self, name)) for name in
                ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed")]

    def _refresh_command(self, *_args) -> None:
        self._set_command(self._command())
        if self.ram_kv.isChecked():
            self.kv_capacity_hint.setText("RAM-KV 已启用：GPU 容量使用实验页的 GPU window。")
        else:
            self.kv_capacity_hint.setText(
                f"留空跟随上下文（{self.max_context.value()} token）；auto 按可用显存选择。\n"
                f"手动填写整数时，至少为 {self.max_context.value()} token。")
        if hasattr(self, 'estimate_timer'):
            self.estimate_timer.start()

    def _memory_options(self) -> MemoryOptions:
        return MemoryOptions(tp=int(self.tp.currentText()), pp=int(self.pp.currentText()),
            context=self.max_context.value(), capacity=self.kv_capacity.text().strip(),
            kv_dtype=self.kv_dtype.currentText(), chunk=self.prefill_chunk.value(),
            backend=self.spec.currentText(), drafts=self.draft_tokens.value(),
            optimized=self.lm_head_draft.isChecked(), concurrency=self.max_concurrency.value(),
            vision=self.vision.isChecked(), vision_device=self.vision_device.currentText(), vision_tokens=self.vision_max_tokens.value(),
            graphs=self.cuda_graph.isChecked(), ram_window=self.ram_window.value() if self.ram_kv.isChecked() else 0)

    def _refresh_memory_estimate(self) -> None:
        try:
            estimates = estimate_memory(self._local_path(self.artifact.text()), self._memory_options())
            note = "按预留容量估算；工作区、Graph 与 CUDA 开销以区间计。"
            if any(estimate.auto for estimate in estimates):
                note = "auto 按上下文下限预估；实际 KV 容量由引擎根据空闲显存选择。"
        except (OSError, ValueError, KeyError, TypeError) as error:
            estimates = []
            note = f"预估不可用：{error}"
        host_vision = sum(estimate.host_vision_bytes for estimate in estimates)
        if host_vision:
            note += f" CPU 视觉塔权重约 {host_vision / GIB:.2f} GiB RAM，另需媒体工作区；图片/视频编码较慢。"
        ids = self.devices.text().split(',')
        count = min(4, int(self.tp.currentText()) * int(self.pp.currentText()))
        for rank, (block, heading, detail, bar) in enumerate(self.memory_rows):
            block.setVisible(rank < count)
            if rank >= count:
                continue
            try:
                logical = int(ids[rank].strip())
            except (ValueError, IndexError):
                logical = rank
            gpu = self.gpu_memory.get(logical)
            estimate = estimates[rank] if rank < len(estimates) else None
            amount = (f"≥ {estimate.low / GIB:.2f} GiB" if estimate and estimate.auto else
                      f"{estimate.low / GIB:.2f}–{estimate.high / GIB:.2f} GiB" if estimate else '—')
            heading.setText(f"GPU {logical} · {amount}")
            detail.setText(f"当前 {gpu.used / GIB:.2f} / {gpu.total / GIB:.2f} GiB · 空闲 {gpu.free / GIB:.2f}" if gpu else
                           "正在读取设备…" if self.gpu_query_running else "设备容量不可读")
            bar.setRange(0, 1000)
            projected = gpu.used if self.process is not None and gpu else estimate.high if estimate else 0
            bar.setValue(min(1000, int(projected / gpu.total * 1000)) if gpu and gpu.total else 0)
            warning = ''
            if gpu and estimate and self.process is None:
                if estimate.low > gpu.free:
                    warning = ' · 预计空闲显存不足'
                elif estimate.high > gpu.free:
                    warning = ' · 显存余量较小'
            if warning:
                detail.setText(detail.text() + warning)
            if estimate:
                block.setToolTip(
                    f"权重约 {estimate.weights / GIB:.3f} GiB\nKV 约 {estimate.kv / GIB:.3f} GiB\n"
                    f"持久状态约 {estimate.state / GIB:.3f} GiB\n"
                    f"工作区约 {estimate.workspace_low / GIB:.3f}–{estimate.workspace_high / GIB:.3f} GiB\n"
                    f"Graph 预算约 {estimate.graph / GIB:.3f} GiB\nCUDA 及分配余量 0.25–0.75 GiB\n"
                    "目录字节与固定形状估算，不是 Engine 的精确分配计划。")
            else:
                block.setToolTip('')
        self.memory_note.setText(note)

    def _query_resources(self) -> None:
        if self.gpu_query_running:
            return
        self.gpu_query_running = True
        def worker():
            try:
                self.resource_results.put(query_gpu_memory())
            except (OSError, ValueError, subprocess.SubprocessError):
                # Missing drivers/nvidia-smi must not prevent configuring the launcher.
                self.resource_results.put({})
        threading.Thread(target=worker, daemon=True).start()

    def _drain_resource_results(self) -> None:
        try:
            self.gpu_memory = self.resource_results.get_nowait()
        except queue.Empty:
            return
        self.gpu_query_running = False
        self._refresh_memory_estimate()

    def _render_load_progress(self) -> None:
        stage = self.load_stage
        self.load_label.setText(stage.label + (f" · {stage.percent:.1f}%" if stage.percent is not None and stage.label == '上传权重' else ''))
        if stage.percent is None:
            self.load_progress.setRange(0, 0)
        else:
            self.load_progress.setRange(0, 1000)
            self.load_progress.setValue(int(stage.percent * 10))

    def _set_configuration_lock(self, locked: bool) -> None:
        self.config_selector.setEnabled(not locked)
        self.config_import.setEnabled(not locked)
        selected = self.config_selector.currentIndex() >= 0
        self.config_load.setEnabled(not locked and selected)
        self.config_delete.setEnabled(not locked and selected)

    def _set_runtime_lock(self, locked: bool) -> None:
        for widget in (*self._option_widgets().values(), self.api_key, self.kv_capacity_toggle, *self.path_buttons):
            widget.setEnabled(not locked)
        if not locked:
            self._update_feature_controls()
            self.draft_tokens.setEnabled(self.spec.currentText() != 'none')
            self.lm_head_draft.setEnabled(self.spec.currentText() != 'none')

    def _vision_changed(self, enabled: bool) -> None:
        if enabled:
            self.volta_fast_decode.setChecked(False)
            # Vision starts with cold requests; users may enable prefix reuse.
            self.prefix_reuse.setChecked(False)
            self.ram_kv.setChecked(False)
        self._update_feature_controls()

    def _update_feature_controls(self, *_args) -> None:
        self.vision_device.setEnabled(self.vision.isChecked())
        self.vision_max_tokens.setEnabled(self.vision.isChecked())
        self.pipeline_chunks.setEnabled(self.pp.currentText() == "2")
        self.ram_window.setEnabled(self.ram_kv.isChecked())
        self.ram_budget.setEnabled(self.ram_kv.isChecked())
        self.kv_capacity.setEnabled(not self.ram_kv.isChecked())
        self.kv_capacity_toggle.setEnabled(not self.ram_kv.isChecked())

    def _toggle_kv_capacity(self, expanded: bool) -> None:
        self.kv_capacity_panel.setVisible(expanded)
        self.kv_capacity_toggle.setArrowType(
            (Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow) if QT6 else
            (Qt.DownArrow if expanded else Qt.RightArrow))
        if not expanded:
            self.kv_capacity.clear()
        self._refresh_command()

    def _spec_changed(self, spec: str) -> None:
        self.draft_tokens.setEnabled(spec != "none")
        self.lm_head_draft.setEnabled(spec != "none")
        if spec == "none" or spec == "dflash":
            self.lm_head_draft.setChecked(False)

    def start_server(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "参数检查失败", error)
            return
        if self.process is not None:
            return
        self.server_starts += 1
        self.cache_monitor = CacheMonitor(
            Path(self.request_log_directory.name) / f"requests-{self.server_starts}.jsonl"
        )
        self.cache_latest.setText("最近请求：—")
        self.cache_total.setText("本次启动累计：—")
        self.draft_live.setText("实时草稿接受率：—")
        self.throughput_live.setText("实时吞吐：—")
        self.request_last.setText("上次请求：—")
        command = self._command()
        self._save_settings()
        self._set_command(command)
        self.log.clear()
        self._append_log("$ " + self.command.text())
        self.output_decoder.reset()
        self.output_tail = ''
        self.load_stage = LoadStage()
        self.load_stage.label, self.load_stage.percent = '读取模型目录', None
        self._render_load_progress()
        self.server_listening = False
        self.server_stopping = False
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo))
        mode = QProcess.ProcessChannelMode.MergedChannels if QT6 else QProcess.MergedChannels
        self.process.setProcessChannelMode(mode)
        environment = os.environ.copy()
        library_parts = [self.repo / "build/_deps/install/lib", Path("/usr/local/cuda-12.8/lib64")]
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(
            [str(path) for path in library_parts if path.is_dir()] + ([existing] if existing else [])
        )
        process_environment = self.process.processEnvironment()
        for key, value in environment.items():
            process_environment.insert(key, value)
        self.process.setProcessEnvironment(process_environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)
        self.process.start(str(command[0]), command[1:])
        self.status.setText("启动中…")
        self.status.setStyleSheet("color: #b06a00; font-weight: bold")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.restart_button.setEnabled(True)
        self._set_runtime_lock(True)
        self._set_configuration_lock(True)

    def stop_server(self) -> None:
        if self.process is None:
            return
        self._append_log("stopping server…")
        self.server_stopping = True
        self.offload_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.status.setText("停止中…")
        self.load_stage.label, self.load_stage.percent = '停止中', None
        self._render_load_progress()
        self.stop_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        process = self.process
        self.process.terminate()
        QTimer.singleShot(5000, lambda: self._kill_if_running(process))

    def restart_server(self) -> None:
        if self.process is not None:
            self.restart_requested = True
            self.stop_server()

    def _kill_if_running(self, process: QProcess) -> None:
        if self.process is not process:
            return
        not_running = QProcess.ProcessState.NotRunning if QT6 else QProcess.NotRunning
        if self.process.state() != not_running:
            self._append_log("server did not exit after 5s; terminating child")
            self.process.kill()

    def _read_output(self) -> None:
        if self.process is not None:
            data = self.output_decoder.decode(bytes(self.process.readAllStandardOutput()))
            if data:
                if self.load_stage.feed(data):
                    if not self.server_stopping:
                        self._render_load_progress()
                        if self.load_stage.ready:
                            self.status.setText('运行中')
                            self.status.setStyleSheet('color: #188038; font-weight: bold')
                self.output_tail = (self.output_tail + data)[-4096:]
                if 'listening on http' in self.output_tail:
                    self.server_listening = True
                self._append_log(data.rstrip("\n"))

    def _process_finished(self, exit_code: int, _status) -> None:
        self._read_output()
        self._poll_cache()
        self._append_log(f"server exited with code {exit_code}")
        process = self.process
        self.process = None
        self.model_control = None
        self.model_view_state = 'unknown'
        self.offload_button.setEnabled(False)
        self.load_button.setEnabled(False)
        if process:
            process.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.status.setText("已停止")
        self.status.setStyleSheet("color: #777; font-weight: bold")
        self._set_runtime_lock(False)
        self._set_configuration_lock(False)
        self.load_stage.label = '启动失败' if exit_code != 0 and not self.server_stopping else '已停止'
        self.load_stage.percent = 0
        self._render_load_progress()
        self.server_listening = False
        self.server_stopping = False
        if self.restart_requested:
            self.restart_requested = False
            QTimer.singleShot(0, self.start_server)

    def _process_error(self, error) -> None:
        self._append_log(f"process error: {self.process.errorString() if self.process else error}")
        failed = QProcess.ProcessError.FailedToStart if QT6 else QProcess.FailedToStart
        if error == failed:
            self._process_finished(-1, None)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _poll_cache(self) -> None:
        try:
            changed = self.cache_monitor.poll()
        except OSError as error:
            self.cache_latest.setText(f"缓存统计读取失败：{error}")
            return
        if not changed:
            return
        stats = self.cache_monitor
        if stats.live_interval_seconds:
            self.throughput_live.setText(
                f"实时吞吐：Prefill {stats.live_prefill_rate:,.1f} tok/s · "
                f"已提交 Decode {stats.live_decode_rate:,.1f} tok/s · 最近 {stats.live_interval_seconds:.1f}s")
            drafted = stats.live_drafted_tokens
            accepted = stats.live_accepted_tokens
            value = f"{accepted / drafted:.1%} · {accepted:,} / {drafted:,}" if drafted else "—（本区间无草稿）"
            self.draft_live.setText(f"实时草稿接受率：{value} · 最近 {stats.live_interval_seconds:.1f}s")
        if not stats.requests:
            return
        self.cache_latest.setText(
            f"最近请求：{stats.latest_rate:.1%} · "
            f"{stats.latest_cached_tokens:,} / {stats.latest_prompt_tokens:,} tokens"
        )
        self.cache_total.setText(
            f"本次启动累计：{stats.total_rate:.1%} · "
            f"{stats.cached_tokens:,} / {stats.prompt_tokens:,} tokens · {stats.requests:,} 次请求"
        )
        def phase(name, tokens, seconds):
            if seconds is None:
                return f"{name} —"
            speed = ("全缓存" if name == "Prefill" and tokens == 0 else
                     f"{tokens / seconds:,.1f} tok/s" if tokens > 0 and seconds > 0 else "—")
            return f"{name} {speed}（{tokens:,} tokens / {seconds:.3f}s）"
        drafted = stats.latest_drafted_tokens
        accepted = stats.latest_accepted_tokens
        draft = f"{accepted / drafted:.1%} · {accepted:,} / {drafted:,}" if drafted else "—（无草稿）"
        self.request_last.setText(
            "上次请求：" + phase("Prefill", stats.latest_prefill_tokens, stats.latest_prefill_seconds) + "\n" +
            phase("Decode", stats.latest_decode_tokens, stats.latest_decode_seconds) +
            f"\n草稿接受率 {draft}"
        )

    def _poll_health(self) -> None:
        if self.health_in_flight:
            return
        if self.process is None and not self.host.text().strip():
            return
        self.health_in_flight = True
        url = self._health_url()
        process = self.process

        def worker() -> None:
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(url, timeout=1.5) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self.health_results.put((url, process, json.loads(body) == {"status": "ok"}, body))
            except (OSError, ValueError) as error:
                self.health_results.put((url, process, False, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _model_action(self, action: str) -> None:
        if self.model_control and not self.server_stopping:
            self.model_control.action(action)
            self._poll_model()

    def _poll_model(self) -> None:
        if self.process is None or self.server_stopping or not self.server_listening:
            return
        if self.model_control is None:
            self.model_control = ModelControl(self._health_url().removesuffix('/health'), self.api_key.text())
        model = self.model_control
        model.poll()
        if model.state != self.model_view_state:
            self.model_view_state = model.state
            self._query_resources()
            if model.state == 'ram':
                self.throughput_live.setText('实时吞吐：模型已卸载到内存')
                self.draft_live.setText('实时草稿接受率：暂停')
            elif model.state == 'gpu':
                self.throughput_live.setText('实时吞吐：等待请求')
                self.draft_live.setText('实时草稿接受率：—')
        self.offload_button.setEnabled(model.state == 'gpu' and not model.busy and model.active_requests == 0)
        self.load_button.setEnabled(model.state == 'ram' and not model.busy)
        if model.state != 'unknown':
            self.status.setText(model.label)
            self.status.setStyleSheet('color: #188038; font-weight: bold' if model.state == 'gpu'
                                     else 'color: #b06a00; font-weight: bold')
            if model.busy or model.state in ('offloading', 'loading'):
                phases = {'retain_host': '保留权重到内存', 'release_gpu': '释放显存',
                          'weights': '上传权重', 'load_host': '从内存读取模型'}
                self.load_stage.label = phases.get(model.phase, model.phase or model.label)
                self.load_stage.percent = model.done / model.total * 100 if model.total else None
            else:
                self.load_stage.label = '已卸载到内存 · KV/前缀缓存已清空' if model.state == 'ram' else '就绪'
                self.load_stage.percent = 100
            self._render_load_progress()
        if model.error:
            error, model.error = model.error, ''
            self._append_log('model residency: ' + error)
            QMessageBox.warning(self, '模型切换失败', error)

    def _check_health(self, force: bool = False) -> None:
        if force:
            self._poll_health()

    def _drain_health_results(self) -> None:
        try:
            url, process, ok, detail = self.health_results.get_nowait()
        except queue.Empty:
            return
        self.health_in_flight = False
        if url != self._health_url() or process is not self.process or self.server_stopping:
            return
        if self.process is not None and self.model_control and self.model_control.state != 'unknown':
            return
        if self.process is not None and not self.server_listening:
            self.status.setText("模型加载中…（等待 API 监听）")
            self.status.setStyleSheet("color: #b06a00; font-weight: bold")
        elif ok:
            self.status.setText("运行中 · health OK" if self.process else "已有 API · 非本窗口启动")
            self.status.setStyleSheet("color: #188038; font-weight: bold")
        elif self.process is not None:
            self.status.setText("进程运行中 · 等待 health")
            self.status.setStyleSheet("color: #b06a00; font-weight: bold")
        else:
            self.status.setText("未启动 · health 不可用")
            self.status.setStyleSheet("color: #777; font-weight: bold")

    def _health_url(self) -> str:
        host = self.host.text().strip()
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port.value()}/health"

    def open_api(self) -> None:
        QDesktopServices.openUrl(QUrl(self._health_url()))

    def copy_command(self) -> None:
        command = shlex.split(self.command.text())
        # The GUI owns this temporary path; a terminal command must outlive the window.
        if "--request-log-jsonl" in command:
            index = command.index("--request-log-jsonl")
            del command[index:index + 2]
        QApplication.clipboard().setText(shlex.join(command))
        self.status.setText("命令已复制")

    def save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存服务日志", str(self.repo / "ninfer-server.log"), "Log (*.log);;All files (*)")
        if path:
            try:
                Path(path).write_text(self.log.toPlainText() + '\n', encoding='utf-8')
            except OSError as error:
                QMessageBox.warning(self, "日志保存失败", str(error))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._save_settings()
        if self.process is not None:
            yes = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
            no = QMessageBox.StandardButton.No if QT6 else QMessageBox.No
            choice = QMessageBox.question(self, "退出控制台", "关闭窗口将停止此窗口启动的 API。继续？", yes | no, no)
            if choice != yes:
                event.ignore()
                return
            self.restart_requested = False
            process = self.process
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
        self.health_timer.stop()
        self.queue_timer.stop()
        self.estimate_timer.stop()
        self.resource_timer.stop()
        self.request_log_directory.cleanup()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("NINFER_GUI_REPO", ROOT)))
    parser.add_argument("--binary", default="")
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    window = NInferGui(args.repo.expanduser().resolve(), args.binary, args.artifact)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
