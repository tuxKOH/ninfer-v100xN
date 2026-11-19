"""Owned-server cache display: true reused tokens, incremental logs and weighted rates."""

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.v100.gui_cache import CacheMonitor


class CacheDisplay(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "requests.jsonl"
        self.monitor = CacheMonitor(self.path)

    def append(self, prompt, cached, event="request_done"):
        record = {"event": event, "result": {"prompt_tokens": prompt,
                                             "prefix_cache_hit_tokens": cached}}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def test_cold_partial_full_and_token_weighted_total(self):
        self.assertFalse(self.monitor.poll())  # startup before log creation
        self.append(100, 0)
        self.assertTrue(self.monitor.poll())
        self.assertEqual(self.monitor.latest_rate, 0)
        self.append(1000, 900)
        self.append(100, 100)
        self.assertTrue(self.monitor.poll())
        self.assertEqual(self.monitor.latest_rate, 1)
        self.assertAlmostEqual(self.monitor.total_rate, 1000 / 1200)
        self.assertEqual(self.monitor.requests, 3)
        self.assertEqual((self.monitor.latest_prompt_tokens, self.monitor.latest_cached_tokens),
                         (100, 100))
        self.assertFalse(self.monitor.poll())
        self.assertEqual(self.monitor.requests, 3)  # polling must not double-count

    def test_partial_line_and_unicode_are_not_lost(self):
        line = json.dumps({"event": "request_done", "note": "缓存",
                           "result": {"prompt_tokens": 80, "prefix_cache_hit_tokens": 20}},
                          ensure_ascii=False).encode()
        split = line.index("存".encode()) + 1  # inside a UTF-8 codepoint
        self.path.write_bytes(line[:split])
        self.assertFalse(self.monitor.poll())
        with self.path.open("ab") as stream:
            stream.write(line[split:] + b"\n")
        self.assertTrue(self.monitor.poll())
        self.assertEqual(self.monitor.latest_rate, 0.25)
        self.assertFalse(self.monitor.poll())

    def test_ignore_nonterminal_invalid_and_missing_metrics(self):
        for event in ("request_start", "request_error", "throughput", "server_start"):
            self.append(100, 50, event)
        for prompt, cached in ((100, 101), (100, -1), (-1, 0), (100, None), (True, 0)):
            self.append(prompt, cached)
        with self.path.open("a") as stream:
            stream.write('not JSON\n[]\n{"event":"request_done"}\n')
        self.assertFalse(self.monitor.poll())
        self.assertEqual(self.monitor.requests, 0)
        self.append(0, 0)
        self.assertTrue(self.monitor.poll())
        self.assertEqual((self.monitor.total_rate, self.monitor.latest_rate), (0, 0))
        self.append(100, 80)
        self.assertTrue(self.monitor.poll())
        self.assertEqual(self.monitor.latest_rate, 0.8)

    def test_restart_has_separate_log_and_counters(self):
        self.append(100, 100)
        self.monitor.poll()
        restarted = CacheMonitor(Path(self.directory.name) / "requests-2.jsonl")
        self.assertFalse(restarted.poll())
        self.assertEqual((restarted.requests, restarted.total_rate), (0, 0))

    def test_live_acceptance_and_last_completed_request_are_independent(self):
        records = [
            {"event": "throughput", "interval_seconds": 1.25,
             "speculative": {"drafted_tokens": 12, "accepted_tokens": 9}},
            {"event": "request_done", "result": {"prompt_tokens": 1000,
             "prefix_cache_hit_tokens": 800, "completion_tokens": 101},
             "timings_seconds": {"prefill": 0.5, "decode": 2.0},
             "speculative": {"drafted_tokens": 120, "accepted_tokens": 80}},
        ]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.assertTrue(self.monitor.poll())
        self.assertEqual((self.monitor.live_drafted_tokens, self.monitor.live_accepted_tokens),
                         (12, 9))
        self.assertEqual(self.monitor.live_interval_seconds, 1.25)
        self.assertEqual((self.monitor.latest_prefill_tokens, self.monitor.latest_decode_tokens),
                         (200, 100))  # First output is produced by prefill, not decode.
        self.assertEqual((self.monitor.latest_prefill_seconds, self.monitor.latest_decode_seconds),
                         (0.5, 2.0))
        self.assertEqual((self.monitor.latest_drafted_tokens, self.monitor.latest_accepted_tokens),
                         (120, 80))
        with self.path.open("a") as stream:
            stream.write(json.dumps({"event": "throughput", "interval_seconds": 1.0,
                "speculative": {"drafted_tokens": 0, "accepted_tokens": 0}}) + "\n")
        self.monitor.poll()
        self.assertEqual(self.monitor.live_drafted_tokens, 0)
        self.assertEqual(self.monitor.latest_decode_tokens, 100)
        self.assertEqual(self.monitor.requests, 1)

    def test_missing_or_invalid_phase_durations_are_unavailable(self):
        for invalid in (None, -1, True, float("nan"), float("inf")):
            record = {"event": "request_done", "result": {"prompt_tokens": 10,
                "prefix_cache_hit_tokens": 10, "completion_tokens": 1},
                "timings_seconds": {"prefill": invalid}}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.monitor.poll()
            self.assertIsNone(self.monitor.latest_prefill_seconds)
            self.assertIsNone(self.monitor.latest_decode_seconds)
            self.assertEqual(self.monitor.latest_decode_tokens, 0)


@unittest.skipUnless(importlib.util.find_spec("PyQt6") or importlib.util.find_spec("PyQt5"),
                     "PyQt6/PyQt5 is needed for the GUI lifecycle check")
class GuiCacheDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from tools.v100.ninfer_gui import QApplication, NInferGui, QSettings, QT6
        cls.app = QApplication.instance() or QApplication([])
        cls.Gui = NInferGui
        cls.Settings = QSettings
        cls.ini = QSettings.Format.IniFormat if QT6 else QSettings.IniFormat

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        artifact = root / "model.ninfer"
        artifact.touch()
        self.gui = self.Gui(root, sys.executable, str(artifact),
                            settings=self.Settings(str(root / "settings.ini"), self.ini))
        self.gui.health_timer.stop()
        self.gui.queue_timer.stop()
        self.addCleanup(self.close_gui)

    def close_gui(self):
        if self.gui.process is not None:
            self.gui.process.kill()
            self.gui.process.waitForFinished(2000)
        self.gui.close()
        self.app.processEvents()

    def test_display_uses_real_log_counts(self):
        command = self.gui._command()
        self.assertEqual(command[command.index("--request-log-jsonl") + 1],
                         str(self.gui.cache_monitor.path))
        records = [{"event": "request_done", "result": {"prompt_tokens": prompt,
                    "prefix_cache_hit_tokens": cached}}
                   for prompt, cached in ((100, 0), (1000, 900))]
        self.gui.cache_monitor.path.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.gui._poll_cache()
        self.assertIn("90.0%", self.gui.cache_latest.text())
        self.assertIn("900 / 1,000 tokens", self.gui.cache_latest.text())
        self.assertIn("81.8%", self.gui.cache_total.text())
        self.assertIn("2 次请求", self.gui.cache_total.text())
        self.gui._poll_cache()
        self.assertEqual(self.gui.cache_monitor.requests, 2)

    def test_model_residency_controls_and_progress(self):
        from tools.v100.launcher_model import ModelControl
        from unittest.mock import Mock
        self.assertFalse(self.gui.offload_button.isEnabled())
        self.assertFalse(self.gui.load_button.isEnabled())
        process = object()
        self.gui.process = process
        self.gui.server_listening = True
        model = ModelControl('http://127.0.0.1:1')
        model.poll = Mock()
        self.gui.model_control = model
        try:
            for state, unload, load in [('gpu', True, False), ('ram', False, True),
                                        ('offloading', False, False), ('loading', False, False)]:
                model.state = state
                model.retained_host_bytes = 2 * 1024**3
                model.done, model.total = 40, 100
                self.gui._poll_model()
                self.assertEqual(self.gui.offload_button.isEnabled(), unload)
                self.assertEqual(self.gui.load_button.isEnabled(), load)
                self.assertIn('2.00 GiB', self.gui.status.text())
                if state in ('loading', 'offloading'):
                    self.assertEqual(self.gui.load_progress.value(), 400)
            model.state = 'gpu'
            model.active_requests = 1
            self.gui._poll_model()
            self.assertFalse(self.gui.offload_button.isEnabled())
            self.gui.server_stopping = True
            self.gui._poll_model()
            self.assertFalse(self.gui.offload_button.isEnabled())
        finally:
            self.gui.process = None

    def test_copied_command_does_not_depend_on_gui_temporary_log(self):
        self.gui.api_key.setText("private-key-must-not-be-copied")
        self.gui.copy_command()
        copied = self.app.clipboard().text()
        self.assertNotIn("--request-log-jsonl", copied)
        self.assertNotIn("private-key-must-not-be-copied", copied)
        self.assertIn("<API_KEY>", copied)

    def test_experimental_attention_flag_preview_and_copy(self):
        flag = "--volta-long-attention"
        self.assertFalse(self.gui.volta_long_attention.isChecked())
        self.assertNotIn(flag, self.gui._command())
        before = self.gui._command()
        self.gui.volta_long_attention.setChecked(True)
        self.assertIsNone(self.gui._validate())
        self.assertEqual(self.gui._command(), before + [flag])
        self.assertIn(flag, self.gui.command.text())
        self.gui.copy_command()
        self.assertIn(flag, self.app.clipboard().text())
        self.gui.volta_long_attention.setChecked(False)
        self.assertEqual(self.gui._command(), before)
        self.assertNotIn(flag, self.gui.command.text())

    def test_history_budget_configuration_and_concurrent_long_attention(self):
        self.assertEqual(self.gui.prefix_cache_tokens.value(), 384000)
        for budget in (0, 123456):
            self.gui.prefix_cache_tokens.setValue(budget)
            settings = self.gui.configuration()
            self.gui.prefix_cache_tokens.setValue(384000)
            self.gui.apply_configuration(settings)
            self.assertEqual(self.gui.prefix_cache_tokens.value(), budget)
            command = self.gui._command()
            self.assertEqual(command[command.index('--prefix-cache-tokens') + 1], str(budget))
        self.gui.volta_long_attention.setChecked(True)
        for tp in (2, 4):
            self.gui.tp.setCurrentText(str(tp))
            self.gui.devices.setText(",".join(map(str, range(tp))))
            for concurrency in (1, 2, 4, 8):
                self.gui.max_concurrency.setValue(concurrency)
                self.assertIsNone(self.gui._validate())

    def test_experimental_attention_rejects_incompatible_configuration(self):
        initial = self.gui.configuration()
        for name, value in (("tp", "1"), ("kv_dtype", "bf16")):
            with self.subTest(name=name, value=value):
                self.gui.apply_configuration(initial)
                self.gui.volta_long_attention.setChecked(True)
                self.gui.max_concurrency.setValue(1)
                widget = getattr(self.gui, name)
                if name == "max_concurrency":
                    widget.setValue(value)
                else:
                    widget.setCurrentText(value)
                if name == "tp":
                    self.gui.devices.setText(",".join(str(i) for i in range(int(value))))
                self.assertIn("实验性长上下文注意力要求", self.gui._validate())
        self.gui.apply_configuration(initial)
        self.gui.spec.setCurrentText("none")
        self.assertIsNone(self.gui._validate())

    def test_dflash_tp2_accepts_both_experimental_profiles(self):
        self.gui.spec.setCurrentText("dflash")
        self.gui.draft_tokens.setValue(7)
        self.gui.volta_long_attention.setChecked(True)
        self.gui.volta_fast_decode.setChecked(True)
        self.assertIsNone(self.gui._validate())
        self.assertIn('--volta-long-attention', self.gui._command())
        self.assertIn('--volta-fast-decode', self.gui._command())
        self.gui.tp.setCurrentText("4")
        self.gui.devices.setText("0,1,2,3")
        self.assertIsNotNone(self.gui._validate())

    def test_named_configuration_replaces_previous_execution_and_sampling(self):
        initial = self.gui.configuration()
        self.gui.save_named_configuration('文本')
        variants = (
            dict(initial, vision=True, prefix_reuse=False, max_context=98304,
                 vision_max_tokens=4096, temperature='0.2', kv_capacity='auto'),
            dict(initial, volta_fast_decode=True, volta_long_attention=True,
                 greedy=True, prefix_reuse=False, temperature='0.4'),
            dict(initial, ram_kv=True, kv_capacity='300000'),
        )
        for values in variants:
            with self.subTest(values=values):
                self.gui.apply_configuration(values)
                self.gui.load_named_configuration('文本')
                self.assertEqual(self.gui.configuration(), initial)
                self.assertFalse(self.gui.kv_capacity_toggle.isChecked())
                self.assertIsNone(self.gui._validate())

    def test_vision_manual_prefix_choice_survives_reload(self):
        self.gui.vision.setChecked(True)
        self.gui.prefix_reuse.setChecked(True)
        self.gui._save_settings()
        restored = self.Gui(self.gui.repo, settings=self.gui.settings)
        try:
            self.assertTrue(restored.vision.isChecked())
            self.assertTrue(restored.prefix_reuse.isChecked())
            self.assertNotIn('--no-prefix-reuse', restored._command())
        finally:
            restored.close()

    def test_named_configurations_restore_all_fields_and_keep_api_key_out(self):
        existing = set(self.gui.config_library.entries())
        self.gui.vision.setChecked(True)
        self.gui.prefix_reuse.setChecked(True)
        self.gui.top_p.setText('0.83')
        self.gui.api_key.setText('session-secret')
        expected = self.gui.configuration()
        self.gui.save_named_configuration('视觉 / 个人')
        self.gui.vision.setChecked(False)
        self.gui.volta_fast_decode.setChecked(True)
        self.gui.volta_long_attention.setChecked(True)
        self.gui.save_named_configuration('文本实验')
        self.gui.load_named_configuration('视觉 / 个人')
        self.assertEqual(self.gui.configuration(), expected)
        self.assertTrue(self.gui.prefix_reuse.isChecked())
        self.assertEqual(self.gui.api_key.text(), 'session-secret')
        self.assertNotIn('session-secret', self.gui.config_library.path.read_text())
        self.assertEqual(set(self.gui.config_library.entries()), existing | {'视觉 / 个人', '文本实验'})
        restored = self.Gui(self.gui.repo, settings=self.gui.settings)
        try:
            self.assertEqual(restored.configuration(), expected)
            self.assertEqual(restored.config_selector.currentText(), '视觉 / 个人')
        finally:
            restored.close()

    def test_default_configurations_initialize_once_and_keep_user_changes(self):
        defaults = self.gui.config_library.entries()
        self.assertEqual(len(defaults), 2)
        for values in defaults.values():
            self.gui._validate_configuration(values)
            self.assertEqual(values['draft_tokens'], 4)
            self.assertEqual(values['kv_capacity'], '')
            self.assertFalse(values['volta_fast_decode'])
        vision_name = next(name for name, values in defaults.items() if values['vision'])
        text_name = next(name for name, values in defaults.items() if not values['vision'])
        self.assertEqual(defaults[vision_name]['max_context'], 248000)
        self.assertFalse(defaults[vision_name]['volta_long_attention'])
        self.assertEqual(defaults[text_name]['max_context'], 262144)
        self.assertTrue(defaults[text_name]['volta_long_attention'])
        self.assertEqual(self.gui.config_selector.currentIndex(), -1)
        self.assertFalse(self.gui.config_load.isEnabled())
        self.gui.config_selector.setCurrentText(vision_name)
        self.assertTrue(self.gui.config_load.isEnabled())
        self.gui.config_library.delete(vision_name)
        changed = dict(defaults[text_name], port=6201, binary=self.gui.binary.text(),
                       artifact=self.gui.artifact.text())
        self.gui.config_library.save(text_name, changed)
        self.gui._reload_configurations()
        self.assertNotIn(vision_name, self.gui.config_library.entries())
        self.gui.load_named_configuration(text_name)
        self.assertEqual(self.gui.port.value(), 6201)
        self.assertIsNone(self.gui._validate())
        self.gui.config_library.delete(text_name)
        self.gui._reload_configurations()
        self.assertEqual(self.gui.config_library.entries(), {})

    def test_configuration_export_import_and_invalid_import_are_atomic(self):
        from tools.v100.ninfer_gui import QFileDialog, QMessageBox
        exported = Path(self.directory.name) / 'portable.json'
        self.gui.vision.setChecked(True)
        self.gui.prefix_reuse.setChecked(True)
        expected = self.gui.configuration()
        self.gui.api_key.setText('not-on-disk')
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(exported), '')):
            self.gui.export_configuration()
        self.assertNotIn('not-on-disk', exported.read_text())
        self.gui.vision.setChecked(False)
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(exported), '')):
            self.gui.import_configuration()
        self.assertEqual(self.gui.configuration(), expected)
        invalid = json.loads(exported.read_text())
        invalid['settings']['max_context'] = -10
        exported.write_text(json.dumps(invalid))
        bank_before = self.gui.config_library.path.read_bytes()
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(exported), '')), \
                patch.object(QMessageBox, 'warning') as warning:
            self.gui.import_configuration()
        warning.assert_called_once()
        self.assertEqual(self.gui.configuration(), expected)
        self.assertEqual(self.gui.config_library.path.read_bytes(), bank_before)

    def test_load_progress_widget_tracks_real_output_until_ready(self):
        self.gui.load_stage.feed('[INFO] load weights 35.25% 1 GiB / 3 GiB\n')
        self.gui._render_load_progress()
        self.assertEqual(self.gui.load_progress.value(), 352)
        self.assertIn('35.2%', self.gui.load_label.text())
        self.gui.load_stage.feed('[INFO] load weights 100.00% 3 GiB / 3 GiB\n')
        self.gui._render_load_progress()
        self.assertEqual(self.gui.load_progress.maximum(), 0)
        self.assertIn('初始化', self.gui.load_label.text())
        self.gui.load_stage.feed('[INFO] listening on http://127.0.0.1:6200\n')
        self.gui._render_load_progress()
        self.assertEqual(self.gui.load_progress.value(), 1000)
        self.assertEqual(self.gui.load_label.text(), '就绪')

    def test_memory_estimate_refreshes_from_settings_and_handles_missing_model(self):
        from tools.v100.launcher_resources import GpuMemory, MemoryEstimate, GIB
        self.gui.gpu_memory = {0: GpuMemory('0', 'GPU-test', 'V100', 16 * GIB, 2 * GIB, 14 * GIB)}
        estimate = MemoryEstimate(8 * GIB, 3 * GIB, GIB // 4, GIB, 2 * GIB, 0, False)
        with patch('tools.v100.ninfer_gui.estimate_memory', return_value=[estimate, estimate]) as calculate:
            self.gui._refresh_memory_estimate()
            self.assertEqual(calculate.call_args.args[1].context, 180000)
            self.assertIn('12.50', self.gui.memory_rows[0][1].text())
            self.assertIn('14.00', self.gui.memory_rows[0][1].text())
            self.assertGreater(self.gui.memory_rows[0][3].value(), 0)
            self.gui.max_context.setValue(98304)
            self.gui._refresh_memory_estimate()
            self.assertEqual(calculate.call_args.args[1].context, 98304)
        self.gui.artifact.setText('missing.ninfer')
        self.gui._refresh_memory_estimate()
        self.assertIn('不可用', self.gui.memory_note.text())

    def test_pipeline_ram_kv_and_exact_integer_sampling(self):
        self.gui.tp.setCurrentText('1')
        self.gui.pp.setCurrentText('2')
        self.gui.ram_kv.setChecked(True)
        self.gui.ram_window.setValue(32768)
        self.gui.seed.setText('18446744073709551615')
        self.assertIsNone(self.gui._validate())
        command = self.gui._command()
        for option, value in (('--pp', '2'), ('--pipeline-chunks', '8'),
                              ('--kv-capacity', '32768'), ('--ram-kv-window', '32768'),
                              ('--ram-kv-budget-bytes', '32000000000')):
            self.assertEqual(command[command.index(option) + 1], value)
        self.gui.seed.setText('18446744073709551616')
        self.assertIn('seed', self.gui._validate())
        self.gui.seed.clear()
        self.gui.ram_window.setValue(8192)
        self.assertIn('RAM-KV', self.gui._validate())
        self.gui.ram_window.setValue(32768)
        self.gui.vision.setChecked(True)
        self.assertIn('PP2', self.gui._validate())

    def test_kv_capacity_tracks_context_and_accepts_each_supported_mode(self):
        self.gui.max_context.setValue(262144)
        self.assertIsNone(self.gui._validate())
        self.assertNotIn('--kv-capacity', self.gui._command())
        self.gui.kv_capacity_toggle.setChecked(True)
        for capacity in ('auto', '262144', '300000'):
            with self.subTest(capacity=capacity):
                self.gui.kv_capacity.setText(capacity)
                self.assertIsNone(self.gui._validate())
                command = self.gui._command()
                self.assertEqual(command[command.index('--kv-capacity') + 1], capacity)
        self.gui.kv_capacity.setText('256000')
        error = self.gui._validate()
        self.assertIn('256000', error)
        self.assertIn('262144', error)
        self.gui.max_context.setValue(256000)
        self.assertIsNone(self.gui._validate())
        self.gui.save_named_configuration('独立 KV')
        self.gui.kv_capacity_toggle.setChecked(False)
        self.assertTrue(self.gui.kv_capacity_panel.isHidden())
        self.gui.max_context.setValue(262144)
        self.assertIsNone(self.gui._validate())
        self.assertNotIn('--kv-capacity', self.gui._command())
        self.assertEqual(self.gui._memory_options().capacity, '')
        self.assertEqual(self.gui.configuration()['kv_capacity'], '')
        self.gui.save_named_configuration('跟随上下文')
        self.gui.load_named_configuration('独立 KV')
        self.assertTrue(self.gui.kv_capacity_toggle.isChecked())
        self.assertFalse(self.gui.kv_capacity_panel.isHidden())
        self.assertEqual(self.gui.kv_capacity.text(), '256000')
        self.assertIsNone(self.gui._validate())
        self.gui.load_named_configuration('跟随上下文')
        self.assertFalse(self.gui.kv_capacity_toggle.isChecked())
        self.assertNotIn('--kv-capacity', self.gui._command())

    def test_relative_paths_use_repository_and_cuda_ids_are_numeric(self):
        self.gui.artifact.setText('model.ninfer')
        self.assertIsNone(self.gui._validate())
        self.assertEqual(self.gui._command()[1], str(self.gui.repo / 'model.ninfer'))
        self.gui.devices.setText('0,00')
        self.assertIn('设备 ID', self.gui._validate())

    def test_health_waits_for_listening_and_discards_previous_child(self):
        child = object()
        self.gui.process = child
        try:
            url = self.gui._health_url()
            self.gui.health_results.put((url, child, True, 'ok'))
            self.gui._drain_health_results()
            self.assertIn('模型加载中', self.gui.status.text())
            self.gui.server_listening = True
            self.gui.health_results.put((url, child, True, 'ok'))
            self.gui._drain_health_results()
            self.assertIn('运行中', self.gui.status.text())
            self.gui.health_results.put((url, object(), False, 'old child'))
            self.gui._drain_health_results()
            self.assertIn('运行中', self.gui.status.text())
        finally:
            self.gui.process = None

    def test_experimental_attention_survives_settings_reload(self):
        self.gui.volta_long_attention.setChecked(True)
        self.gui._save_settings()
        restored = self.Gui(self.gui.repo, settings=self.gui.settings)
        try:
            self.assertTrue(restored.volta_long_attention.isChecked())
            self.assertIn("--volta-long-attention", restored._command())
            self.assertIsNone(restored._validate())
        finally:
            restored.close()
        self.app.processEvents()

    def test_vision_on_off_command_settings_and_experiment_exclusion(self):
        self.assertEqual(self.gui.draft_tokens.value(), 4)
        self.assertFalse(self.gui.vision.isChecked())
        self.assertNotIn('--vision', self.gui._command())
        self.gui.volta_fast_decode.setChecked(True)
        self.gui.volta_long_attention.setChecked(True)
        self.gui.vision.setChecked(True)
        self.assertFalse(self.gui.volta_fast_decode.isChecked())
        self.assertTrue(self.gui.volta_long_attention.isChecked())
        self.assertFalse(self.gui.prefix_reuse.isChecked())
        self.gui.vision_max_tokens.setValue(512)
        self.gui.vision_device.setCurrentText("cpu")
        command = self.gui._command()
        self.assertIn('--vision', command)
        self.assertIn('--vision', self.gui.command.text())
        self.assertEqual(command[command.index('--vision-max-tokens') + 1], '512')
        self.assertEqual(command[command.index('--vision-device') + 1], 'cpu')
        self.assertEqual(command[command.index('--pending-timeout-ms') + 1], '600000')
        self.assertIsNone(self.gui._validate())
        self.gui._save_settings()
        restored = self.Gui(self.gui.repo, settings=self.gui.settings)
        try:
            self.assertTrue(restored.vision.isChecked())
            self.assertEqual(restored.vision_max_tokens.value(), 512)
            self.assertEqual(restored.vision_device.currentText(), "cpu")
            self.assertTrue(restored.volta_long_attention.isChecked())
            self.assertIn('--vision', restored._command())
        finally:
            restored.close()
        self.gui.spec.setCurrentText('dflash')
        self.assertIn('视觉', self.gui._validate())
        self.gui.spec.setCurrentText('mtp')
        self.gui.volta_fast_decode.setChecked(True)
        self.assertIn('纯文本', self.gui._validate())
        self.gui.volta_fast_decode.setChecked(False)
        self.gui.vision.setChecked(False)
        self.assertNotIn('--vision', self.gui._command())
        self.assertNotIn('--vision-max-tokens', self.gui._command())

    def test_live_and_completed_phase_displays(self):
        record = {"event": "throughput", "interval_seconds": 1.0,
                  "throughput_tokens_per_second": {"prefill": 1234.5, "decode": 198.2},
                  "speculative": {"drafted_tokens": 40, "accepted_tokens": 30}}
        self.gui.cache_monitor.path.write_text(json.dumps(record) + "\n")
        self.gui._poll_cache()
        self.assertIn("75.0%", self.gui.draft_live.text())
        self.assertIn('198.2', self.gui.throughput_live.text())
        self.assertEqual(self.gui.request_last.text(), "上次请求：—")
        record = {"event": "request_done", "result": {"prompt_tokens": 1000,
            "prefix_cache_hit_tokens": 800, "completion_tokens": 101},
            "timings_seconds": {"prefill": 0.5, "decode": 2.0},
            "speculative": {"drafted_tokens": 150, "accepted_tokens": 100}}
        with self.gui.cache_monitor.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.gui._poll_cache()
        display = self.gui.request_last.text()
        self.assertIn("Prefill 400.0 tok/s", display)
        self.assertIn("Decode 50.0 tok/s", display)
        self.assertIn("66.7%", display)
        self.assertIn("75.0%", self.gui.draft_live.text())  # Not request-average acceptance.
        record = {"event": "throughput", "interval_seconds": 1.0,
                  "speculative": {"drafted_tokens": 0, "accepted_tokens": 0}}
        with self.gui.cache_monitor.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.gui._poll_cache()
        self.assertIn("本区间无草稿", self.gui.draft_live.text())
        self.assertEqual(display, self.gui.request_last.text())
        record = {"event": "request_done", "result": {"prompt_tokens": 1000,
            "prefix_cache_hit_tokens": 1000, "completion_tokens": 1},
            "timings_seconds": {"prefill": 0.02, "decode": 0}}
        with self.gui.cache_monitor.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.gui._poll_cache()
        self.assertIn("Prefill 全缓存", self.gui.request_last.text())
        self.assertIn("Decode —", self.gui.request_last.text())

    def test_child_exit_is_read_and_restart_resets_statistics(self):
        child = ("import json, pathlib, sys; "
                 "pathlib.Path(sys.argv[1]).write_text(json.dumps({'event': 'request_done', "
                 "'result': {'prompt_tokens': 100, 'prefix_cache_hit_tokens': 75}}) + '\\n')")
        paths = []
        for _ in range(2):
            with patch.object(self.gui, "_command", side_effect=lambda: [
                    sys.executable, "-c", child, str(self.gui.cache_monitor.path)]):
                self.gui.start_server()
                self.assertFalse(self.gui.kv_capacity_toggle.isEnabled())
                self.assertTrue(self.gui.controls.isEnabled())  # Tabs remain readable.
                self.assertFalse(self.gui.artifact.isEnabled())
                self.assertFalse(self.gui.config_import.isEnabled())
                before = self.gui._command()
                with self.assertRaisesRegex(ValueError, '请先停止服务'):
                    self.gui.apply_configuration(self.gui.configuration())
                self.assertEqual(self.gui._command(), before)
                paths.append(self.gui.cache_monitor.path)
                self.assertEqual(self.gui.cache_monitor.requests, 0)
                deadline = time.monotonic() + 5
                while self.gui.process is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
            self.assertIsNone(self.gui.process)
            self.assertTrue(self.gui.kv_capacity_toggle.isEnabled())
            self.assertTrue(self.gui.config_import.isEnabled())
            self.assertEqual(self.gui.cache_monitor.requests, 1)
            self.assertIn("75.0%", self.gui.cache_latest.text())
        self.assertNotEqual(paths[0], paths[1])

    def test_restart_stops_owned_child_before_starting_next(self):
        script = "print('listening on http://127.0.0.1', flush=True); import time; time.sleep(30)"
        with patch.object(self.gui, '_command', return_value=[sys.executable, '-u', '-c', script]):
            self.gui.start_server()
            first = self.gui.process
            deadline = time.monotonic() + 3
            while not self.gui.server_listening and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertTrue(self.gui.server_listening)
            self.assertIn('运行中', self.gui.status.text())
            self.gui.restart_server()
            deadline = time.monotonic() + 5
            while (self.gui.server_starts < 2 or not self.gui.server_listening) and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertEqual(self.gui.server_starts, 2)
            self.assertIsNot(self.gui.process, first)
            self.assertTrue(self.gui.server_listening)
            self.gui.stop_server()
            deadline = time.monotonic() + 3
            while self.gui.process is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIsNone(self.gui.process)
            self.assertFalse(self.gui.restart_requested)


if __name__ == "__main__":
    unittest.main()
