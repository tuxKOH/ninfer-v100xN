"""Launcher command/admission and owned-child lifecycle checks, without CUDA."""
from pathlib import Path
import json
import os
import pty
import select
import fcntl
import struct
import termios
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.v100.ninfer_tui import Launcher, Server, Settings, main


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact = Path(self.directory.name) / 'model.ninfer'
        self.artifact.touch()
        self.settings = Settings(binary=sys.executable, artifact=str(self.artifact),
                                 log=str(Path(self.directory.name) / 'request.jsonl'))

    def test_coding_and_dflash_commands(self):
        for profile, drafts, context in [('quasar-mtp4', '4', '180000'),
                                          ('quasar-long', '4', '180000'),
                                          ('quasar-dflash7', '7', '98304')]:
            with self.subTest(profile=profile):
                self.settings.apply_profile(profile)
                self.settings.artifact = str(self.artifact)
                self.settings.validate()
                command = self.settings.command()
                self.assertEqual(command[command.index('--draft-tokens') + 1], drafts)
                self.assertEqual(command[command.index('--max-context') + 1], context)
                self.assertEqual('--volta-long-attention' in command, profile == 'quasar-long')
                self.assertEqual('--lm-head-draft' in command, profile != 'quasar-dflash7')
                if profile == 'quasar-long':
                    self.assertIn('--greedy', command)
                    self.assertIn('--no-prefix-reuse', command)

    def test_vision_switch_and_experimental_admission(self):
        self.assertNotIn('--vision', self.settings.command())
        self.settings.vision = True
        self.settings.validate()
        command = self.settings.command()
        self.assertIn('--vision', command)
        self.assertEqual(command[command.index('--vision-max-tokens') + 1], '1024')
        self.settings.long_attention = True
        self.settings.vision_device = 'cpu'
        self.settings.validate()
        command = self.settings.command()
        self.assertEqual(command[command.index('--vision-device') + 1], 'cpu')
        self.assertEqual(command[command.index('--pending-timeout-ms') + 1], '600000')
        self.settings.fast_decode = True
        with self.assertRaises(ValueError):
            self.settings.validate()
        self.settings.fast_decode = False
        self.settings.long_attention = False
        self.settings.backend = 'dflash'
        with self.assertRaises(ValueError):
            self.settings.validate()
        self.settings.apply_profile('quasar-vision')
        self.settings.artifact = str(self.artifact)
        self.settings.validate()
        self.assertTrue(self.settings.vision)
        self.assertEqual(self.settings.drafts, 4)
        self.assertFalse(self.settings.prefix_reuse)
        self.settings.apply_profile('quasar-long')
        self.settings.artifact = str(self.artifact)
        self.settings.validate()
        self.assertFalse(self.settings.vision)
        self.assertIn('--volta-fast-decode', self.settings.command())

    def test_history_budget_round_trip_and_long_attention_concurrency(self):
        self.assertEqual(self.settings.prefix_cache_tokens, 384000)
        self.settings.prefix_cache_tokens = 123456
        self.settings.long_attention = True
        for tp in (2, 4):
            self.settings.tp = tp
            self.settings.devices = ','.join(map(str, range(tp)))
            for concurrency in (1, 2, 4, 8):
                self.settings.max_concurrency = concurrency
                self.settings.validate()
            self.assertIn('--volta-long-attention', self.settings.command())
        path = Path(self.directory.name) / 'history.json'
        self.settings.save(path)
        restored = Settings(binary='', artifact='')
        restored.load(path)
        self.assertEqual(restored.prefix_cache_tokens, 123456)
        command = restored.command()
        self.assertEqual(command[command.index('--prefix-cache-tokens') + 1], '123456')
        for budget in (0, 1000000):
            restored.prefix_cache_tokens = budget
            restored.validate()
        restored.prefix_cache_tokens = -1
        with self.assertRaises(ValueError):
            restored.validate()

    def test_plain_decode_omits_speculative_options(self):
        self.settings.backend = 'none'
        command = self.settings.command()
        self.assertNotIn('--spec', command)
        self.assertNotIn('--draft-tokens', command)
        self.assertNotIn('--lm-head-draft', command)

    def test_settings_round_trip_excludes_secret_and_cli_overrides(self):
        path = Path(self.directory.name) / 'settings.json'
        self.settings.vision = True
        self.settings.prefix_reuse = True
        self.settings.top_p = '0.85'
        self.settings.api_key = 'private-test-secret'
        self.settings.save(path)
        self.assertNotIn('private-test-secret', path.read_text())
        restored = Settings(binary='', artifact='')
        self.assertTrue(restored.load(path))
        self.assertTrue(restored.vision)
        self.assertTrue(restored.prefix_reuse)
        self.assertEqual(restored.top_p, '0.85')
        self.assertEqual(restored.api_key, '')
        self.assertNotIn('private-test-secret', self.settings.display_command())
        self.assertIn('<API_KEY>', self.settings.display_command())
        self.assertIn('private-test-secret', self.settings.command())
        from io import StringIO
        with patch.object(sys, 'argv', ['tui', '--settings', str(path), '--profile', 'quasar-long',
                                       '--binary', sys.executable, '--artifact', str(self.artifact), '--print-command']), \
                patch('sys.stdout', new_callable=StringIO) as output:
            self.assertEqual(main(), 0)
        self.assertIn('--volta-fast-decode', output.getvalue())
        self.assertNotIn('--vision ', output.getvalue())
        self.assertNotIn('--top-p', output.getvalue())

    def test_sampling_and_service_options(self):
        self.settings.api_key = 'local secret'
        self.settings.preserve_thinking = self.settings.cors = True
        self.settings.kv_capacity = 'auto'
        self.settings.temperature = '0.7'
        self.settings.seed = '18446744073709551615'
        self.settings.validate()
        command = self.settings.command()
        self.assertIn('--cors', command)
        self.assertIn('--preserve-thinking', command)
        self.assertEqual(command[command.index('--kv-capacity') + 1], 'auto')
        for name, value in (('seed', '18446744073709551616'), ('temperature', 'nan'),
                            ('top_p', '1.1'), ('top_k', '3.5')):
            before = getattr(self.settings, name)
            setattr(self.settings, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.settings.validate()
            setattr(self.settings, name, before)

    def test_pipeline_and_ram_kv_admission_and_restore(self):
        self.settings.pp, self.settings.tp = 2, 1
        self.settings.ram_kv = True
        self.settings.validate()
        command = self.settings.command()
        for option, value in (('--pp', '2'), ('--pipeline-chunks', '8'),
                              ('--ram-kv-window', '32768'), ('--kv-capacity', '32768')):
            self.assertEqual(command[command.index(option) + 1], value)
        self.settings.ram_window = 8192
        with self.assertRaisesRegex(ValueError, 'window'):
            self.settings.validate()
        self.settings.ram_window = 32768
        self.settings.vision = True
        with self.assertRaisesRegex(ValueError, 'PP2'):
            self.settings.validate()
        self.settings.apply_profile('quasar-mtp4')
        self.settings.artifact = str(self.artifact)
        self.settings.validate()
        self.assertEqual(self.settings.pp, 1)
        self.assertFalse(self.settings.ram_kv)
        self.assertNotIn('--pp', self.settings.command())

    def test_session_metrics_skip_historical_log_and_reset_on_start(self):
        path = Path(self.settings.log)
        path.write_text(json.dumps({'event': 'request_done', 'result': {
            'prompt_tokens': 99999, 'prefix_cache_hit_tokens': 0}}) + '\n')
        server = Server()
        self.addCleanup(server.close)
        script = 'import time; time.sleep(30)'
        with patch.object(self.settings, 'command', return_value=[sys.executable, '-c', script]):
            server.start(self.settings)
        records = [
            {'event': 'throughput', 'interval_seconds': 1.0,
             'speculative': {'drafted_tokens': 40, 'accepted_tokens': 30}},
            {'event': 'request_done', 'result': {'prompt_tokens': 1000,
                'prefix_cache_hit_tokens': 800, 'completion_tokens': 101},
             'timings_seconds': {'prefill': 0.5, 'decode': 2.0},
             'speculative': {'drafted_tokens': 120, 'accepted_tokens': 80}},
        ]
        with path.open('a') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        server.poll(self.settings)
        launcher = Launcher(self.settings, 'custom')
        launcher.server = server
        display = '\n'.join(launcher.metrics_lines())
        for expected in ('75.0%', '80.0%', '400.0 tok/s', '50.0 tok/s', '66.7%', '1 requests'):
            self.assertIn(expected, display)
        self.assertEqual(server.stats.requests, 1)
        server.close()
        with patch.object(self.settings, 'command', return_value=[sys.executable, '-c', script]):
            server.start(self.settings)
        server.poll(self.settings)
        self.assertEqual(server.stats.requests, 0)

    def test_explicit_paths_and_cuda_mapping(self):
        self.settings.artifact = str(Path(self.directory.name) / 'model with spaces.ninfer')
        Path(self.settings.artifact).touch()
        self.settings.devices = '4,6'
        self.settings.validate()
        command = self.settings.command()
        self.assertEqual(command[1], self.settings.artifact)
        self.assertEqual(command[command.index('--devices') + 1], '4,6')
        self.settings.devices = '0,0'
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.settings.validate()

    def test_unsupported_long_attention_and_tp4_dflash(self):
        for tp, backend, long_attention in [(4, 'dflash', False),
                                            (1, 'mtp', True)]:
            self.settings.tp = tp
            self.settings.devices = ','.join(map(str, range(tp)))
            self.settings.backend = backend
            self.settings.long_attention = long_attention
            with self.subTest(tp=tp, backend=backend), self.assertRaises(ValueError):
                self.settings.validate()

    def test_dflash_tp2_experimental_profiles(self):
        self.settings.apply_profile('quasar-dflash7')
        self.settings.artifact = str(self.artifact)
        self.settings.long_attention = self.settings.fast_decode = True
        self.settings.validate()
        self.assertIn('--volta-long-attention', self.settings.command())
        self.assertIn('--volta-fast-decode', self.settings.command())

    def test_owned_child_stop_and_logs(self):
        external = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: external.poll() is None and external.terminate())
        server = Server()
        self.addCleanup(server.close)
        script = "print('owned child ready', flush=True); import time; time.sleep(30)"
        with patch.object(self.settings, 'command', return_value=[sys.executable, '-u', '-c', script]):
            server.start(self.settings)
        child = server.process
        lines = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and 'owned child ready' not in lines:
            lines.extend(server.drain())
            time.sleep(0.01)
        self.assertIn('owned child ready', lines)
        server.close()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(external.poll())
        external.terminate()
        external.wait(timeout=3)

    def test_interval_rates_do_not_use_request_average(self):
        server = Server()
        server.lines.put('throughput interval=1.000s prefill=1234.5tok/s decode=143.5tok/s')
        server.drain()
        self.assertEqual((server.prefill, server.decode), (1234.5, 143.5))
        server.lines.put('[req 1] done prefill=900.0tok/s decode=100.0tok/s')
        server.drain()
        self.assertEqual((server.prefill, server.decode), (1234.5, 143.5))

    def test_restart_waits_for_previous_child_to_exit(self):
        server = Server()
        self.addCleanup(server.close)
        with patch.object(self.settings, 'command', return_value=[sys.executable, '-c', 'import time; time.sleep(30)']):
            server.start(self.settings)
            first = server.process
            server.restart()
            deadline = time.monotonic() + 3
            while server.process is first and time.monotonic() < deadline:
                server.poll(self.settings)
                time.sleep(0.01)
            self.assertIsNotNone(first.poll())
            self.assertTrue(server.running)
            self.assertIsNot(server.process, first)
            self.assertFalse(server.restart_pending)

    def test_terminal_start_metrics_stop_and_saved_preferences(self):
        root = Path(self.directory.name)
        fake = root / 'fake-serve'
        records = [
            {'event': 'throughput', 'interval_seconds': 1.0,
             'speculative': {'drafted_tokens': 40, 'accepted_tokens': 30}},
            {'event': 'request_done', 'result': {'prompt_tokens': 1000,
                'prefix_cache_hit_tokens': 800, 'completion_tokens': 101},
             'timings_seconds': {'prefill': 0.5, 'decode': 2.0},
             'speculative': {'drafted_tokens': 120, 'accepted_tokens': 80}},
        ]
        fake.write_text(f'#!{sys.executable}\n'
            'import sys, pathlib, time\n'
            'log = pathlib.Path(sys.argv[sys.argv.index("--request-log-jsonl") + 1])\n'
            'print("listening on http://127.0.0.1", flush=True)\n'
            f'with log.open("a") as stream: stream.write({repr("".join(json.dumps(r) + chr(10) for r in records))})\n'
            'time.sleep(15)\n')
        fake.chmod(0o700)
        self.settings.binary = str(fake)
        preferences = root / 'tui.json'
        self.settings.save(preferences)
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 140, 0, 0))
        env = dict(os.environ, TERM='xterm-256color')
        process = subprocess.Popen([sys.executable, '-m', 'tools.v100.ninfer_tui',
                                    '--settings', str(preferences)],
                                   cwd=Path(__file__).resolve().parents[1], env=env,
                                   stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        def cleanup():
            if process.poll() is None:
                os.write(master, b'q')
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        self.addCleanup(cleanup)
        captured = bytearray()
        def expect(text):
            deadline = time.monotonic() + 5
            while text not in captured and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        captured.extend(os.read(master, 65536))
                    except OSError:
                        break
            self.assertIn(text, captured, captured.decode(errors='replace'))
        expect(b'NInfer V100')
        os.write(master, b's')
        expect(b'listening on http')
        os.write(master, b'v')
        expect(b'Last prefix cache: 80.0%')
        expect(b'Decode: 50.0 tok/s')
        expect(b'Live draft acceptance: 75.0%')
        os.write(master, b'x')
        os.write(master, b's')
        expect(b'EXIT')
        os.write(master, b'q')
        self.assertEqual(process.wait(timeout=3), 0)
        restored = Settings(binary='', artifact='')
        restored.load(preferences)
        self.assertEqual(restored.binary, str(fake))


if __name__ == '__main__':
    unittest.main()
