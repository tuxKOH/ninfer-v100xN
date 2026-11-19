#!/usr/bin/env python3
"""Terminal launcher for one owned ninfer-serve child; Python standard library only."""
from __future__ import annotations

import argparse
from collections import deque
import curses
from dataclasses import asdict, dataclass, field, fields
import json
import locale
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

if __package__:
    from .gui_cache import CacheMonitor
    from .launcher_sampling import SAMPLING_LIMITS, validate_sampling
    from .launcher_model import ModelControl
else:
    from gui_cache import CacheMonitor
    from launcher_sampling import SAMPLING_LIMITS, validate_sampling
    from launcher_model import ModelControl

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get('NINFER_MODEL_DIR', '/Models/ninfer-V100X2'))
MODELS = {
    'quasar': MODEL_ROOT / 'quasar-v3/qwen3_8_27b_nvfp4.ninfer',
    'nvfp4': MODEL_ROOT / 'qwen3_8_27b_nvfp4.ninfer',
    'q4': MODEL_ROOT / 'qwen3_8_27b_q4_k_m.ninfer',
    'fp8': MODEL_ROOT / 'qwen3_8_27b_fp8.ninfer',
}


def local_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Profile:
    label: str
    model: str = 'quasar'
    tp: int = 2
    context: int = 180000
    chunk: int = 2560
    backend: str = 'mtp'
    drafts: int = 4
    optimized: bool = True
    long_attention: bool = False
    fast_decode: bool = False
    vision: bool = False
    vision_max_tokens: int = 1024
    greedy: bool = False
    prefix_reuse: bool = True


PROFILES = {
    'quasar-mtp4': Profile('QUASAR / TP2 / MTP4 / 180K', drafts=4),
    'quasar-long': Profile('QUASAR / TP2 / MTP4 / experimental attention',
                           drafts=4, long_attention=True, fast_decode=True, greedy=True, prefix_reuse=False),
    'quasar-vision': Profile('QUASAR / TP2 / MTP4 / Vision', vision=True, context=98304, chunk=1024,
                             vision_max_tokens=4096, prefix_reuse=False),
    'quasar-dflash7': Profile('QUASAR / TP2 / DFlash7 / 98K', context=98304,
                              chunk=1024, backend='dflash', drafts=7, optimized=False),
    'nvfp4-tp4': Profile('Official NVFP4 / TP4 / MTP3 / 180K', model='nvfp4', tp=4, drafts=3),
    'fp8-tp4': Profile('Native FP8 / TP4 / MTP3 / 180K', model='fp8', tp=4, drafts=3),
    'q4-mtp3': Profile('Q4_K_M / TP2 / MTP3 / 180K', model='q4', chunk=4096, drafts=3),
}


@dataclass
class Settings:
    binary: str
    artifact: str
    tp: int = 2
    pp: int = 1
    pipeline_chunks: int = 8
    devices: str = '0,1'
    context: int = 180000
    kv_capacity: str = ''
    kv_dtype: str = 'int8'
    chunk: int = 2560
    backend: str = 'mtp'
    drafts: int = 4
    optimized: bool = True
    long_attention: bool = False
    fast_decode: bool = False
    vision: bool = False
    vision_device: str = 'gpu'
    vision_max_tokens: int = 1024
    graphs: bool = True
    prefix_reuse: bool = True
    prefix_cache_tokens: int = 384000
    thinking: bool = True
    preserve_thinking: bool = False
    greedy: bool = False
    host: str = '127.0.0.1'
    port: int = 8080
    max_output: int = 65536
    max_concurrency: int = 1
    api_key: str = field(default='', repr=False)
    cors: bool = False
    temperature: str = ''
    top_p: str = ''
    top_k: str = ''
    min_p: str = ''
    presence_penalty: str = ''
    frequency_penalty: str = ''
    seed: str = ''
    ram_kv: bool = False
    ram_window: int = 32768
    ram_budget: int = 32000000000
    log: str = str(ROOT / 'profiles/tui-server.jsonl')

    def apply_profile(self, name: str) -> None:
        preset = PROFILES[name]
        self.artifact = str(MODELS[preset.model])
        self.tp, self.context, self.chunk = preset.tp, preset.context, preset.chunk
        self.pp, self.pipeline_chunks = 1, 8
        self.kv_capacity, self.kv_dtype = '', 'int8'
        self.ram_kv, self.max_concurrency = False, 1
        self.graphs = True
        for name in SAMPLING_LIMITS:
            setattr(self, name, '')
        self.devices = ','.join(map(str, range(preset.tp)))
        self.backend, self.drafts = preset.backend, preset.drafts
        self.optimized, self.long_attention = preset.optimized, preset.long_attention
        self.fast_decode, self.vision = preset.fast_decode, preset.vision
        self.vision_max_tokens = preset.vision_max_tokens
        self.greedy, self.prefix_reuse = preset.greedy, preset.prefix_reuse

    def command(self) -> list[str]:
        args = [str(local_path(self.binary)), str(local_path(self.artifact)),
                '--host', self.host, '--port', str(self.port), '--tp', str(self.tp),
                '--devices', self.devices, '--max-context', str(self.context),
                '--kv-dtype', self.kv_dtype, '--prefill-chunk', str(self.chunk),
                '--max-concurrency', str(self.max_concurrency),
                '--prefix-cache-tokens', str(self.prefix_cache_tokens), '--default-max-tokens', str(self.max_output),
                '--request-log-jsonl', str(local_path(self.log)),
                '--log-stats-interval-ms', '1000']
        if self.pp != 1:
            args += ['--pp', str(self.pp), '--pipeline-chunks', str(self.pipeline_chunks)]
        if self.ram_kv:
            args += ['--ram-kv-window', str(self.ram_window), '--kv-capacity', str(self.ram_window),
                     '--ram-kv-budget-bytes', str(self.ram_budget)]
        elif self.kv_capacity.strip():
            args += ['--kv-capacity', self.kv_capacity.strip()]
        if self.api_key:
            args += ['--api-key', self.api_key]
        for name in SAMPLING_LIMITS:
            value = getattr(self, name).strip()
            if value:
                args += ['--' + name.replace('_', '-'), value]
        if self.backend != 'none':
            args += ['--spec', self.backend, '--draft-tokens', str(self.drafts)]
            if self.optimized:
                args += ['--lm-head-draft']
        if self.vision:
            args += ['--vision', '--vision-device', self.vision_device,
                     '--vision-max-tokens', str(self.vision_max_tokens)]
            if self.vision_device == 'cpu':
                args += ['--pending-timeout-ms', '600000']
        for enabled, flag in ((self.long_attention, '--volta-long-attention'),
                              (self.fast_decode, '--volta-fast-decode'),
                              (not self.graphs, '--no-cuda-graph'),
                              (not self.prefix_reuse, '--no-prefix-reuse'),
                              (not self.thinking, '--no-thinking'), (self.greedy, '--greedy'),
                              (self.preserve_thinking, '--preserve-thinking'), (self.cors, '--cors')):
            if enabled:
                args.append(flag)
        return args

    def display_command(self) -> str:
        args = self.command()
        if '--api-key' in args:
            args[args.index('--api-key') + 1] = '<API_KEY>'
        return shlex.join(args)

    def save(self, path: Path) -> None:
        data = asdict(self)
        del data['api_key']
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)

    def load(self, path: Path) -> bool:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return False
        if not isinstance(data, dict):
            raise ValueError('Saved settings must be a JSON object')
        values = {}
        for item in fields(self):
            if item.name == 'api_key' or item.name not in data:
                continue
            value = data[item.name]
            if type(value) is not type(getattr(self, item.name)):
                raise ValueError(f'Invalid saved setting: {item.name}')
            values[item.name] = value
        for name, value in values.items():
            setattr(self, name, value)
        return True

    def validate(self) -> None:
        binary, artifact = local_path(self.binary), local_path(self.artifact)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'Executable not found: {binary}')
        if not artifact.is_file() or artifact.suffix != '.ninfer':
            raise ValueError(f'Model .ninfer not found: {artifact}')
        ids = self.devices.split(',')
        if (self.tp not in (1, 2, 4) or self.pp not in (1, 2) or len(ids) != self.tp * self.pp or
                any(not item.strip().isdecimal() for item in ids) or
                len(set(map(int, ids))) != len(ids)):
            raise ValueError('Devices must contain TP * PP distinct logical CUDA IDs')
        if self.pp == 2 and (self.tp != 1 or self.max_concurrency != 1 or self.vision or
                             self.backend == 'dflash' or self.fast_decode or self.long_attention):
            raise ValueError('PP2 requires TP1, concurrency 1, NVFP4/QUASAR Text None/MTP')
        if not 1 <= self.pipeline_chunks <= 8:
            raise ValueError('Pipeline chunks must be 1..8')
        if not 1 <= self.context <= 262144:
            raise ValueError('Native context capacity must be 1..262144')
        if self.chunk <= 0 or self.chunk % 128:
            raise ValueError('Prefill chunk must be a positive multiple of 128')
        if self.backend not in ('none', 'mtp', 'dflash'):
            raise ValueError('Speculation must be none, mtp or dflash')
        if self.backend == 'mtp' and not 1 <= self.drafts <= 5:
            raise ValueError('MTP draft window must be 1..5; accepting zero is valid')
        if self.backend == 'dflash' and (self.tp == 4 or not 1 <= self.drafts <= 7):
            raise ValueError('27B DFlash requires TP1/TP2 and a draft window of 1..7')
        if not 0 <= self.prefix_cache_tokens <= 100000000:
            raise ValueError('Historical prefix cache budget must be 0..100000000 tokens (host RAM)')
        if self.long_attention and (self.tp not in (2, 4) or (self.backend == 'dflash' and self.tp != 2) or
                                    self.kv_dtype != 'int8' or self.ram_kv):
            raise ValueError('Experimental long attention requires TP2/TP4 INT8 KV concurrency 1..8, Vision/DFlash TP2 only, no RAM-KV')
        if self.fast_decode and ((self.backend == 'dflash' and self.tp != 2) or self.vision):
            raise ValueError('Experimental fast decode requires QUASAR SM70 TP1/TP2/TP4 Text, DFlash TP2 only')
        if self.vision and (self.tp != 2 or self.backend == 'dflash'):
            raise ValueError('Vision requires TP2 None/MTP; fast decode remains Text-only')
        if self.vision and local_path(self.artifact).name in ('qwen3_8_27b_q4_k_m.ninfer', 'qwen3_8_27b_fp8.ninfer'):
            raise ValueError('Q4_K_M and native FP8 do not support Vision')
        if self.vision_device not in ('gpu', 'cpu'):
            raise ValueError('Vision device must be gpu or cpu')
        if not 1 <= self.vision_max_tokens <= 32768:
            raise ValueError('Vision budget must be 1..32768')
        if self.kv_dtype not in ('int8', 'bf16') or not 1 <= self.max_concurrency <= 8:
            raise ValueError('KV dtype must be int8/bf16; concurrency must be 1..8')
        capacity = self.kv_capacity.strip()
        if not self.ram_kv and capacity and capacity != 'auto' and (
                not capacity.isdecimal() or int(capacity) < self.context):
            raise ValueError('KV capacity must be auto, empty, or at least context capacity')
        if self.ram_kv:
            if self.vision or self.backend == 'dflash' or self.tp == 4 or self.max_concurrency != 1:
                raise ValueError('RAM-KV requires TP1/TP2/PP2 concurrency 1 Text None/MTP')
            minimum = self.chunk * (self.pipeline_chunks if self.pp == 2 else 1) + 8192
            if (self.ram_window % 64 or not minimum <= self.ram_window <= self.context or
                    not 0 < self.ram_budget <= 18446744073709551615):
                raise ValueError(f'RAM-KV window must be a multiple of 64 in {minimum}..{self.context}; budget must be positive')
        for name in SAMPLING_LIMITS:
            validate_sampling(name, getattr(self, name))
        if local_path(self.artifact).name == 'qwen3_8_27b_fp8.ninfer' and self.tp != 4:
            raise ValueError('Native FP8 requires TP4')
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError('Host must be nonempty; port must be 1..65535')
        if self.max_output <= 0:
            raise ValueError('Output limit must be positive; remaining context still limits output')
        if not self.log.strip():
            raise ValueError('Request log path is required')


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    libraries = [ROOT / 'build/_deps/install/lib', Path('/usr/local/cuda-12.8/lib64')]
    parts = [str(path) for path in libraries if path.is_dir()]
    if env.get('LD_LIBRARY_PATH'):
        parts.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH'] = ':'.join(parts)
    return env


class Server:
    """Owns only the process launched here; no process-name matching or external kills."""
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.status = 'STOPPED'
        self.stop_deadline = 0.0
        self.next_health = 0.0
        self.health = None
        self.prefill = self.decode = 0.0
        self.stats: CacheMonitor | None = None
        self.stats_error = ''
        self.restart_pending = False
        self.model_control: ModelControl | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, settings: Settings) -> None:
        if self.running:
            return
        settings.validate()
        self.restart_pending = False
        local_path(settings.log).parent.mkdir(parents=True, exist_ok=True)
        self.lines = queue.Queue()
        log_path = local_path(settings.log)
        self.stats = CacheMonitor(log_path)
        self.stats.offset = log_path.stat().st_size if log_path.exists() else 0
        self.stats_error = ''
        self.prefill = self.decode = 0.0
        self.status, self.stop_deadline = 'LOADING', 0.0
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(settings.host, settings.host)
        if ':' in host:
            host = '[' + host.strip('[]') + ']'
        self.model_control = ModelControl(f'http://{host}:{settings.port}', settings.api_key)
        self.process = subprocess.Popen(settings.command(), cwd=ROOT, env=runtime_environment(),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace')
        process = self.process
        output_queue = self.lines

        def read_output() -> None:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    output_queue.put(line.rstrip('\n'))
        threading.Thread(target=read_output, daemon=True).start()

    def stop(self) -> None:
        self.restart_pending = False
        if self.running and not self.stop_deadline:
            self.process.terminate()
            self.status = 'STOPPING'
            self.stop_deadline = time.monotonic() + 5

    def restart(self) -> None:
        if self.running:
            self.stop()
            self.restart_pending = True

    def poll(self, settings: Settings) -> None:
        now = time.monotonic()
        if self.running and not self.stop_deadline and self.status.startswith('ONLINE') and self.model_control:
            self.model_control.poll()
            if self.model_control.error:
                self.lines.put('Model control failed: ' + self.model_control.error)
                self.model_control.error = ''
        if self.stats is not None:
            try:
                self.stats.poll()
                self.stats_error = ''
            except OSError as error:
                self.stats_error = str(error)
        if self.stop_deadline and self.running and now >= self.stop_deadline:
            self.process.kill()
            self.stop_deadline = 0.0
        if self.process is not None and not self.running:
            self.status = f'EXIT {self.process.returncode}'
            self.process = None
            self.model_control = None
            self.stop_deadline = 0.0
            if self.restart_pending:
                self.start(settings)
        if not self.running or self.stop_deadline or self.health or now < self.next_health:
            return
        self.next_health = now + 2
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(settings.host, settings.host)
        if ':' in host:
            host = '[' + host.strip('[]') + ']'
        process = self.process

        def check_health() -> None:
            try:
                opener = build_opener(ProxyHandler({}))
                with opener.open(f'http://{host}:{settings.port}/health', timeout=0.5) as response:
                    online = json.load(response).get('status') == 'ok'
                # /health can report readiness before model upload completes; the
                # listening log is the authoritative transition for our own child.
                if self.process is process and self.status.startswith('ONLINE'):
                    self.status = 'ONLINE / HEALTH OK' if online else 'ONLINE / HEALTH UNAVAILABLE'
            except (OSError, URLError, ValueError):
                if self.process is process and self.status.startswith('ONLINE'):
                    self.status = 'ONLINE / HEALTH UNAVAILABLE'
            finally:
                self.health = None
        self.health = threading.Thread(target=check_health, daemon=True)
        self.health.start()

    def drain(self) -> list[str]:
        import re
        lines = []
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            lines.append(line)
            if 'listening on http' in line and self.running and not self.stop_deadline:
                self.status = 'ONLINE'
            rates = re.search(r'throughput interval=.*?prefill=([\d.]+)tok/s decode=([\d.]+)tok/s', line)
            if rates:
                self.prefill, self.decode = map(float, rates.groups())
        return lines

    def close(self) -> None:
        self.restart_pending = False
        if self.running:
            self.stop()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


FIELDS = [
    ('artifact', 'Model .ninfer', None), ('binary', 'ninfer-serve', None),
    ('tp', 'Tensor parallel', (1, 2, 4)), ('devices', 'CUDA devices', None),
    ('pp', 'Experimental PP', (1, 2)), ('pipeline_chunks', 'PP prefill chunks', None),
    ('context', 'Context capacity', None), ('kv_capacity', 'KV capacity / auto', None),
    ('kv_dtype', 'KV dtype', ('int8', 'bf16')), ('chunk', 'Prefill chunk', None),
    ('backend', 'Speculation', ('mtp', 'dflash', 'none')), ('drafts', 'Max drafts', None),
    ('optimized', 'Optimized draft head', (False, True)),
    ('vision', 'Vision ON / OFF', (False, True)),
    ('vision_device', 'Vision encoder GPU / CPU', ('gpu', 'cpu')),
    ('vision_max_tokens', 'Visual token budget', None),
    ('fast_decode', 'Fast math (Text only)', (False, True)),
    ('long_attention', 'Long attention (Text/Vision)', (False, True)),
    ('ram_kv', 'Experimental RAM-KV', (False, True)),
    ('ram_window', 'RAM-KV GPU tokens', None), ('ram_budget', 'RAM archive bytes', None),
    ('graphs', 'CUDA Graphs', (False, True)), ('prefix_reuse', 'Prefix cache', (False, True)),
    ('prefix_cache_tokens', 'History RAM tokens (384K ~12.8 GiB + state; 0 off)', None),
    ('thinking', 'Default thinking', (False, True)), ('greedy', 'Force greedy', (False, True)),
    ('preserve_thinking', 'Preserve thinking', (False, True)),
    ('max_concurrency', 'Max active requests', None),
    ('max_output', 'Max output tokens', None), ('host', 'API host', None),
    ('port', 'API port', None), ('api_key', 'API key (not saved)', None),
    ('cors', 'Allow CORS', (False, True)), ('log', 'Request JSONL log', None),
    *[(name, name.replace('_', ' ').title(), None) for name in SAMPLING_LIMITS],
]


def put(screen, y: int, x: int, text: str, style: int = 0) -> None:
    height, width = screen.getmaxyx()
    if y < height and x < width:
        try:
            screen.addnstr(y, x, text, max(0, width - x - 1), style)
        except curses.error:
            pass


class Launcher:
    def __init__(self, settings: Settings, profile: str, settings_path: Path | None = None) -> None:
        self.settings, self.profile = settings, profile
        self.server = Server()
        self.selected = 0
        self.logs: deque[str] = deque(maxlen=2000)
        self.notice = ''
        self.settings_path = settings_path

    def save_settings(self) -> None:
        if self.settings_path is not None:
            try:
                self.settings.save(self.settings_path)
            except OSError as error:
                self.notice = f'Settings not saved: {error}'

    def edit(self, screen, key: str, label: str) -> None:
        value = str(getattr(self.settings, key))
        text = value
        while True:
            self.draw(screen)
            height, width = screen.getmaxyx()
            put(screen, height - 3, 0, ' ' * (width - 1))
            shown = '*' * len(text) if key == 'api_key' else text
            put(screen, height - 3, 1, label + ' (Enter save, Esc cancel): ' + shown,
                curses.A_REVERSE)
            screen.refresh()
            try:
                character = screen.get_wch()
            except curses.error:
                continue
            if character == '\x1b':
                return
            if character in ('\n', '\r', curses.KEY_ENTER):
                try:
                    setattr(self.settings, key, int(text) if isinstance(getattr(self.settings, key), int)
                            else text.strip())
                    self.profile = 'custom'
                    self.save_settings()
                except ValueError:
                    self.notice = 'Enter an integer'
                return
            if character in ('\b', '\x7f', curses.KEY_BACKSPACE):
                text = text[:-1]
            elif character == '\x15':
                text = ''
            elif isinstance(character, str) and character.isprintable():
                text += character

    def choose(self, screen, title: str, choices: list[tuple[str, str]]) -> str | None:
        index = 0
        while True:
            self.server.poll(self.settings)
            self.logs.extend(self.server.drain())
            screen.erase()
            height, width = screen.getmaxyx()
            put(screen, 0, 1, title + ' / Up Down Enter / Esc cancel', curses.A_BOLD)
            visible = max(1, height - 4)
            start = max(0, index - visible + 1)
            for row, (_, label) in enumerate(choices[start:start + visible]):
                put(screen, row + 2, 1, label, curses.A_REVERSE if row + start == index else 0)
            screen.refresh()
            key = screen.getch()
            if key in (curses.KEY_UP, ord('k')):
                index = (index - 1) % len(choices)
            elif key in (curses.KEY_DOWN, ord('j')):
                index = (index + 1) % len(choices)
            elif key in (10, 13, curses.KEY_ENTER):
                return choices[index][0]
            elif key == 27:
                return None

    def show_lines(self, screen, title: str, lines: list[str]) -> None:
        offset = 0
        while True:
            self.server.poll(self.settings)
            self.logs.extend(self.server.drain())
            screen.erase()
            height, width = screen.getmaxyx()
            wrapped = [part for line in lines for part in textwrap.wrap(line, max(1, width - 4))]
            visible = max(1, height - 4)
            offset = min(offset, max(0, len(wrapped) - visible))
            put(screen, 0, 1, title + ' / Up Down PgUp PgDn scroll / other key returns', curses.A_BOLD)
            for row, line in enumerate(wrapped[offset:offset + visible]):
                put(screen, row + 2, 2, line)
            screen.refresh()
            key = screen.getch()
            if key in (curses.KEY_DOWN, curses.KEY_NPAGE):
                offset += visible if key == curses.KEY_NPAGE else 1
            elif key in (curses.KEY_UP, curses.KEY_PPAGE):
                offset = max(0, offset - (visible if key == curses.KEY_PPAGE else 1))
            elif key != -1 and key != curses.KEY_RESIZE:
                return

    def show_command(self, screen) -> None:
        self.show_lines(screen, 'Launch command (API key hidden)', [self.settings.display_command()])

    def show_experiments(self, screen) -> None:
        lines = [
            'EXPERIMENTAL INFERENCE - startup switches, default OFF',
            'Fast math (Text only): NVFP4 FP32 scale-after-reduction and packed MTP W8 reductions.',
            'Long attention (Text/images/video): complete history; different softmax/PV reduction order.',
            'Long attention: TP2 Text None/MTP/DFlash; Vision None/MTP; complete INT8 KV.',
            'DFlash fast math changes target NVFP4 only; native W8/BF16 drafter is retained.',
            'DFlash7 TP2 / 85K raw-prefix: normal 122.65, long 167.49, fast 128.11, both 168.15 tok/s.',
            'DFlash both: +37.10%; HumanEval32 x2: 301.21->302.38 tok/s, not pass@1.',
            'DFlash both changes 2/64 short outputs and all 85K outputs; quality/GPQA unmeasured.',
            'Combined Text-only 85K / 180K capacity / TP2 / INT8 KV / MTP4: ~198.1 committed tok/s.',
            'Remote TP4 / 220 W / 1200 MHz, same 85K/MTP4 fixture: 267.34 / 267.64 wall tok/s.',
            'TP4: +14.57% over 231.30; prefill ~2339 tok/s. Split-D can change output.',
            'TP4 normal, prefix and actual-85K Graph/eager state checks PASS; quality is separate.',
            'Final packed-W8 step: +0.84% over the preceding ~196.45 tok/s pair.',
            'These are combined results, not independent per-switch speedups.',
            'Prior 25 full-vocabulary probes: argmax 25/25, min cosine 0.998830814,',
            'max KL 0.001903103; only 9/25 pass the near-lossless bounds.',
            'Output and draft acceptance may change. IQ impact unknown; no current GPQA.',
            'Vision keeps long attention enabled; fast math and RAM-KV remain Text-only.',
            'Vision preset starts with experiments OFF: 98K context / 4096 visual tokens.',
            'PP2: TP1 on two devices; serial decode, no established speed gain.',
            'RAM-KV: approximate retrieved history during prefill AND decode; quality can drop.',
            'RAM-KV gains/quality unmeasured here. Archive bytes are NOT a process RSS limit.',
            'Details: docs/performance.md, docs/tp4.md and docs/v100-decode-survey.md',
        ]
        self.show_lines(screen, 'Experimental gains and output impact', lines)

    def metrics_lines(self) -> list[str]:
        stats = self.server.stats
        if stats is None:
            return ['No requests from this launcher yet.']
        def phase(label, tokens, seconds):
            if seconds is None:
                return label + ': unavailable'
            rate = ('fully cached' if label == 'Prefill' and tokens == 0 else
                    f'{tokens / seconds:,.1f} tok/s' if tokens > 0 and seconds > 0 else 'unavailable')
            return f'{label}: {rate} ({tokens:,} tokens / {seconds:.3f}s)'
        lines = [f'Live interval: {self.server.prefill:.1f} prefill / {self.server.decode:.1f} committed decode tok/s']
        if stats.live_interval_seconds:
            draft = (f'{stats.live_accepted_tokens / stats.live_drafted_tokens:.1%} '
                     f'({stats.live_accepted_tokens:,}/{stats.live_drafted_tokens:,})'
                     if stats.live_drafted_tokens else 'no drafts')
            lines.append(f'Live draft acceptance: {draft} / {stats.live_interval_seconds:.1f}s')
        if stats.requests:
            lines += [f'Last prefix cache: {stats.latest_rate:.1%} ({stats.latest_cached_tokens:,}/{stats.latest_prompt_tokens:,})',
                      f'Session prefix cache: {stats.total_rate:.1%} ({stats.cached_tokens:,}/{stats.prompt_tokens:,}); {stats.requests} requests',
                      phase('Prefill', stats.latest_prefill_tokens, stats.latest_prefill_seconds),
                      phase('Decode', stats.latest_decode_tokens, stats.latest_decode_seconds)]
            draft = (f'{stats.latest_accepted_tokens / stats.latest_drafted_tokens:.1%} '
                     f'({stats.latest_accepted_tokens:,}/{stats.latest_drafted_tokens:,})'
                     if stats.latest_drafted_tokens else 'no drafts')
            lines.append('Last request draft acceptance: ' + draft)
        if self.server.stats_error:
            lines.append('Log read error: ' + self.server.stats_error)
        return lines

    def show_metrics(self, screen) -> None:
        while True:
            self.server.poll(self.settings)
            self.logs.extend(self.server.drain())
            screen.erase()
            height, width = screen.getmaxyx()
            put(screen, 0, 1, 'Inference metrics / any key returns', curses.A_BOLD)
            row = 2
            for line in self.metrics_lines():
                for wrapped in textwrap.wrap(line, max(1, width - 4)):
                    if row < height - 1:
                        put(screen, row, 2, wrapped)
                    row += 1
            screen.refresh()
            if screen.getch() != -1:
                return

    def draw(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 18 or width < 80:
            put(screen, 0, 0, 'Resize terminal to at least 80 x 18; q exits.')
            screen.refresh()
            return
        divider = width // 2
        put(screen, 0, 1, 'NInfer V100 / QUASAR NVFP4 / API launcher', curses.A_BOLD)
        put(screen, 1, 1, 'Profile: ' + PROFILES[self.profile].label if self.profile in PROFILES
            else 'Profile: custom')
        put(screen, 2, 1, f'{self.server.status}   Prefill {self.server.prefill:.1f} tok/s'
            f'   Committed decode {self.server.decode:.1f} tok/s  [1s intervals]')
        put(screen, 3, 1, 'SETTINGS (locked while running)', curses.A_BOLD)
        put(screen, 3, divider + 1, 'LIVE SERVER LOG', curses.A_BOLD)
        visible = height - 9
        start = max(0, self.selected - visible + 1)
        for row, (key, label, _) in enumerate(FIELDS[start:start + visible]):
            value = getattr(self.settings, key)
            if key == 'api_key':
                value = 'set (hidden)' if value else ''
            value = ('ON' if value else 'OFF') if isinstance(value, bool) else str(value)
            if len(value) > divider - 25:
                value = '…' + value[-max(1, divider - 26):]
            line = f'{label:22} {value}'
            put(screen, 4 + row, 1, line[:divider - 2],
                curses.A_REVERSE if start + row == self.selected else 0)
        wrapped = [part for line in list(self.logs)[-visible:]
                   for part in textwrap.wrap(line, max(1, width - divider - 3))]
        for row, line in enumerate(wrapped[-visible:]):
            put(screen, row + 4, divider + 1, line)
        stats = self.server.stats
        summary = 'Cache / draft acceptance: no completed request yet'
        if stats is not None and (stats.requests or stats.live_interval_seconds):
            acceptance = f'{stats.live_accepted_tokens / stats.live_drafted_tokens:.1%}' if stats.live_drafted_tokens else 'no drafts'
            summary = f'Cache {stats.latest_rate:.1%} / session {stats.total_rate:.1%}  Drafts {acceptance}  [v details]'
        put(screen, height - 5, 1, summary)
        model = self.server.model_control
        residency = ''
        if model and self.server.running:
            residency = f'Model: {model.state.upper()}  RAM {model.retained_host_bytes / 1024**3:.2f} GiB'
            if model.busy:
                residency += f'  {model.phase}'
                if model.total:
                    residency += f' {model.done / model.total:.1%}'
        put(screen, height - 4, 1, residency or self.notice)
        put(screen, height - 3, 1, 'u unload->RAM  g load->GPU  s start/stop  r restart  v metrics  l logs  q quit')
        persistence = 'settings autosaved' if self.settings_path else 'session settings only'
        put(screen, height - 2, 1, 'Arrows/Tab field  Enter edit  c command  i impact  p preset  m model  (' + persistence + ')')
        screen.refresh()

    def run(self, screen) -> None:
        curses.curs_set(0)
        screen.timeout(100)
        screen.keypad(True)
        try:
            while True:
                self.server.poll(self.settings)
                self.logs.extend(self.server.drain())
                self.draw(screen)
                key = screen.getch()
                if key in (ord('q'), 27):
                    return
                if key in (curses.KEY_DOWN, ord('j'), 9):
                    self.selected = (self.selected + 1) % len(FIELDS)
                elif key in (curses.KEY_UP, ord('k'), curses.KEY_BTAB):
                    self.selected = (self.selected - 1) % len(FIELDS)
                elif key == ord('c'):
                    self.show_command(screen)
                elif key == ord('i'):
                    self.show_experiments(screen)
                elif key == ord('v'):
                    self.show_metrics(screen)
                elif key == ord('l'):
                    self.show_lines(screen, 'Server log snapshot', list(self.logs))
                elif key == ord('r'):
                    self.server.restart()
                elif key in (ord('u'), ord('g')) and self.server.running and not self.server.stop_deadline:
                    model = self.server.model_control
                    if model:
                        model.action('offload' if key == ord('u') else 'load')
                elif key == ord('s'):
                    if self.server.running:
                        self.server.stop()
                    else:
                        try:
                            self.server.start(self.settings)
                            self.save_settings()
                            self.notice = 'Started owned child; q stops it before exit'
                        except (OSError, ValueError) as error:
                            self.notice = str(error)
                elif not self.server.running:
                    if key == ord('p'):
                        choice = self.choose(screen, 'Select profile',
                                             [(name, preset.label) for name, preset in PROFILES.items()])
                        if choice:
                            self.settings.apply_profile(choice)
                            self.profile = choice
                            self.notice = 'Preset applied; edit paths before starting if needed'
                            self.save_settings()
                    elif key == ord('m'):
                        choices = [(str(path), name + ' / ' + str(path))
                                   for name, path in MODELS.items() if path.is_file()]
                        choices.append(('', 'Enter an explicit model path'))
                        choice = self.choose(screen, 'Model artifact', choices)
                        if choice:
                            self.settings.artifact = choice
                            self.profile = 'custom'
                            self.save_settings()
                        elif choice == '':
                            self.edit(screen, 'artifact', 'Model .ninfer')
                    elif key in (10, 13, curses.KEY_ENTER, ord(' ')):
                        field, label, choices = FIELDS[self.selected]
                        if choices:
                            value = getattr(self.settings, field)
                            setattr(self.settings, field, choices[(choices.index(value) + 1) % len(choices)])
                            if field == 'vision' and self.settings.vision:
                                self.settings.fast_decode = False
                                self.settings.prefix_reuse = False
                                self.settings.ram_kv = False
                            if field == 'backend' and self.settings.backend in ('none', 'dflash'):
                                self.settings.optimized = False
                            self.profile = 'custom'
                            self.save_settings()
                        else:
                            self.edit(screen, field, label)
        finally:
            self.server.close()
            self.save_settings()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, help='apply a preset after loading saved settings')
    parser.add_argument('--settings', type=Path, default=Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'ninfer/tui.json')
    parser.add_argument('--no-settings', action='store_true', help='do not read or save preferences')
    parser.add_argument('--binary', type=Path, help='explicit ninfer-serve executable')
    parser.add_argument('--artifact', type=Path, help='explicit model .ninfer path')
    parser.add_argument('--host', help='override the API listen address')
    parser.add_argument('--log', type=Path, help='override the request JSONL log path')
    parser.add_argument('--print-command', action='store_true', help='print command without loading a model')
    args = parser.parse_args()
    candidates = [ROOT / 'build-v100/apps/ninfer-serve', ROOT / 'build-v100-tp4/apps/ninfer-serve']
    binary = args.binary or next((path for path in candidates if path.is_file()), candidates[0])
    settings = Settings(binary=str(binary), artifact='')
    settings.apply_profile('quasar-mtp4')
    profile = 'quasar-mtp4'
    settings_path = None if args.no_settings else args.settings.expanduser().resolve()
    if settings_path is not None:
        try:
            if settings.load(settings_path):
                profile = 'custom'
        except (OSError, ValueError) as error:
            parser.error(f'Cannot load settings ({settings_path}): {error}; use --no-settings to reset for this run')
    if args.profile:
        settings.apply_profile(args.profile)
        profile = args.profile
    if args.binary:
        settings.binary = str(args.binary.expanduser().resolve())
    if args.artifact:
        settings.artifact = str(args.artifact.expanduser().resolve())
    if args.host:
        settings.host = args.host
    if args.log:
        settings.log = str(args.log.expanduser().resolve())
    if args.print_command:
        print(settings.display_command())
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error('TUI requires a terminal; use --print-command for command output')
    locale.setlocale(locale.LC_ALL, '')
    try:
        curses.wrapper(Launcher(settings, profile, settings_path).run)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
