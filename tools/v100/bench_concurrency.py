#!/usr/bin/env python3
"""Resume the 2K/16K/85K × 1/2/4/8 QUASAR MTP4 total-throughput matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v100.health_log import HealthLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument('--corpus', type=Path,
                        help='Use a prefix of one saved raw token corpus')
    corpus.add_argument('--corpus-dir', type=Path,
                        help='Exact complete prompts named code-chat-<input tokens>.ids')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--tp', type=int, choices=[2, 4], default=2)
    parser.add_argument('--kv-capacity', type=int, default=180000)
    parser.add_argument('--prefill-decode-rounds', type=int, choices=range(1, 65), default=1)
    parser.add_argument('--input-tokens', type=int, nargs='+', default=[2048, 16384, 85000],
                        choices=[2048, 16384, 85000])
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8],
                        choices=[1, 2, 4, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for length in args.input_tokens:
        prompt = (args.corpus_dir / f'code-chat-{length}.ids'
                  if args.corpus_dir else args.corpus)
        ids = prompt.read_text().split()
        if len(ids) < length or (args.corpus_dir and len(ids) != length):
            raise ValueError(f'prompt length mismatch: {prompt}')
        for concurrency in args.concurrency:
            key = f'{length}-c{concurrency}'
            report = args.output / f'{key}.json'
            if report.exists():
                saved = json.loads(report.read_text())
                if (saved['input_tokens'], saved['concurrency'], saved['output_tokens_per_request'],
                        saved['tp'], saved['kv_capacity'], saved['prefill_decode_rounds'],
                        saved['prompt_fixture']) != (
                        length, concurrency, args.output_tokens, args.tp, args.kv_capacity,
                        args.prefill_decode_rounds, prompt.name):
                    raise ValueError(f'workload mismatch in {report}')
                print(f'resume: keeping {key}', flush=True)
                continue
            command = [str(args.binary.resolve()), str(args.artifact.resolve()), str(prompt.resolve()),
                       str(length), str(concurrency), str(args.output_tokens), str(report.resolve()),
                       '--tp', str(args.tp), '--kv-capacity', str(args.kv_capacity),
                       '--prefill-decode-rounds', str(args.prefill_decode_rounds)]
            attempt = time.time_ns()
            with HealthLog(args.output / f'{key}-health-{attempt}', interval=0.5) as health:
                with (args.output / f'{key}-{attempt}.log').open('w') as output:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                    health.attach_pid(process.pid)
                    health.event('benchmark_start', command=command)
                    code = process.wait()
                    health.event('benchmark_end', returncode=code)
                if code:
                    raise RuntimeError(f'{key} failed: see {args.output}')
            result = json.loads(report.read_text())
            print(f"{key}: {result['total_decode_tok_s']:.3f} decode tok/s; "
                  f"mean batch {result['mean_decode_batch']:.2f}; "
                  f"makespan {result['group_makespan_seconds']:.2f}s; "
                  f"accepted {result['accepted_drafts']}/{result['proposed_drafts']} "
                  f"({result['draft_acceptance_rate']:.1%})", flush=True)


if __name__ == '__main__':
    main()
