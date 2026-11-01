#!/usr/bin/env python3
"""Publish a compact report and SVG from the saved long-concurrency matrix."""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        help='Compare matching saved cases from a preceding implementation')
    parser.add_argument('--incomplete-reason',
                        help='Allow missing cases and publish this explanation alongside the chart')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for length in (2048, 16384, 85000):
        for concurrency in (1, 2, 4, 8):
            path = args.input / f'{length}-c{concurrency}.json'
            if not path.exists() and args.incomplete_reason:
                missing.append({'input_tokens': length, 'concurrency': concurrency,
                                'reason': args.incomplete_reason})
                continue
            source = json.loads(path.read_text())
            if (source['input_tokens'], source['concurrency']) != (length, concurrency):
                raise ValueError(f'workload mismatch in {path}')
            if (source['kv_capacity'], source['mtp_drafts'], source['output_tokens_per_request'],
                    source['prefix_reuse']) != (180000, 4, 512, False):
                raise ValueError(f'configuration mismatch in {path}')
            if (len(source['requests']) != concurrency or
                    source['committed_decode_tokens'] != concurrency * 511 or
                    any(request['prompt_tokens'] != length or request['completion_tokens'] != 512 or
                        request['cached_tokens'] != 0 or request['accepted'] > request['drafted']
                        for request in source['requests'])):
                raise ValueError(f'incomplete or inconsistent request counters in {path}')
            row = {key: value for key, value in source.items()
                   if key not in ('scheduler_samples', 'requests')}
            row['accepted_drafts'] = sum(request['accepted'] for request in source['requests'])
            row['proposed_drafts'] = sum(request['drafted'] for request in source['requests'])
            row['cached_tokens'] = sum(request['cached_tokens'] for request in source['requests'])
            if args.baseline:
                baseline = json.loads((args.baseline / path.name).read_text())
                for key in ('input_tokens', 'concurrency', 'output_tokens_per_request', 'tp',
                            'kv_capacity', 'mtp_drafts', 'graphs', 'prefill_chunk',
                            'prefill_decode_rounds', 'long_attention', 'fast_decode', 'prefix_reuse'):
                    if baseline[key] != source[key]:
                        raise ValueError(f'baseline configuration mismatch for {key} in {path}')
                if len(baseline['requests']) != concurrency:
                    raise ValueError(f'incomplete baseline in {path}')
                pairs = list(zip(source['requests'], baseline['requests']))
                row['baseline'] = {
                    'total_decode_tok_s': baseline['total_decode_tok_s'],
                    'all_live_sampled_tok_s': baseline.get('all_live_sampled_tok_s'),
                    'group_output_tok_s': baseline['group_output_tok_s'],
                    'mean_decode_batch': baseline['mean_decode_batch'],
                    'accepted_drafts': sum(request['accepted'] for request in baseline['requests']),
                    'proposed_drafts': sum(request['drafted'] for request in baseline['requests']),
                }
                row['decode_speedup'] = source['total_decode_tok_s'] / baseline['total_decode_tok_s']
                row['output_ids_equal'] = all(
                    actual['generated_token_ids'] == old['generated_token_ids'] for actual, old in pairs)
                row['speculative_counters_equal'] = all(
                    actual[key] == old[key] for actual, old in pairs for key in ('accepted', 'drafted'))
            rows.append(row)
    summary = {
        'hardware': '2 x Tesla V100-SXM2 16 GB, PCIe P2P, no NVLink',
        'toolchain': 'CUDA 12.8, SM70',
        'artifact': 'Qwen3.8-27B QUASAR NVFP4 v3',
        'sampling': 'greedy, zero presence/frequency penalties, model stops disabled',
        'method': 'One cold group per case; 512 output tokens/request. Decode throughput is '
                  'committed tokens excluding each first output divided by summed executor '
                  'decode-round wall time, including commit/publication and excluding prefill. '
                  'Group output throughput includes prefill and queue time. '
                  'All-live sampled throughput uses complete 10ms sample intervals inside all '
                  'requests\' common decode lifetime, including any intervening stalls; it is '
                  'not HTTP streaming timing. '
                  '180000 shared KV tokens can limit actual active requests.',
        'quality': 'Speed only; long attention and fast decode can change output. No new GPQA.',
        'completed_cases': len(rows),
        'expected_cases': 12,
        'missing_cases': missing,
        'cases': rows,
    }
    if args.baseline:
        summary['comparison'] = 'Same-host preceding implementation, one cold group per case. '
        summary['comparison'] += ('Changed output IDs or draft acceptance prevent attributing the entire '
                                  'speed difference to kernel execution alone. The new T>32 batch '
                                  'reduction differs from the former CUTLASS path. No new GPQA.')
    (args.output / 'quasar-concurrency.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    for length, label, color in ((2048, '2K input', '#2563eb'),
                                 (16384, '16K input', '#0f766e'),
                                 (85000, '85K input', '#b45309')):
        cases = [row for row in rows if row['input_tokens'] == length]
        x = [row['concurrency'] for row in cases]
        y = [row['total_decode_tok_s'] for row in cases]
        axes[0].plot(x, y, 'o-', label=label, color=color, linewidth=2)
        if args.baseline:
            axes[0].plot(x, [row['baseline']['total_decode_tok_s'] for row in cases],
                         '--', color=color, linewidth=1.3, alpha=0.65)
        for a, b in zip(x, y):
            offset = -16 if (length == 16384 and a == 1) or (length == 85000 and a == 2) else 10
            if args.baseline:
                if length == 16384 and a in (1, 2):
                    offset = -18
                elif length == 85000:
                    offset = 28 if a == 1 else -30
            axes[0].annotate(f'{b:.1f}', (a, b), xytext=(0, offset),
                             textcoords='offset points', ha='center', fontsize=9, color=color)
        axes[1].plot(x, [row['group_output_tok_s'] for row in cases], 'o-',
                     label=label, color=color, linewidth=2)
        axes[2].plot(x, [row['mean_decode_batch'] for row in cases], 'o-',
                     label=label, color=color, linewidth=2)
    axes[0].set(ylabel='Total committed decode tokens / second',
                title='Decode throughput (prefill excluded)',
                ylim=(0, max(row['total_decode_tok_s'] for row in rows) * 1.2))
    axes[1].set(ylabel='Total output tokens / second',
                title='Whole group (prefill + queue included)', ylim=(0, None))
    axes[2].set(ylabel='Mean requests per decode round', title='Actual decode batch',
                ylim=(0, 8.5), yticks=range(0, 9, 2))
    for axis in axes:
        axis.set(xscale='log', xticks=[1, 2, 4, 8], xticklabels=['1', '2', '4', '8'],
                 xlabel='Submitted concurrent requests')
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(axis='y', alpha=0.18)
        axis.legend(frameon=False)
    fig.suptitle('QUASAR NVFP4 / TP2 / MTP4 — 2 × V100 16 GB', fontsize=14)
    footer = '180K shared KV · 512 output tokens/request · long attention + fast decode · one cold group/case'
    if args.baseline:
        footer += '\nSolid: current · Dashed: preceding implementation · output/acceptance may differ'
    if missing:
        note = 'Not measured in this campaign: ' + ', '.join(
            f"{row['input_tokens'] / 1000:g}K/c{row['concurrency']}" for row in missing) + '; ' + args.incomplete_reason
        footer += '\n' + '\n'.join(textwrap.wrap(note, width=140))
    fig.supxlabel(footer,
                  fontsize=9)
    svg = args.output / 'quasar-concurrency.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    print('| Input | Submitted | Decode tok/s | Mean batch | Group seconds | Including prefill tok/s |')
    print('|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        print(f"| {row['input_tokens']} | {row['concurrency']} | {row['total_decode_tok_s']:.2f} | "
              f"{row['mean_decode_batch']:.2f} | {row['group_makespan_seconds']:.2f} | "
              f"{row['group_output_tok_s']:.2f} |")
    for row in missing:
        print(f"| {row['input_tokens']} | {row['concurrency']} | — | — | — | — |")


if __name__ == '__main__':
    main()
