#!/usr/bin/env python3
"""Publish complete-code 2K/85K concurrency measurements for three NInfer routes and 1Cat."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


LENGTHS = (2048, 85000)
CONCURRENCIES = (1, 2, 4, 8)
ROUTES = {
    'tp4_nvlink': ('NInfer TP4 NVLink', '#2563eb', 4),
    'tp2_nvlink': ('NInfer TP2 NVLink', '#0f766e', 2),
    'tp2_pcie': ('NInfer TP2 PCIe (local)', '#b45309', 2),
    'onecat_tp4': ('1Cat 1.5.2 TP4', '#9333ea', 4),
}


def ninfer_case(path, length, concurrency, tp, kv_capacity=180000):
    source = json.loads(path.read_text())
    required = {'input_tokens': length, 'output_tokens_per_request': 512, 'tp': tp,
                'concurrency': concurrency, 'mtp_drafts': 4, 'graphs': True,
                'max_context': 180000, 'kv_capacity': kv_capacity,
                'prefill_chunk': 1024, 'prefill_decode_rounds': 1,
                'long_attention': True, 'fast_decode': True, 'prefix_reuse': False,
                'prompt_fixture': f'code-chat-{length}.ids',
                'committed_decode_tokens': concurrency * 511}
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError(f'workload mismatch: {path}')
    requests = source['requests']
    if (len(requests) != concurrency or any(
            row['prompt_tokens'] != length or row['completion_tokens'] != 512
            or row['cached_tokens'] != 0 or len(row['generated_token_ids']) != 512
            or not 0 <= row['accepted'] <= row['drafted'] for row in requests)):
        raise ValueError(f'incomplete fixed wave: {path}')
    if (source['accepted_drafts'] != sum(row['accepted'] for row in requests)
            or source['proposed_drafts'] != sum(row['drafted'] for row in requests)):
        raise ValueError(f'MTP counter mismatch: {path}')
    compact = {key: value for key, value in source.items()
               if key not in ('requests', 'scheduler_samples')}
    request_wall = sum(row['total_seconds'] - row['first_token_seconds'] for row in requests)
    compact['request_capacity_tok_s'] = concurrency * source['committed_decode_tokens'] / request_wall
    return compact


def onecat_case(path, length, concurrency, repetition=1):
    source = json.loads(path.read_text())
    required = {'tp': 4, 'mtp_drafts': 4, 'input_tokens': length,
                'output_tokens_per_request': 512, 'concurrency': concurrency,
                'repetition': repetition, 'prefix_reuse': False, 'temperature': 0,
                'capacity': 180000, 'prompt_fixture': f'code-chat-{length}.ids',
                'runtime_version': '1Cat 1.5.2+cu128',
                'checkpoint': 'Qwen3.8-27B-QUASAR-NVFP4 (existing HF checkpoint)',
                'kv_format': 'FP16 activations / FP8 E4M3 KV'}
    if any(source.get(key) != value for key, value in required.items()):
        raise ValueError(f'1Cat workload mismatch: {path}')
    if (len(source['requests']) != concurrency or any(
            len(row['token_ids']) != 512 or row['usage']['prompt_tokens'] != length
            or row['usage']['completion_tokens'] != 512
            or (row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) != 0
            or row['first_chunk_tokens'] != 1
            or sum(chunk['tokens'] for chunk in row['chunks']) != 512
            for row in source['requests'])):
        raise ValueError(f'incomplete 1Cat wave: {path}')
    compact = {key: value for key, value in source.items() if key != 'requests'}
    # The union excludes periods when no request is decoding, but retains any
    # prefill/queueing stalls within a request's first-to-last token lifetime.
    intervals = sorted((row['chunks'][0]['time'], row['chunks'][-1]['time'])
                       for row in source['requests'])
    start, end = intervals[0]
    seconds = 0.
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            seconds += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    seconds += end - start
    tokens = sum(len(row['token_ids']) - row['first_chunk_tokens']
                 for row in source['requests'])
    compact['active_decode_union_seconds'] = seconds
    compact['active_decode_union_tokens'] = tokens
    compact['active_decode_union_tok_s'] = tokens / seconds
    return compact


def onecat_settings(manifest):
    if manifest['versions']['1cat-vllm'] != '1.5.2+cu128':
        raise ValueError('1Cat server manifest is not release 1.5.2')
    command = manifest['server_command']
    expected = {'--tensor-parallel-size': '4', '--dtype': 'half',
                '--kv-cache-dtype': 'fp8_e4m3', '--max-model-len': '180000',
                '--max-num-seqs': '8', '--max-num-batched-tokens': '1024',
                '--block-size': '2048', '--attention-backend': 'FLASH_ATTN_V100'}
    for flag, value in expected.items():
        if command[command.index(flag) + 1] != value:
            raise ValueError(f'1Cat server setting mismatch: {flag}')
    spec = json.loads(command[command.index('--speculative-config') + 1])
    if spec != {'method': 'mtp', 'num_speculative_tokens': 4, 'draft_sample_method': 'greedy'}:
        raise ValueError('1Cat server is not greedy MTP4')
    if '--no-enable-prefix-caching' not in command or manifest['physical_kv_tokens'] < 1:
        raise ValueError('invalid 1Cat cache configuration')
    return {'versions': manifest['versions'], 'checkpoint': manifest['checkpoint'],
            'server_command': command,
            'kv_cache_memory_bytes_per_rank': int(command[command.index('--kv-cache-memory-bytes') + 1]),
            'physical_kv_tokens': manifest['physical_kv_tokens']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp4-nvlink', type=Path, required=True)
    parser.add_argument('--tp2-nvlink', type=Path, required=True)
    parser.add_argument('--tp2-pcie', type=Path, required=True)
    parser.add_argument('--onecat', type=Path, required=True,
                        help='1.5.2 campaign directory with manifest.json and results/')
    parser.add_argument('--tp4-expanded', type=Path,
                        help='Separate 85K/C8 capacity control with 688000 shared KV tokens')
    parser.add_argument('--tp4-repeat', type=Path,
                        help='Release-build repeat of the complete 2K/C8 prompt')
    parser.add_argument('--tp4-before', type=Path,
                        help='Pre-TP4-batch implementation, same saved complete 2K/C8 prompt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path, help='Optional local PNG for visual inspection')
    args = parser.parse_args()
    onecat_manifest = json.loads((args.onecat / 'manifest.json').read_text())
    onecat_by_input = {str(length): onecat_settings(onecat_manifest['by_input'][str(length)])
                      for length in LENGTHS}
    if any('--long-prefill-token-threshold' in row['server_command']
           for row in onecat_by_input.values()):
        raise ValueError('main matrix must use default long-prefill scheduling')
    cases = []
    for route, directory in [('tp4_nvlink', args.tp4_nvlink),
                             ('tp2_nvlink', args.tp2_nvlink), ('tp2_pcie', args.tp2_pcie)]:
        for length in LENGTHS:
            for concurrency in CONCURRENCIES:
                row = ninfer_case(directory / f'{length}-c{concurrency}.json',
                                  length, concurrency, ROUTES[route][2])
                cases.append({'route': route, **row})
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            row = onecat_case(args.onecat / 'results' / f'{length}-c{concurrency}-rep1.json',
                              length, concurrency)
            cases.append({'route': 'onecat_tp4', **row})
    summary = {
        'date': '2026-11-08',
        'ninfer_measurement_date': '2026-11-05',
        'remote_hardware': '4 x V100-SXM2 16 GB, complete pairwise NV2, existing 300 W limits',
        'local_hardware': '2 x V100-SXM2 16 GB, PCIe P2P, no NVLink',
        'toolchain': 'CUDA 12.8, SM70; remote driver 570.211.01, local driver 580.178.04',
        'ninfer': 'QUASAR NVFP4 v3 / BF16 / INT8 group-64 KV; long/fast enabled',
        'onecat': '1Cat 1.5.2+cu128 / QUASAR NVFP4 HF checkpoint / FP16 / E4M3 KV; '
                  'isolated TileLang 0.1.10 / apache-tvm-ffi 0.1.10',
        'workload': 'Complete code-chat-2048.ids or code-chat-85000.ids, MTP4, greedy, '
                    '512 forced output tokens per request, cold independent request waves, '
                    'no prefix or historical reuse. One run per matrix case.',
        'method': 'NInfer headline decode divides committed tokens after the first output '
                  'by summed executor decode-round wall time. All-live uses 10 ms executor '
                  'samples for NInfer and returned SSE token IDs for 1Cat. Whole-wave includes '
                  'prefill and queueing. 1Cat request_capacity_tok_s is an HTTP per-request '
                  'estimate, not NInfer executor-round time. Non-overlapping streams have no '
                  'common all-live rate. 180K is NInfer shared KV capacity and 1Cat per-request '
                  'max-model-len, not a claim of equal physical KV pools. '
                  'active_decode_union_tok_s divides all post-first outputs by the union of '
                  'individual first-to-last-token lifetimes, including stalls within those '
                  'lifetimes; this is not a common all-request window or steady C8 rate.',
        'quality': 'Different artifacts, KV and arithmetic profiles; no equal-quality claim or new GPQA.',
        'onecat_settings_by_input': onecat_by_input,
        'cases': cases,
    }
    repeated = onecat_case(args.onecat / 'results' / '2048-c8-rep2.json', 2048, 8, 2)
    first = next(row for row in cases if row['route'] == 'onecat_tp4'
                 and row['input_tokens'] == 2048 and row['concurrency'] == 8)
    values = [first['all_live_client_tok_s'], repeated['all_live_client_tok_s']]
    mean = sum(values) / 2
    summary['onecat_2k_c8_repeat'] = {
        'case': repeated, 'common_window_mean_tok_s': mean,
        'common_window_sample_sd_tok_s': math.sqrt(sum((value - mean) ** 2 for value in values)),
        'pooled_draft_acceptance_rate': (first['accepted_drafts'] + repeated['accepted_drafts'])
                                      / (first['proposed_drafts'] + repeated['proposed_drafts'])}
    summary['onecat_long_prefill_controls'] = []
    for control in onecat_manifest['long_prefill_controls']:
        settings = onecat_settings(control)
        case = onecat_case(args.onecat / control['result'], 85000, 8)
        summary['onecat_long_prefill_controls'].append({
            'settings': settings, 'case': case,
            'observed_preemptions': control.get('observed_preemptions')})
    if args.tp4_expanded:
        summary['expanded_capacity_control'] = ninfer_case(args.tp4_expanded, 85000, 8, 4, 688000)
    if args.tp4_repeat:
        repeated = ninfer_case(args.tp4_repeat, 2048, 8, 4)
        original = json.loads((args.tp4_nvlink / '2048-c8.json').read_text())['requests']
        observed = json.loads(args.tp4_repeat.read_text())['requests']
        summary['release_build_repeat'] = {
            'case': repeated,
            'complete_output_ids_equal': all(a['generated_token_ids'] == b['generated_token_ids']
                                             for a, b in zip(original, observed)),
            'speculative_counters_equal': all(a[key] == b[key]
                for a, b in zip(original, observed) for key in ('accepted', 'drafted')),
        }
    if args.tp4_before:
        # This frozen executable predates the prompt/counter summary fields, but was
        # invoked directly with these same saved IDs and workload options.
        before = json.loads(args.tp4_before.read_text())
        expected = {'input_tokens': 2048, 'output_tokens_per_request': 512, 'tp': 4,
                    'concurrency': 8, 'mtp_drafts': 4, 'kv_capacity': 180000,
                    'prefill_chunk': 1024, 'prefill_decode_rounds': 1, 'graphs': True,
                    'long_attention': True, 'fast_decode': True, 'prefix_reuse': False}
        if any(before.get(key) != value for key, value in expected.items()):
            raise ValueError('software control workload mismatch')
        old_requests = before['requests']
        current = json.loads((args.tp4_nvlink / '2048-c8.json').read_text())
        if len(old_requests) != 8 or any(
                row['prompt_tokens'] != 2048 or row['completion_tokens'] != 512
                or len(row['generated_token_ids']) != 512 or row['cached_tokens'] != 0
                for row in old_requests):
            raise ValueError('incomplete software control')
        accepted = sum(row['accepted'] for row in old_requests)
        proposed = sum(row['drafted'] for row in old_requests)
        summary['same_prompt_software_control'] = {
            'workload': 'code-chat-2048.ids, TP4/C8, same host and command options',
            'before_decode_tok_s': before['total_decode_tok_s'],
            'current_decode_tok_s': current['total_decode_tok_s'],
            'change_fraction': current['total_decode_tok_s'] / before['total_decode_tok_s'] - 1,
            'before_accepted_drafts': accepted, 'before_proposed_drafts': proposed,
            'before_draft_acceptance_rate': accepted / proposed,
            'before_mean_decode_batch': before['mean_decode_batch'],
            'before_group_makespan_seconds': before['group_makespan_seconds'],
            'complete_output_ids_equal': all(a['generated_token_ids'] == b['generated_token_ids']
                for a, b in zip(old_requests, current['requests'])),
            'speculative_counters_equal': all(a[key] == b[key]
                for a, b in zip(old_requests, current['requests']) for key in ('accepted', 'drafted')),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'quasar-concurrency.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), layout='constrained')
    for i, length in enumerate(LENGTHS):
        for j, (title, nkey, ckey) in enumerate([
                ('NInfer executor decode rounds', 'total_decode_tok_s', None),
                ('Common all-live window', 'all_live_sampled_tok_s', 'all_live_client_tok_s'),
                ('Whole wave, including prefill', 'group_output_tok_s', 'group_output_tok_s')]):
            axis = axes[i, j]
            for route, (label, color, _) in ROUTES.items():
                if route == 'onecat_tp4' and ckey is None:
                    continue
                rows = [row for row in cases if row['route'] == route and row['input_tokens'] == length]
                key = ckey if route == 'onecat_tp4' else nkey
                ys = [row.get(key) if row.get(key) is not None else math.nan for row in rows]
                axis.plot(CONCURRENCIES, ys, 'o--' if route == 'onecat_tp4' else 'o-',
                          label=label, color=color, linewidth=1.8)
                if j == 0:
                    offset = (12 if route == 'tp2_nvlink' else -14
                              if route == 'tp2_pcie' else 7) if i == 1 else 7
                    axis.annotate(f'{ys[-1]:.1f}', (8, ys[-1]), xytext=(0, offset),
                                  textcoords='offset points', ha='center', color=color, fontsize=9)
            axis.set(title=f'{length:,} input: {title}', xlabel='Offered concurrent requests',
                     ylabel='Committed decode tok/s' if j == 0 else 'Output tok/s',
                     xscale='log', xticks=CONCURRENCIES,
                     xticklabels=[str(c) for c in CONCURRENCIES], ylim=(0, None))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.margins(y=.2)
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle('Complete coding task / MTP4 / 512 output tokens per request', fontsize=15)
    fig.supxlabel('NVLink routes share the remote host; PCIe TP2 runs locally. 85K C4/C8 can queue '
                  'at 180K shared KV.\nAll-live timing and model/KV formats differ. Single runs; '
                  'no new GPQA. Missing points: no common decode window.', fontsize=9)
    path = args.output / 'quasar-concurrency.svg'
    fig.savefig(path)
    if args.preview:
        fig.savefig(args.preview, dpi=120)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    for length in LENGTHS:
        print(f'\n{length} input')
        for route, (label, _, _) in ROUTES.items():
            rows = [row for row in cases if row['route'] == route and row['input_tokens'] == length]
            surfaces = [('HTTP common window', 'all_live_client_tok_s'),
                        ('whole wave', 'group_output_tok_s')] if route == 'onecat_tp4' else [
                            ('executor decode', 'total_decode_tok_s')]
            for surface, rate in surfaces:
                values = ' / '.join('—' if row[rate] is None else f'{row[rate]:.3f}' for row in rows)
                print(f'{label} ({surface}): {values}')
            print('acceptance: ' + ' / '.join(f"{row['draft_acceptance_rate']:.1%}" for row in rows))


if __name__ == '__main__':
    main()
