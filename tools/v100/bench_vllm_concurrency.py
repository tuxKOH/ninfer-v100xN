#!/usr/bin/env python3
"""Measure a fixed wave of concurrent 1Cat/vLLM streaming completions.

Counts returned token IDs, never SSE chunks or retokenized text. Reports client
common-window and complete-wave throughput separately; neither is executor-only
round time. The supplied checkpoint/KV descriptions are required provenance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid


def speculative_counters(base_url, timeout):
    names = {'vllm:spec_decode_num_drafts_total': 'draft_rounds',
             'vllm:spec_decode_num_draft_tokens_total': 'proposed_drafts',
             'vllm:spec_decode_num_accepted_tokens_total': 'accepted_drafts'}
    with urllib.request.urlopen(base_url.rstrip('/') + '/metrics', timeout=timeout) as response:
        lines = response.read().decode().splitlines()
    counters = {}
    for line in lines:
        match = re.fullmatch(r'([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?', line)
        if match and match[1] in names:
            key = names[match[1]]
            counters[key] = counters.get(key, 0) + float(match[2])
    if set(counters) != set(names.values()):
        raise RuntimeError('server did not expose complete MTP counters')
    return counters


def request(base_url, model, prompt, output_tokens, barrier, timeout):
    payload = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
               "temperature": 0, "top_p": 1, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}, "return_token_ids": True,
               "cache_salt": uuid.uuid4().hex}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    start = time.perf_counter()
    chunks, ids = [], []
    usage = finish = None
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                item = json.loads(data)
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    returned = choice.get("token_ids") or []
                    if returned:
                        chunks.append({"time": time.perf_counter(), "tokens": len(returned)})
                        ids.extend(returned)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    end = time.perf_counter()
    if (not done or not usage or usage.get("prompt_tokens") != len(prompt)
            or usage.get("completion_tokens") != output_tokens
            or len(ids) != output_tokens or finish != "length" or len(chunks) < 2):
        raise RuntimeError("incomplete or uncountable fixed-window HTTP generation")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if cached:
        raise RuntimeError(f"cold wave unexpectedly reused {cached} prompt tokens")
    return {"start": start, "end": end, "chunks": chunks, "token_ids": ids,
            "usage": usage, "ttft_seconds": chunks[0]["time"] - start,
            "decode_seconds": chunks[-1]["time"] - chunks[0]["time"],
            "first_chunk_tokens": chunks[0]["tokens"]}


def summarize(rows):
    origin = min(row["start"] for row in rows)
    end = max(row["end"] for row in rows)
    live_start = max(row["chunks"][0]["time"] for row in rows)
    live_end = min(row["chunks"][-1]["time"] for row in rows)
    live_seconds = max(0., live_end - live_start)
    live_tokens = sum(chunk["tokens"] for row in rows for chunk in row["chunks"]
                      if live_start < chunk["time"] <= live_end)
    output_tokens = sum(len(row["token_ids"]) for row in rows)
    request_decode_seconds = sum(row["decode_seconds"] for row in rows)
    after_first = sum(len(row["token_ids"]) - row["first_chunk_tokens"] for row in rows)
    summary = {"group_makespan_seconds": end - origin,
               "group_output_tok_s": output_tokens / (end - origin),
               "request_capacity_tok_s": len(rows) * after_first / request_decode_seconds,
               "all_live_client_seconds": live_seconds,
               "all_live_client_tokens": live_tokens,
               "all_live_client_tok_s": live_tokens / live_seconds if live_seconds else None}
    for row in rows:
        row["start"] -= origin
        row["end"] -= origin
        for chunk in row["chunks"]:
            chunk["time"] -= origin
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kv-format", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--tp", type=int, choices=[2, 4], required=True)
    parser.add_argument("--capacity", type=int, required=True)
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--corpus", type=Path)
    corpus.add_argument("--corpus-dir", type=Path,
                        help="Exact complete prompts named code-chat-<input tokens>.ids")
    parser.add_argument("--input-tokens", type=int, nargs='+', default=[2048])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, nargs="+", choices=[1, 2, 4, 8], default=[1, 8])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if (any(length < 1 or length + args.output_tokens > args.capacity for length in args.input_tokens)
            or args.output_tokens < 2 or args.repetitions < 1):
        raise ValueError("invalid fixed-window workload")
    args.output.mkdir(parents=True, exist_ok=True)
    request(args.base_url, args.model, [198, 220, 198, 220], 32, None, args.timeout)
    time.sleep(2)  # Exclude the warmup from the first MTP metric interval.
    for length in args.input_tokens:
        prompt = (args.corpus_dir / f'code-chat-{length}.ids' if args.corpus_dir else args.corpus)
        ids = [int(word) for word in prompt.read_text().split()]
        if len(ids) < length or (args.corpus_dir and len(ids) != length):
            raise ValueError(f'prompt length mismatch: {prompt}')
        ids = ids[:length]
        for concurrency in args.concurrency:
            for repetition in range(1, args.repetitions + 1):
                before = speculative_counters(args.base_url, args.timeout)
                barrier = threading.Barrier(concurrency)
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(request, args.base_url, args.model, ids,
                                               args.output_tokens, barrier, args.timeout)
                               for _ in range(concurrency)]
                    rows = [future.result() for future in futures]
                # vLLM publishes counters periodically, independently of SSE completion.
                time.sleep(2)
                after = speculative_counters(args.base_url, args.timeout)
                counters = {key: after[key] - value for key, value in before.items()}
                if (any(value < 0 for value in counters.values())
                        or counters['proposed_drafts'] <= 0
                        or counters['accepted_drafts'] > counters['proposed_drafts']):
                    raise RuntimeError('invalid MTP counter interval')
                counters['draft_acceptance_rate'] = counters['accepted_drafts'] / counters['proposed_drafts']
                report = {"artifact_type": "vllm_concurrent_http_wave",
                          "runtime_version": args.runtime_version, "checkpoint": args.checkpoint,
                          "kv_format": args.kv_format, "tp": args.tp, "mtp_drafts": 4,
                          "capacity": args.capacity, "input_tokens": length,
                          "prompt_fixture": prompt.name,
                          "output_tokens_per_request": args.output_tokens, "concurrency": concurrency,
                          "repetition": repetition, "temperature": 0, "prefix_reuse": False,
                          **summarize(rows), **counters, "requests": rows}
                path = args.output / f'{length}-c{concurrency}-rep{repetition}.json'
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in report.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
