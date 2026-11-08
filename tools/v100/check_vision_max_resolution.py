#!/usr/bin/env python3
"""Submit one maximum-area image and check its answer and preprocessing counts.

Pass requires an isolated request and the server's request JSONL log. The log
reports patch and token counts, not the actual grid dimensions.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


SIZE = 4096
FEATURES = (
    ("top_left", 512, 512, "circle", "red", (255, 40, 40), "A"),
    ("top_right", 3584, 512, "square", "blue", (40, 130, 255), "B"),
    ("center", 2048, 2048, "triangle", "green", (40, 255, 40), "C"),
    ("bottom_left", 512, 3584, "diamond", "yellow", (255, 255, 40), "D"),
    ("bottom_right", 3584, 3584, "cross", "magenta", (255, 40, 255), "E"),
)
GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
}
PROMPT = (
    "The image has five colored shapes on a black background. Identify the color "
    "and shape at each position. Return only a JSON object with keys top_left, "
    "top_right, center, bottom_left, bottom_right. Each value must be an object "
    "with keys color and shape. Use lowercase single words."
)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def fixture_png() -> bytes:
    compressor = zlib.compressobj(level=6)
    compressed = bytearray()
    for y in range(SIZE):
        row = bytearray(SIZE * 3)
        for _, cx, cy, shape, _, color, label in FEATURES:
            dy = y - cy
            half = -1
            if shape == "circle" and abs(dy) <= 220:
                half = math.isqrt(220 * 220 - dy * dy)
            elif shape == "square" and abs(dy) <= 220:
                half = 220
            elif shape == "triangle" and -220 <= dy <= 220:
                half = (dy + 220) // 2
            elif shape == "diamond" and abs(dy) <= 220:
                half = 220 - abs(dy)
            elif shape == "cross" and abs(dy) <= 220:
                half = 220 if abs(dy) <= 65 else 65
            if half >= 0:
                start, end = cx - half, cx + half + 1
                row[start * 3:end * 3] = bytes(color) * (end - start)
            glyph_y = y - (cy + 270)
            if 0 <= glyph_y < 70:
                glyph = GLYPHS[label][glyph_y // 10]
                for column, pixel in enumerate(glyph):
                    if pixel == "1":
                        start = cx - 25 + column * 10
                        row[start * 3:(start + 10) * 3] = b"\xff" * 30
        compressed.extend(compressor.compress(b"\x00" + row))
    compressed.extend(compressor.flush())
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
            + png_chunk(b"IDAT", bytes(compressed)) + png_chunk(b"IEND", b""))


def read_request_evidence(path: str, snapshot: os.stat_result, model: str) -> dict:
    with open(path, "rb") as log:
        current = os.fstat(log.fileno())
        if (current.st_dev, current.st_ino) != (snapshot.st_dev, snapshot.st_ino):
            raise ValueError("request log was replaced")
        if current.st_size < snapshot.st_size:
            raise ValueError("request log was truncated")
        log.seek(snapshot.st_size)
        data = log.read()
    if data and not data.endswith(b"\n"):
        raise ValueError("request log ends with an incomplete record")
    records = [json.loads(line) for line in data.splitlines()]
    request_records = [
        record for record in records
        if isinstance(record, dict) and record.get("event") in
        ("request_start", "request_done", "request_error", "request_rejected")
    ]
    identities = {
        (record.get("server_instance_id"), record.get("request", {}).get("request_id"))
        for record in request_records if isinstance(record.get("request"), dict)
    }
    if len(identities) != 1 or len(request_records) != 2:
        raise ValueError("expected exactly one isolated request start and completion in log window")
    instance_id, request_id = next(iter(identities))
    if not instance_id or type(request_id) is not int:
        raise ValueError("request log has no valid server or request identity")
    starts = [record for record in request_records if record["event"] == "request_start"]
    dones = [record for record in request_records if record["event"] == "request_done"]
    if len(starts) != 1 or len(dones) != 1:
        raise ValueError("request log does not contain one start and one successful completion")
    start = starts[0]
    request_info = start["request"]
    if (request_info.get("protocol") != "openai_chat_completions"
            or request_info.get("model") != model
            or request_info.get("media_item_count") != 1):
        raise ValueError("logged request does not match the single-image request")
    preparation = start.get("preparation_seconds")
    if not isinstance(preparation, dict):
        raise ValueError("request log has no preprocessing counts")
    return {
        "server_instance_id": instance_id,
        "request_id": request_id,
        "correlation": "isolated log window; response ID is independent of log request ID",
        "media_items": preparation.get("media_items"),
        "raw_patches": preparation.get("raw_patches"),
        "vision_tokens": preparation.get("vision_tokens"),
        "preparation_seconds": preparation,
        "timings_seconds": dones[0].get("timings_seconds"),
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, required=True,
                        help="HTTP socket timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-log-jsonl",
                        help="server request log for an isolated request")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_tokens <= 0:
        parser.error("--timeout must be finite and positive; --max-tokens must be positive")

    image = fixture_png()
    expected = {position: {"color": color, "shape": shape}
                for position, _, _, shape, color, _, _ in FEATURES}
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}},
            {"type": "text", "text": PROMPT},
        ]}],
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers=headers, method="POST")
    report = {
        "fixture": {"format": "png", "width": SIZE, "height": SIZE,
                    "sha256": hashlib.sha256(image).hexdigest(), "expected": expected},
        "request": {"url": request.full_url, "model": args.model, "prompt": PROMPT,
                    "image_count": 1, "max_tokens": args.max_tokens,
                    "enable_thinking": False, "socket_timeout_seconds": args.timeout},
        "expected_grid": [1, 256, 256],
        "expected_raw_patches": 65536,
        "expected_visual_tokens": 16384,
        "observed_grid": None,
        "grid_diagnostic": (
            "Grid dimensions are not directly exposed. For this fixture, the logged "
            "patch and token counts establish the full maximum effective pixel area, "
            "not directly observed dimensions."
        ),
        "status": "non-passing: request log evidence unavailable",
    }
    snapshot = None
    if args.request_log_jsonl:
        try:
            snapshot = os.stat(args.request_log_jsonl)
        except OSError as exc:
            report["log_error"] = str(exc)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            report["http_status"] = response.status
            raw = response.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            report["error"] = "Response is not valid JSON"
            report["response_body"] = raw
        else:
            report["response"] = result
            report["usage"] = result.get("usage") if isinstance(result, dict) else None
            try:
                content = result["choices"][0]["message"]["content"]
                text = content.strip()
                if text.startswith("```") and text.endswith("```"):
                    first_line, _, text = text.partition("\n")
                    if first_line not in ("```", "```json"):
                        raise ValueError("unsupported code fence")
                    text = text[:-3].strip()
                actual = json.loads(text)
                report["features_match"] = actual == expected
            except (KeyError, IndexError, TypeError, ValueError):
                report["features_match"] = False
    except urllib.error.HTTPError as exc:
        report["http_status"] = exc.code
        report["error"] = exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        report["http_status"] = None
        report["error"] = str(exc)
    finally:
        report["wall_seconds"] = time.monotonic() - started
    if snapshot is not None:
        try:
            evidence = read_request_evidence(args.request_log_jsonl, snapshot, args.model)
            report["request_log"] = evidence
            counts = (evidence["media_items"], evidence["raw_patches"], evidence["vision_tokens"])
            report["counts_match"] = (
                all(type(value) is int for value in counts) and counts == (1, 65536, 16384)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            report["log_error"] = str(exc)
    if "error" in report:
        report["status"] = "non-passing: request or response failed"
    elif report.get("features_match") is False:
        report["status"] = "non-passing: features do not match"
    elif report.get("counts_match") is False:
        report["status"] = "non-passing: preprocessing counts do not match"
    elif report.get("features_match") is True and report.get("counts_match") is True:
        report["status"] = "pass: full maximum effective pixel area and features"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"].startswith("pass:") else 1


if __name__ == "__main__":
    sys.exit(main())
