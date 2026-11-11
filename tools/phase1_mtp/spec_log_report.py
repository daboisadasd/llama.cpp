#!/usr/bin/env python3
"""Read cumulative native MTP draft timing and transfer-path diagnostics from verbose llama-server logs.

Requires server --verbosity 4 to show common_speculative_print_stats().
The draft timer includes host-side and GPU sync in common_speculative_draft,
so it is not a pure GPU-kernel timing. Difference snapshots to exclude warmup.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

STATS = re.compile(
    r'statistics\s+draft-mtp\s*:\s*#calls\(b,g,a\)\s*=\s*'
    r'(\d+)\s+(\d+)\s+(\d+).*?dur\(b,g,a\)\s*=\s*'
    r'([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*ms')
DEVICE = re.compile(r'target h rows on the device:\s*(yes|no)', re.I)


def parse_log(text):
    events = []
    for m in STATS.finditer(text):
        fields = m.groups()
        events.append(dict(zip(
            ['begin_calls','draft_calls','accept_calls','begin_ms','draft_ms','accept_ms'],
            [int(v) if i < 3 else float(v) for i,v in enumerate(fields)])))
    state = DEVICE.findall(text)
    result = {'cumulative_snapshots':events, 'device_hidden_state_path':state[-1] if state else None}
    if len(events) >= 2:
        first, last = events[0], events[-1]
        delta = {k:round(last[k]-first[k], 3) for k in last}
        result['after_first_snapshot_delta'] = delta
        if delta['draft_calls'] > 0:
            result['mean_draft_ms_per_draft_call'] = round(delta['draft_ms']/delta['draft_calls'],4)
    elif events:
        result['note'] = 'One snapshot; includes warmup, cannot isolate measured requests.'
    else:
        result['note'] = 'No MTP trace stats; run the profiling script with --verbosity 4.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', type=Path)
    args = p.parse_args()
    for path in args.files:
        output={'file':str(path), **parse_log(path.read_text(encoding='utf-8',errors='replace'))}
        print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
