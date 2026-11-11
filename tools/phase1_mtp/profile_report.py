#!/usr/bin/env python3
"""Summarize CUDA kernel execution and memcpy transfers from an Nsight Systems SQLite export.

Runs on the .sqlite file created via nsys export --type sqlite.
No assumed device mapping: compare device IDs with Nsight's GPU table and --list-devices.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path


def available_columns(conn, table):
    return {row[1] for row in conn.execute('PRAGMA table_info("'+table+'")')}


def summarize(sqlite_file: Path):
    conn = sqlite3.connect(f'file:{sqlite_file.resolve()}?mode=ro', uri=True)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        output = {'input':str(sqlite_file.resolve()), 'tables': {}, 'warnings': []}
        for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY'):
            if name not in tables:
                output['warnings'].append(f'{name} unavailable: collection may be unsupported by WSL2/CUPTI or capture incomplete')
                continue
            cols = available_columns(conn, name)
            group_col = 'deviceId' if 'deviceId' in cols else 'device' if 'device' in cols else None
            if not group_col:
                output['warnings'].append(f'{name} lacks a device column; found {sorted(cols)}')
                continue
            extras = ''
            if {'start', 'end'} <= cols:
                extras += ', SUM(CASE WHEN "end">="start" THEN "end"-"start" ELSE 0 END)/1000000.0 AS cumulative_gpu_ms'
            if name.endswith('MEMCPY') and 'bytes' in cols:
                extras += ', SUM(bytes) AS bytes'
            query = f'SELECT {group_col} AS device_id, COUNT(*) AS count{extras} FROM {name} GROUP BY {group_col} ORDER BY {group_col}'
            output['tables'][name] = [dict(zip([d[0] for d in cur.description], row))
                                      for cur in [conn.execute(query)] for row in cur]
        return output
    finally:
        conn.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', nargs='+', type=Path, help='Nsight Systems SQLite exports')
    args=p.parse_args()
    for path in args.sqlite:
        print(json.dumps(summarize(path), indent=2))


if __name__ == '__main__':
    main()
