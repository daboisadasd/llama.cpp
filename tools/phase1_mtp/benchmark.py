#!/usr/bin/env python3
"""Phase 1 A/B/C llama.cpp MTP benchmark: measured tokens, strict device flags, GPU telemetry.

Requires only Python 3.10+, llama-server with CUDA, target GGUF and MTP GGUF.
Use --help for a smoke test or a longer benchmark matrix. Never synthesizes results.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_gguf import inspect as inspect_gguf

MTP_DEVICES = {'A': 'target only', 'B': 'MTP on primary', 'C': 'MTP on secondary'}
PROMPT_SHORT = (
    'Write a detailed engineering explanation of speculative decoding in llama.cpp. '
    'Discuss hidden states, cache synchronization, verification, and useful tokens per second. '
    'Use precise technical language with several paragraphs and concrete examples.\n'
)
LONG_SENTENCE = (
    'Observation: the host performs transformer verification while a separate device performs '
    'native MTP prediction; logits, hidden states, cache positions, and synchronization must stay coherent. '
)


def numlist(text: str, min_value: int = 1):
    result = [int(v.strip()) for v in text.split(',') if v.strip()]
    if not result or min(result) < min_value:
        raise argparse.ArgumentTypeError(f'expected comma-separated integers >= {min_value}')
    return list(dict.fromkeys(result))


def prompt_for(profile: str, n_ctx: int):
    if profile == 'short':
        return PROMPT_SHORT
    # ~3.5 English characters per desired context token. Tokenization varies,
    # so this only approximates a long prefill; actual token count is recorded.
    prefix_bytes = max(300, int(n_ctx * 3.5))
    return (LONG_SENTENCE * (1 + prefix_bytes // len(LONG_SENTENCE)))[:prefix_bytes] + '\n' + PROMPT_SHORT


def build_command(a, mode, ctx, depth, port):
    cmd = [str(a.server), '--model', str(a.target), '--device', a.primary_device,
           '--n-gpu-layers', str(a.target_ngl), '--ctx-size', str(ctx),
           '--batch-size', str(a.batch_size), '--ubatch-size', str(a.ubatch_size),
           '--parallel', '1', '--flash-attn', a.flash_attn, '--fit', 'off',
           '--host', '127.0.0.1', '--port', str(port), '--load-mode', 'none' if a.no_mmap else 'auto']
    if mode == 'A':
        cmd += ['--spec-type', 'none']
    else:
        draft = a.primary_device if mode == 'B' else a.draft_device
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-model', str(a.mtp),
                '--spec-draft-device', draft, '--spec-draft-ngl', 'all',
                '--spec-draft-n-max', str(depth)]
    cmd += a.extra_server_arg
    return cmd


def read_log_tail(path: Path, n: int = 45):
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-n:])
    except OSError:
        return '<log not readable>'


def request_json(method, url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = Request(url, method=method, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode('utf-8'))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def check_devices(server: Path, primary: str, draft: str, expect_primary: str, expect_draft: str):
    """Preflight exact GGML device names; never permit an invisible CUDA device to fall back."""
    cp = subprocess.run([str(server), '--list-devices'], capture_output=True,
                        text=True, timeout=40)
    output = cp.stdout + '\n' + cp.stderr
    if cp.returncode != 0:
        raise RuntimeError('llama-server --list-devices failed: ' + output[-2000:])
    available = {}
    for line in output.splitlines():
        match = re.match(r'^\s*(CUDA[0-9]+):\s*(.+?)\s*$', line)
        if match:
            available[match.group(1)] = match.group(2)
    if primary not in available or draft not in available:
        raise RuntimeError('Required CUDA devices are absent. Seen: ' + str(available) +
                           '. Verify both GPUs are visible inside WSL2; do not use a fallback device.')
    for dev, expected in ((primary, expect_primary), (draft, expect_draft)):
        if expected and expected.lower() not in available[dev].lower():
            raise RuntimeError(f'{dev} identifies as {available[dev]!r}, not expected {expected!r}. '
                               'Set --primary-device/--draft-device to match --list-devices.')
    print('DEVICE PREFLIGHT', json.dumps({'target': {primary:available[primary]},
                                           'MTP': {draft:available[draft]}}), flush=True)
    return available


class GPUProbe:
    def __init__(self, executable: str, interval: float):
        self.exe, self.interval = executable, interval
        self.samples = []
        self.errors = set()
        self.phase = 'startup'
        self._stop = threading.Event()
        self.thread = None
        self._lock = threading.Lock()

    @staticmethod
    def parse(output, phase, now):
        rows = []
        for line in output.splitlines():
            parts = [v.strip() for v in line.split(',')]
            if len(parts) not in (4, 6):
                continue
            def number(s):
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None
            rows.append(dict(timestamp_utc=now, phase=phase, index=parts[0], uuid=parts[1],
                             name=parts[2], memory_mib=number(parts[3]),
                             gpu_util_pct=number(parts[4]) if len(parts) == 6 else None,
                             memory_util_pct=number(parts[5]) if len(parts) == 6 else None))
        return rows

    def _poll(self):
        if self.exe == 'disabled':
            self.errors.add('nvidia-smi unavailable: GPU telemetry disabled for this run')
            return
        while not self._stop.is_set():
            now = datetime.now(timezone.utc).isoformat()
            try:
                query = 'index,uuid,name,memory.used,utilization.gpu,utilization.memory'
                cp = subprocess.run([self.exe, '--query-gpu=' + query,
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=6)
                if cp.returncode != 0:
                    # NVML utilization is not available in some WSL2/driver builds.
                    cp = subprocess.run([self.exe, '--query-gpu=index,uuid,name,memory.used',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=6)
                if cp.returncode != 0:
                    raise RuntimeError(cp.stderr.strip()[:200])
                with self._lock:
                    self.samples += self.parse(cp.stdout, self.phase, now)
            except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError, OSError) as e:
                self.errors.add(str(e))
            self._stop.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def close(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=7)

    def summary(self):
        by_gpu = {}
        for rec in self.samples:
            name = rec['name'] + ' (' + rec['index'] + ')'
            by_gpu.setdefault(name, []).append(rec)
        results = {}
        for name, samples in by_gpu.items():
            trial_samples = [s for s in samples if s['phase'] == 'trial']
            util = [s['gpu_util_pct'] for s in trial_samples if s['gpu_util_pct'] is not None]
            mem = [s['memory_mib'] for s in samples if s['memory_mib'] is not None]
            results[name] = {
                'trial_samples': len(trial_samples),
                'mean_generation_gpu_util_pct': statistics.mean(util) if util else None,
                'max_memory_used_mib': max(mem) if mem else None,
                'min_memory_used_mib': min(mem) if mem else None,
            }
        return results


@dataclass(frozen=True)
class Case:
    mode: str
    n_ctx: int
    depth: int

    def name(self):
        return f'{self.mode}_ctx{self.n_ctx}_depth{self.depth}'


def cases(contexts, depths, modes):
    for ctx in contexts:
        if 'A' in modes:
            yield Case('A', ctx, 0)
        for depth in depths:
            for mode in ('B', 'C'):
                if mode in modes:
                    yield Case(mode, ctx, depth)


def completion_payload(prompt, n_predict, seed):
    return {'prompt': prompt, 'n_predict': n_predict, 'temperature': 0.0,
            'seed': seed, 'stream': False, 'cache_prompt': False,
            'return_tokens': True, 'repeat_penalty': 1.0, 'dry_multiplier': 0.0,
            'top_k': 1, 'ignore_eos': False}


def flatten_result(result, case, profile, n_predict, repeat, latency_s):
    timings = result.get('timings', {})
    if not isinstance(timings, dict):
        timings = {}
    def numeric(field, default=0):
        return timings.get(field) if isinstance(timings.get(field), (float, int)) else default
    n_pred = int(numeric('predicted_n'))
    draft_n, accepted = int(numeric('draft_n')), int(numeric('draft_n_accepted'))
    if draft_n < 0 or accepted < 0 or accepted > draft_n:
        raise RuntimeError(f'invalid draft counters generated={draft_n} accepted={accepted}')
    if n_pred <= 0:
        raise RuntimeError(f'no predicted tokens in response: {str(result)[:650]}')
    if case.mode != 'A' and draft_n <= 0:
        raise RuntimeError('MTP was requested but no draft tokens were recorded. Check source, model, and logs.')
    if case.mode == 'A' and draft_n > 0:
        raise RuntimeError('baseline unexpectedly generated draft tokens')
    token_ids = result.get('tokens')
    text = result.get('content', '')
    if token_ids is not None and not isinstance(token_ids, list):
        token_ids = None
    identity = ('ids:' + json.dumps(token_ids, separators=(',', ':')) if token_ids else 'text:' + str(text))
    return {
        'case': case.name(), 'mode': case.mode, 'n_ctx': case.n_ctx, 'depth': case.depth,
        'prompt_profile': profile, 'max_tokens': n_predict, 'repeat': repeat,
        'predicted_n': n_pred, 'prompt_n': int(numeric('prompt_n')),
        'predicted_ms': numeric('predicted_ms', None), 'prompt_ms': numeric('prompt_ms', None),
        'predicted_tokens_per_second': numeric('predicted_per_second', None),
        'wall_seconds': latency_s, 'draft_n': draft_n, 'draft_n_accepted': accepted,
        'draft_accept_rate': accepted/draft_n if draft_n else None,
        'output_sha256': hashlib.sha256(identity.encode('utf-8')).hexdigest(),
        'stop_type': result.get('stop_type'), 'truncated': result.get('truncated'),
        'has_token_ids': bool(token_ids), 'output_excerpt': str(text)[:160],
    }, {'content': text, 'tokens': token_ids, 'timings': timings}


def ensure_models(args):
    for p in (args.target, args.mtp):
        if not p.is_file():
            raise RuntimeError(f'missing model file: {p} (run download_models.sh)')
    target, draft = inspect_gguf(args.target), inspect_gguf(args.mtp)
    arch_t = target['metadata'].get('general.architecture')
    arch_d = draft['metadata'].get('general.architecture')
    if arch_t != arch_d:
        raise RuntimeError(f'GGUF architecture mismatch: target {arch_t!r} / MTP {arch_d!r}')
    if arch_t not in ('qwen35', 'qwen35moe'):
        raise RuntimeError(f'expected Qwen3.8 GGUF qwen35 architecture, got {arch_t!r}')
    nextn = [v for k,v in draft['metadata'].items() if k.endswith('nextn_predict_layers')]
    if not nextn or not any(isinstance(v,int) and v > 0 for v in nextn):
        raise RuntimeError('MTP sidecar lacks positive nextn_predict_layers metadata')
    if draft['tensor_count'] < 1 or target['tensor_count'] < 1:
        raise RuntimeError('GGUF tensor header is empty')
    return {'target': target, 'mtp': draft}


def run_server(a, case, outdir, metadata, command_builder=None):
    port = a.port if a.port else free_port()
    logpath = outdir / (case.name() + '.server.log')
    gpupath = outdir / (case.name() + '.gpu.csv')
    args_path = outdir / (case.name() + '.command.json')
    cmd = (command_builder or build_command)(a, case.mode, case.n_ctx, case.depth, port)
    (args_path).write_text(json.dumps({'argv':cmd, 'shell_hint': subprocess.list2cmdline(cmd)}, indent=2))
    env = os.environ.copy()
    if a.trace:
        env['LLAMA_TRACE'] = '1'
    actual_command = cmd
    if a.nsys:
        if not shutil.which('nsys'):
            raise RuntimeError('--nsys requested but nsys not available on PATH')
        actual_command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none',
                          '--output=' + str(outdir / case.name()), '--force-overwrite=true'] + cmd
    probe = GPUProbe(a.nvidia_smi, a.gpu_sample_seconds)
    proc = None
    records, outputs = [], []
    print('RUN', case.name(), flush=True)
    try:
        with logpath.open('w', encoding='utf-8') as logfile:
            probe.start()
            proc = subprocess.Popen(actual_command, stdout=logfile, stderr=subprocess.STDOUT,
                                    cwd=str(a.llama_repo) if a.llama_repo else None, env=env,
                                    start_new_session=(os.name == 'posix'))
            t0 = time.monotonic()
            while time.monotonic()-t0 < a.load_timeout:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode} during startup\n{read_log_tail(logpath)}')
                try:
                    healthy = request_json('GET', f'http://127.0.0.1:{port}/health', timeout=2)
                    if healthy.get('status') == 'ok':
                        break
                except (HTTPError, URLError, ConnectionError, ValueError, TimeoutError):
                    pass
                time.sleep(.7)
            else:
                raise TimeoutError(f'server health startup timed out after {a.load_timeout}s; tail:\n{read_log_tail(logpath)}')
            probe.phase = 'warmup'
            request_json('POST', f'http://127.0.0.1:{port}/completion',
                         completion_payload(PROMPT_SHORT, 12, a.seed), timeout=a.request_timeout)
            probe.phase = 'trial'
            for profile in a.prompt_profiles:
                prompt = prompt_for(profile, case.n_ctx)
                for n_predict in a.generations:
                    for repeat in range(a.repeats):
                        data_in = completion_payload(prompt, n_predict, a.seed)
                        t0 = time.perf_counter()
                        result = request_json('POST', f'http://127.0.0.1:{port}/completion',
                                              data_in, timeout=a.request_timeout)
                        secs = time.perf_counter()-t0
                        row, output = flatten_result(result, case, profile, n_predict, repeat, secs)
                        records.append(row)
                        outputs.append({'prompt_profile': profile, 'max_tokens': n_predict,
                                        'repeat': repeat, 'data': output})
                        print('  ', profile, n_predict, 'repeat', repeat,
                              't/s', row['predicted_tokens_per_second'],
                              'draft', row['draft_n'], 'accepted', row['draft_n_accepted'], flush=True)
            probe.phase = 'shutdown'
    finally:
        if proc and proc.poll() is None:
            # Nsight wraps the actual server: signal the entire process group to
            # avoid an orphan server and give the profiler a chance to flush.
            if a.nsys and os.name == 'posix':
                os.killpg(proc.pid, signal.SIGINT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=25 if a.nsys else 15)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.kill()
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait(timeout=5)
        probe.close()
        with gpupath.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp_utc','phase','index','uuid','name',
                'memory_mib','gpu_util_pct','memory_util_pct'])
            writer.writeheader()
            writer.writerows(probe.samples)
        (outdir / (case.name()+'.gpu_summary.json')).write_text(json.dumps({
            'gpu_summary': probe.summary(), 'errors': sorted(probe.errors)}, indent=2))
        (outdir / (case.name()+'.outputs.json')).write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    return records


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row['mode'], row['n_ctx'], row['depth'], row['prompt_profile'], row['max_tokens'])
        grouped.setdefault(key, []).append(row)
    output = []
    for (mode,ctx,depth,profile,budget), trials in sorted(grouped.items()):
        tps = [v['predicted_tokens_per_second'] for v in trials if isinstance(v['predicted_tokens_per_second'], (float,int))]
        pred_ms = [v['predicted_ms'] for v in trials if isinstance(v['predicted_ms'], (float,int))]
        prompt_ms = [v['prompt_ms'] for v in trials if isinstance(v['prompt_ms'], (float,int))]
        gen = sum(v['draft_n'] for v in trials)
        accepted = sum(v['draft_n_accepted'] for v in trials)
        record = {'mode': mode, 'context': ctx, 'depth': depth, 'prompt': profile,
                  'max_tokens': budget, 'trials':len(trials),
                  'median_accepted_output_tokens_per_sec': statistics.median(tps) if tps else None,
                  'median_generation_ms': statistics.median(pred_ms) if pred_ms else None,
                  'median_prefill_ms': statistics.median(prompt_ms) if prompt_ms else None,
                  'drafts_generated': gen, 'drafts_accepted': accepted,
                  'acceptance_rate': accepted/gen if gen else None,
                  'output_consistent_within_case': len({v['output_sha256'] for v in trials}) == 1}
        output.append(record)
    baselines = {(r['context'],r['prompt'],r['max_tokens']):r
                 for r in output if r['mode']=='A'}
    for r in output:
        b = baselines.get((r['context'],r['prompt'],r['max_tokens']))
        tps, ref = r['median_accepted_output_tokens_per_sec'], (b or {}).get('median_accepted_output_tokens_per_sec')
        r['speedup_vs_A'] = (tps/ref) if tps and ref else None
    return output


def compare_to_baseline(rows):
    baselines = {(r['n_ctx'],r['prompt_profile'],r['max_tokens'],r['repeat']):r['output_sha256']
                 for r in rows if r['mode']=='A'}
    mismatches = []
    for r in rows:
        if r['mode'] == 'A':
            continue
        key = r['n_ctx'],r['prompt_profile'],r['max_tokens'],r['repeat']
        expected = baselines.get(key)
        if expected is not None and r['output_sha256'] != expected:
            mismatches.append({**dict(zip(['context','prompt','max_tokens','repeat'],key)),
                               'case':r['case'], 'expected':expected, 'actual':r['output_sha256']})
    return mismatches


def save_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--server', type=Path, required=True, help='path to CUDA llama-server binary')
    p.add_argument('--target', type=Path, required=True, help='Qwen3.8 target GGUF')
    p.add_argument('--mtp', type=Path, required=True, help='matching native mtp-*.gguf')
    p.add_argument('--primary-device', default='CUDA0', help='exact --list-devices name for RTX 5080')
    p.add_argument('--draft-device', default='CUDA1', help='exact --list-devices name for RTX 3070 Ti')
    p.add_argument('--target-ngl', type=int, default=35, help='fixed target layer GPU budget; same all modes')
    p.add_argument('--expect-primary', default='5080', help='require target --list-devices entry contains this text')
    p.add_argument('--expect-draft', default='3070 Ti', help='require secondary --list-devices entry contains this text')
    p.add_argument('--contexts', type=lambda v:numlist(v,256), default=[4096], help='e.g. 4096,16384')
    p.add_argument('--depths', type=lambda v:numlist(v,1), default=[1,3], help='MTP draft depths, e.g. 1,2,3,4')
    p.add_argument('--generations', type=lambda v:numlist(v,1), default=[128], help='output token budgets, e.g. 128,512')
    p.add_argument('--prompt-profiles', default=['short'], type=lambda v:[x.strip() for x in v.split(',')],
                   help='short,long')
    p.add_argument('--modes', default=['A','B','C'], type=lambda v:[x.strip().upper() for x in v.split(',')],
                   help='A,B,C')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--ubatch-size', type=int, default=512)
    p.add_argument('--flash-attn', choices=['on','off','auto'], default='on')
    p.add_argument('--no-mmap', action='store_true')
    p.add_argument('--load-timeout', type=int, default=1200)
    p.add_argument('--request-timeout', type=int, default=1200)
    p.add_argument('--gpu-sample-seconds', type=float, default=0.5)
    p.add_argument('--nvidia-smi', default=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi')
    p.add_argument('--port', type=int, default=0, help='0 = choose an available port')
    p.add_argument('--output-dir', type=Path, default=Path('phase1_results'))
    p.add_argument('--llama-repo', type=Path, help='optional source tree for revision recording')
    p.add_argument('--extra-server-arg', action='append', default=[], help='additional single server argument; repeat per argument')
    p.add_argument('--allow-divergence', action='store_true', help='record but do not fail on A/B/C output mismatch')
    p.add_argument('--nsys', action='store_true', help='profile each whole server run with nsys (large traces)')
    p.add_argument('--trace', action='store_true', help='enable LLAMA_TRACE=1 (do separately from timed runs)')
    p.add_argument('--dry-run', action='store_true', help='print exact server launch commands without needing models')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if any(v not in ('A','B','C') for v in a.modes) or any(v not in ('short','long') for v in a.prompt_profiles):
        sys.exit('invalid --modes or --prompt-profiles')
    if a.primary_device == a.draft_device and 'C' in a.modes:
        sys.exit('ERROR: primary and dedicated draft device are identical; refusing split-GPU benchmark')
    if a.target_ngl < 0 or a.repeats < 1 or a.gpu_sample_seconds <= 0:
        sys.exit('invalid --target-ngl, --repeats, or GPU sampling interval')
    runs = list(cases(a.contexts,a.depths,a.modes))
    if a.dry_run:
        for case in runs:
            print(case.name(), json.dumps(build_command(a, case.mode, case.n_ctx, case.depth, 8090)))
        return 0
    if not a.server.is_file():
        sys.exit(f'missing llama-server binary: {a.server}')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    meta = ensure_models(a)
    meta['available_devices'] = check_devices(a.server, a.primary_device, a.draft_device,
                                               a.expect_primary, a.expect_draft)
    try:
        server_version = subprocess.run([str(a.server),'--version'], capture_output=True,
                                        text=True, timeout=15)
        meta['server_version'] = (server_version.stdout + server_version.stderr).strip()[:2000]
    except Exception as e:
        meta['server_version_error'] = str(e)
    if a.llama_repo:
        try:
            meta['commit'] = subprocess.check_output(['git','-C',str(a.llama_repo),'rev-parse','HEAD'],
                text=True, timeout=10).strip()
        except Exception as e:
            meta['commit_error'] = str(e)
    meta['benchmark_args'] = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    meta['created_utc'] = datetime.now(timezone.utc).isoformat()
    (a.output_dir / 'manifest.json').write_text(json.dumps(meta,indent=2))
    all_rows = []
    failures = []
    for case in runs:
        try:
            rows = run_server(a, case, a.output_dir, meta)
            all_rows += rows
        except Exception as e:
            failures.append({'case':case.name(),'error':str(e)})
            print('FAIL',case.name(),str(e),file=sys.stderr,flush=True)
        finally:
            save_csv(a.output_dir/'trials.csv',all_rows)
            (a.output_dir/'failures.json').write_text(json.dumps(failures,indent=2))
    summaries = summarize(all_rows)
    save_csv(a.output_dir/'summary.csv',summaries)
    (a.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
    mismatches = compare_to_baseline(all_rows)
    (a.output_dir/'mismatches.json').write_text(json.dumps(mismatches,indent=2))
    print('\nSUMMARY (measured):')
    print('mode ctx depth prompt max_out median_output_tok/s accept speedup_vs_A')
    for x in summaries:
        fmt = lambda n: 'n/a' if n is None else f'{n:.3f}'
        print(x['mode'],x['context'],x['depth'],x['prompt'],x['max_tokens'],
              fmt(x['median_accepted_output_tokens_per_sec']),fmt(x['acceptance_rate']),fmt(x['speedup_vs_A']))
    print(f'files: {a.output_dir.resolve()}')
    print(f'failures: {len(failures)}, output mismatches: {len(mismatches)}')
    if failures or (mismatches and not a.allow_divergence):
        return 2
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as e:
        sys.exit(f'ERROR: {e}')
