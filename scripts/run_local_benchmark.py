"""Own loopback services for a bounded benchmark of existing trained S1 weights.

Click smoke starts only the native decision model; it does not load a text model.
Travel browser modes and the current all-mode child contract require the text helper.
"""

import argparse
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise ValueError('Owned loopback readiness probes must not follow redirects')


def needs_text_service(mode, fixture):
    return mode == 'all' or (mode == 'smoke' and fixture == 'travel')


def benchmark_command(args, *, mode, destination, metadata_path, warmup, count, fixture_port):
    command = [sys.executable, str(ROOT / 'scripts/local_benchmark.py'), '--mode', mode,
               '--policy', args.policy, '--output', str(destination), '--model-metadata', str(metadata_path),
               '--warmup', str(warmup), '--decision-repeats', str(count),
               '--repeats', str(args.repeats), '--task-timeout', str(args.task_timeout),
               '--fixture', args.fixture, '--fixture-url', f'http://127.0.0.1:{fixture_port}']
    if args.chrome_path is not None:
        command.extend(['--chrome-path', str(args.chrome_path)])
    if mode == 'helper-only':
        command.extend(['--text-repeats', '1'])
    return command


def hardware_inventory():
    inventory = {'os': platform.system(), 'architecture': platform.machine()}
    if sys.platform == 'darwin':
        try:
            data = json.loads(subprocess.check_output(
                ['system_profiler', 'SPHardwareDataType', '-json'], text=True,
                stderr=subprocess.DEVNULL, timeout=10))['SPHardwareDataType'][0]
            for key in ('machine_name', 'chip_type', 'physical_memory', 'number_processors'):
                inventory[key] = data.get(key)
            inventory['source'] = 'system_profiler whitelisted non-identifying fields'
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            inventory['hardware_details'] = 'unavailable; no hardware values inferred'
    return inventory


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def ready(url, process, deadline):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Owned process exited {process.returncode}; inspect raw logs')
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return json.load(response)
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f'Owned loopback service not ready: {url}')


def ready_fixture(url, expected, process, deadline):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Owned fixture service exited; no browser will launch')
        try:
            with opener.open(url, timeout=1) as response:
                if response.read() != expected:
                    raise ValueError('Owned fixture content does not match this checkout')
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError('Owned fixture did not become ready within its startup budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s1-repo', type=Path, required=True,
                        help='Existing trained System One repository; no model download occurs')
    parser.add_argument('--quantization', choices=['Q4_K_M', 'Q8_0'], default='Q4_K_M')
    parser.add_argument('--mode', choices=['smoke', 'all', 'decision-only'], default='smoke')
    parser.add_argument('--policy', choices=['baseline', 'noul_v1'], default='baseline')
    parser.add_argument('--fixture', choices=['travel', 'click'], default='travel')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chrome-path', type=Path,
                        help='Existing headless Chromium executable; required for smoke/all modes')
    parser.add_argument('--decision-repeats', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--task-timeout', type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.decision_repeats <= 10 or not 1 <= args.repeats <= 5:
        parser.error('bounded repeats: decision 1..10 and browser 1..5')
    if not 30 <= args.task_timeout <= 180:
        parser.error('task timeout must be 30..180 seconds')
    if not 0 <= args.warmup <= 2:
        parser.error('warmup must be 0..2')
    if args.mode != 'decision-only' and (args.chrome_path is None or not args.chrome_path.is_file()):
        parser.error('--chrome-path must identify an existing executable for smoke/all modes')
    s1 = args.s1_repo.expanduser().resolve()
    package = s1 / 'artifacts/exports' / ('s1-boolq-gguf' if args.quantization == 'Q8_0'
                                         else 's1-boolq-Q4_K_M')
    manifest = json.loads((package / 'conversion-manifest.json').read_text())
    if manifest.get('source_hub_repository') != 'knowlet/Gemma-4-12B-Unified-System-One':
        raise ValueError('Expected the user-trained System One provenance')
    model = package / manifest['model_file']
    if not model.is_file():
        raise FileNotFoundError(model)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = ROOT / '.local-cache'
    (cache / 'tmp').mkdir(parents=True, exist_ok=True)
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL') if key in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               DO_NOT_TRACK='1', HF_HOME=str(cache / 'huggingface'),
               XDG_CACHE_HOME=str(cache), TMPDIR=str(cache / 'tmp'))
    ports, reservations = [], []
    for _ in range(3):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        reservations.append(sock)
        ports.append(sock.getsockname()[1])
    for sock in reservations:
        sock.close()
    decision_port, text_port, fixture_port = ports
    processes, handles = [], []
    metadata = {'artifact_path': str(model), 'version': manifest['source_hub_weights_revision'],
                'policy': args.policy, 'text_helper_required': needs_text_service(args.mode, args.fixture),
                'quantization': args.quantization, 'runtime': 'llama.cpp Metal',
                's1_repo_head': revision(s1), 'jev_repo_head': revision(ROOT),
                'llama_cpp_head': revision(s1 / 'artifacts/export-tools/llama.cpp'),
                'manifest': manifest, 'calibration': json.loads((package / 's1_config.json').read_text()),
                'hardware': hardware_inventory(),
                'settings': {'context': 4096, 'batch': 512, 'threads': 4, 'gpu_layers': 99,
                             'seed': 17, 'text_temperature': 0, 'text_max_tokens': 128,
                             'text_prompt_cache': False},
                'commands': [], 'startup': {}, 'resource_usage': {}}
    metadata_path = output / 'model-metadata.json'

    def save():
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    def start(name, command):
        handle = (output / f'{name}.log').open('w')
        handles.append(handle)
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=handle)
        processes.append((name, process))
        metadata['commands'].append({'name': name, 'argv': command, 'pid': process.pid})
        save()
        return process, started

    def run_bench(mode, destination, warmup, count):
        snapshot = output / f'model-metadata-{destination.name}.json'
        snapshot.write_text(json.dumps(metadata, indent=2) + '\n')
        command = benchmark_command(args, mode=mode, destination=destination, metadata_path=snapshot,
                                    warmup=warmup, count=count, fixture_port=fixture_port)
        stdout_path = output / f'benchmark-{destination.name}.log'
        print(f'Running {mode}: {destination}', flush=True)
        with stdout_path.open('w') as stdout:
            result = subprocess.run(command, cwd=ROOT, env=env, timeout=1200, check=False,
                                    stdout=stdout, stderr=subprocess.STDOUT)
        print(f'Completed {mode}, exit {result.returncode}; raw stdout: {stdout_path}', flush=True)
        metadata['commands'].append({'name': 'benchmark', 'argv': command, 'exit_code': result.returncode})
        save()
        return result.returncode

    def stop_requested(_signum, _frame):
        raise KeyboardInterrupt('Stopping only processes owned by this benchmark')

    signal.signal(signal.SIGTERM, stop_requested)
    try:
        service, began = start('decision-service', [sys.executable, str(ROOT / 'scripts/local_s1_service.py'),
                              '--s1-repo', str(s1), '--model', str(package), '--port', str(decision_port),
                              '--native-runner', str(s1 / 'artifacts/gguf-runtime/bin/s1-gguf'),
                              '--cache-dir', str(cache / 's1'),
                              '--ctx-size', '4096', '--batch-size', '512', '--threads', '4',
                              '--raw-log', str(output / 'native-decision-time.log'),
                              '--request-log', str(output / 'native-requests.jsonl')])
        health = ready(f'http://127.0.0.1:{decision_port}/healthz', service, began + 30)
        if health.get('s1_repo') != str(s1) or health.get('model_package') != str(package):
            raise ValueError('Loopback decision service identity does not match this owned launch')
        metadata['decision_service_identity'] = health
        metadata['startup']['decision_http_bind_ms'] = (time.monotonic() - began) * 1000
        metadata['vm_stat_before'] = subprocess.check_output(['vm_stat'], text=True)
        env['S1_DECISION_URL'] = f'http://127.0.0.1:{decision_port}/decide'
        save()
        if run_bench('decision-only', output / 'cold', 0, 1):
            raise RuntimeError('Real decision smoke failed; no broader benchmark started')
        metadata['startup']['decision_service_to_first_complete_ms'] = (time.monotonic() - began) * 1000
        if args.mode == 'decision-only':
            return run_bench('decision-only', output / 'warm', args.warmup, args.decision_repeats)
        fixture, began = start('fixture', [sys.executable, '-m', 'http.server', str(fixture_port), '--bind',
                                         '127.0.0.1', '--directory', str(ROOT / 'jev_ultrafast/static')])
        if needs_text_service(args.mode, args.fixture):
            text, began = start('text-service', [str(s1 / 'artifacts/export-tools/llama.cpp/build/bin/llama-server'),
                           '--model', str(model), '--host', '127.0.0.1', '--port', str(text_port),
                           '--alias', 'system-one-trained-' + args.quantization, '--ctx-size', '4096',
                           '--batch-size', '512', '--ubatch-size', '512', '--threads', '4',
                           '--gpu-layers', '99', '--parallel', '1', '--seed', '17', '--temp', '0',
                           '--reasoning', 'off', '--no-cache-prompt', '--no-webui', '--offline'])
            ready(f'http://127.0.0.1:{text_port}/health', text, began + 90)
            models = ready(f'http://127.0.0.1:{text_port}/v1/models', text, began + 90)
            if 'system-one-trained-' + args.quantization not in {item['id'] for item in models.get('data', [])}:
                raise ValueError('Owned text runtime did not expose the expected trained model alias')
            metadata['text_runtime_models'] = models
            metadata['startup']['text_model_ready_ms'] = (time.monotonic() - began) * 1000
            env.update(S1_TEXT_BASE_URL=f'http://127.0.0.1:{text_port}/v1',
                       TEXT_MODEL='system-one-trained-' + args.quantization)
            save()
        fixture_name = 'fixture.html' if args.fixture == 'travel' else 'local_click_fixture.html'
        ready_fixture(f'http://127.0.0.1:{fixture_port}/{fixture_name}',
                      (ROOT / 'jev_ultrafast/static' / fixture_name).read_bytes(), fixture, time.monotonic() + 5)
        if needs_text_service(args.mode, args.fixture) and run_bench('helper-only', output / 'helper-probe', 0, 1):
            raise RuntimeError('Real text-helper smoke failed; no browser benchmark started')
        return run_bench(args.mode, output / 'warm', args.warmup, args.decision_repeats)
    finally:
        for name, process in reversed(processes):
            if process.returncode is None:
                process.terminate()
                deadline = time.monotonic() + 40
                while True:
                    try:
                        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    except ChildProcessError:
                        metadata['resource_usage'][name] = {
                            'exit_code': process.returncode, 'peak_rss_bytes': None,
                            'source': 'already reaped; no separate RSS claim'}
                        break
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                        metadata['resource_usage'][name] = {
                            'peak_rss_bytes': usage.ru_maxrss, 'user_seconds': usage.ru_utime,
                            'system_seconds': usage.ru_stime, 'exit_code': process.returncode,
                            'source': 'macOS wait4 owned PID; descendant aggregation may apply; peaks are not additive'}
                        break
                    if time.monotonic() >= deadline:
                        process.kill()
                    time.sleep(0.1)
            else:
                metadata['resource_usage'][name] = {
                    'exit_code': process.returncode, 'peak_rss_bytes': None,
                    'source': 'already reaped during startup; no separate RSS claim'}
        metadata['resource_usage']['launcher_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata['vm_stat_after'] = subprocess.check_output(['vm_stat'], text=True)
        save()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
