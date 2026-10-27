import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import psutil

from common import DATA


CODE = Path(__file__).resolve().parent
THREAD_ENV = {
    'OMP_NUM_THREADS': '4',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'CUDA_VISIBLE_DEVICES': '',
}
ASSET_LIMIT = 16 * 1024 * 1024


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_assets(checkpoint, config):
    files = sorted(path for path in checkpoint.rglob('*') if path.is_file())
    if not files:
        raise ValueError('The checkpoint contains no inference assets')
    assets = {
        path.relative_to(checkpoint).as_posix(): {
            'bytes': path.stat().st_size,
            'sha256': file_hash(path),
        }
        for path in files
    }
    total = sum(value['bytes'] for value in assets.values())
    if total > ASSET_LIMIT:
        raise ValueError(f'Inference assets exceed 16 MiB: {total} bytes')
    declared = config.get('asset_sha256', {})
    if not isinstance(declared, dict):
        raise ValueError('config.asset_sha256 must map relative filenames to SHA256 values')
    for name, expected in declared.items():
        path = (checkpoint / name).resolve()
        try:
            relative = path.relative_to(checkpoint).as_posix()
        except ValueError as error:
            raise ValueError(f'Asset hash path is outside the checkpoint: {name}') from error
        if relative not in assets or assets[relative]['sha256'] != expected:
            raise ValueError(f'Frozen inference asset hash mismatch: {name}')
    fingerprint = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    return assets, total, fingerprint


def inspect_official_files():
    manifest_path = CODE / 'PACKAGE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    results = {}
    for name, expected in manifest['files'].items():
        if name == 'code/submission.py':
            continue
        path = CODE.parent / name
        actual = file_hash(path) if path.is_file() else None
        results[name] = {
            'expected_sha256': expected,
            'actual_sha256': actual,
            'matches_release': actual == expected,
        }
    fixed_tools = (
        'code/common.py',
        'code/check_data.py',
        'code/predict.py',
        'code/score_predictions.py',
    )
    if any(not results.get(name, {}).get('matches_release', False) for name in fixed_tools):
        raise ValueError('An official data, prediction, or scoring tool differs from PACKAGE_MANIFEST.json')
    return {
        'manifest_sha256': file_hash(manifest_path),
        'all_match_excluding_submission': all(row['matches_release'] for row in results.values()),
        'files': results,
    }


def inspect_hardware():
    memory = psutil.virtual_memory()
    hardware = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'physical_cpu_cores': psutil.cpu_count(logical=False),
        'logical_cpu_cores': psutil.cpu_count(logical=True),
        'ram_total_bytes': memory.total,
        'ram_available_bytes_before_evaluation': memory.available,
        'gpu': None,
    }
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            result = subprocess.run(
                [executable, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            hardware['gpu'] = {
                'query': 'nvidia-smi name,driver_version,memory.total',
                'returncode': result.returncode,
                'output': result.stdout.strip(),
                'stderr': result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            hardware['gpu'] = {'query_error': str(error)}
    return hardware


def inspect_dependencies():
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'threadpoolctl', 'joblib', 'psutil', 'pip'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def sample_memory(process):
    current = 0
    observed_peak = 0
    try:
        members = [process, *process.children(recursive=True)]
    except psutil.Error:
        members = [process]
    for member in members:
        try:
            info = member.memory_info()
            current += info.rss
            observed_peak += max(info.rss, getattr(info, 'peak_wset', info.rss))
        except psutil.Error:
            continue
    return current, observed_peak


def run_process(command, environment):
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=CODE,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    try:
        process = psutil.Process(child.pid)
    except psutil.Error:
        process = None
    peak_rss = 0
    peak_working_set = 0
    try:
        while True:
            if process is not None:
                rss, working_set = sample_memory(process)
                peak_rss = max(peak_rss, rss)
                peak_working_set = max(peak_working_set, working_set)
            try:
                stdout, stderr = child.communicate(timeout=0.01)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        child.kill()
        child.communicate()
        raise
    elapsed = time.perf_counter() - started
    return {
        'command': command,
        'pid': child.pid,
        'returncode': child.returncode,
        'wall_seconds': elapsed,
        'peak_sampled_process_tree_rss_bytes': peak_rss,
        'peak_observed_working_set_bytes': peak_working_set,
        'stdout': stdout,
        'stderr': stderr,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output-dir', type=Path, default=CODE / 'runs' / 'evaluation')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        parser.error('--checkpoint must be an inference asset directory containing config.json')
    config_path = checkpoint / 'config.json'
    if not config_path.is_file():
        parser.error('The checkpoint must contain config.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if args.split == 'test' and config.get('frozen') is not True:
        parser.error('Test evaluation requires config.json with frozen set to true')
    assets, asset_bytes, fingerprint = inspect_assets(checkpoint, config)
    output_dir = args.output_dir.resolve()
    predictions_path = output_dir / f'{args.split}-predictions.json'
    score_path = output_dir / f'{args.split}-score.json'
    resource_path = output_dir / f'{args.split}-resources.json'
    if args.split == 'test' and resource_path.exists():
        previous = json.loads(resource_path.read_text(encoding='utf-8'))
        same = previous.get('checkpoint_fingerprint') == fingerprint
        if not same:
            parser.error('Test evaluation already exists for a different checkpoint; --rerun permits identical frozen-checkpoint reproduction only')
        if not args.rerun:
            parser.error('Test evaluation already exists; use --rerun only for explicit frozen-checkpoint reproduction')
    official_files = inspect_official_files()
    source_files = {
        path.name: file_hash(path)
        for path in sorted(CODE.glob('*.py'))
    }
    for name, expected in config.get('source_sha256', {}).items():
        if source_files.get(name) != expected:
            raise ValueError('Frozen source code hash mismatch: ' + name)
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['PYTHONUTF8'] = '1'
    prediction_command = [
        sys.executable,
        str(CODE / 'predict.py'),
        '--method', str(CODE / 'submission.py'),
        '--checkpoint', str(checkpoint),
        '--split', args.split,
        '--data-dir', str(args.data_dir.resolve()),
        '--output', str(predictions_path),
    ]
    scoring_command = [
        sys.executable,
        str(CODE / 'score_predictions.py'),
        '--predictions', str(predictions_path),
        '--split', args.split,
        '--data-dir', str(args.data_dir.resolve()),
        '--output', str(score_path),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (predictions_path, score_path):
        if path.exists():
            path.unlink()
    record = {
        'split': args.split,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint),
        'checkpoint_fingerprint': fingerprint,
        'frozen': config.get('frozen') is True,
        'assets': assets,
        'asset_bytes': asset_bytes,
        'asset_limit_bytes': ASSET_LIMIT,
        'asset_limit_pass': asset_bytes <= ASSET_LIMIT,
        'python_executable': sys.executable,
        'python_version': sys.version,
        'dependencies': inspect_dependencies(),
        'hardware': inspect_hardware(),
        'environment': {name: environment[name] for name in THREAD_ENV},
        'python_environment': {
            name: environment[name] for name in ('PYTHONIOENCODING', 'PYTHONUTF8')
        },
        'source_sha256': source_files,
        'official_release_integrity': official_files,
        'prediction_command': prediction_command,
        'scoring_command': scoring_command,
        'outputs': {
            'predictions': str(predictions_path),
            'score': str(score_path),
            'resources': str(resource_path),
        },
        'measurement': {
            'clock': 'time.perf_counter',
            'scope': 'Fresh prediction process followed by fresh official scoring process, including startup, imports, loading and parent monitoring overhead',
            'memory': 'Process tree RSS sampled at communicate timeouts of at most 10 ms; Windows peak working set when exposed by psutil',
            'inference_device': 'CPU',
            'reference_cpu': 'Xeon Platinum 8457C',
            'reference_cpu_verified': False,
        },
        'completed': False,
        'processes': [],
    }
    write_json(resource_path, record)
    started = time.perf_counter()
    try:
        result = run_process(prediction_command, environment)
        record['processes'].append(result)
        if result['returncode'] == 0:
            result = run_process(scoring_command, environment)
            record['processes'].append(result)
        record['completed'] = len(record['processes']) == 2 and result['returncode'] == 0
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        elapsed = time.perf_counter() - started
        record['complete_cpu_evaluation_seconds'] = elapsed
        record['local_30_second_limit_pass'] = record['completed'] and elapsed <= 30.0
        record['peak_sampled_child_rss_bytes'] = max(
            (value['peak_sampled_process_tree_rss_bytes'] for value in record['processes']),
            default=0,
        )
        record['peak_observed_child_working_set_bytes'] = max(
            (value['peak_observed_working_set_bytes'] for value in record['processes']),
            default=0,
        )
        if record['completed']:
            record['official_score'] = json.loads(score_path.read_text(encoding='utf-8'))
        write_json(resource_path, record)
    for process in record['processes']:
        if process['stdout']:
            print(process['stdout'].rstrip())
        if process['stderr']:
            print(process['stderr'].rstrip(), file=sys.stderr)
    print(json.dumps({
        'completed': record['completed'],
        'resources': str(resource_path),
        'complete_cpu_evaluation_seconds': elapsed,
        'local_30_second_limit_pass': record['local_30_second_limit_pass'],
        'asset_bytes': asset_bytes,
        'reference_cpu_verified': False,
    }))
    if not record['completed']:
        raise SystemExit(result['returncode'] or 1)


if __name__ == '__main__':
    main()
