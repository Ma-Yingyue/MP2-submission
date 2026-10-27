from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CODE = Path(__file__).resolve().parent
PLOT_SOURCE = CODE / 'runs' / 'recheck-20261025' / 'figures' / 'plot_metrics.py'


def run_stage(number, name, script, arguments, environment):
    print(f'[{number}/5] {name}', flush=True)
    subprocess.run(
        [sys.executable, str(script), *map(str, arguments)],
        cwd=CODE,
        env=environment,
        check=True,
    )


def main():
    required_files = [CODE / name for name in ('check_data.py', 'train_selector.py', 'evaluate_selector.py')]
    required_files.append(PLOT_SOURCE)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f'Required script is missing: {path}')
    modules = ('numpy', 'scipy', 'sklearn', 'lightgbm', 'threadpoolctl', 'joblib', 'psutil', 'matplotlib')
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError('Required Python modules are missing: ' + ', '.join(missing))

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    run_root = CODE / 'runs' / f'full-{timestamp}'
    run_root.mkdir(parents=True, exist_ok=False)
    training_dir = run_root / 'final'
    evaluation_dir = run_root / 'evaluation'
    figure_dir = evaluation_dir / 'figures'
    checkpoint = training_dir / 'model_bundle'
    environment = os.environ.copy()
    environment.update({
        'OMP_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'BLIS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'CUDA_VISIBLE_DEVICES': '',
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUTF8': '1',
    })
    print(f'Output directory: {run_root}', flush=True)

    run_stage(1, 'Check data', CODE / 'check_data.py', [], environment)
    run_stage(2, 'Train and freeze model', CODE / 'train_selector.py',
              ['--output-dir', training_dir, '--seed', 20261023], environment)
    for number, split in ((3, 'validation'), (4, 'test')):
        run_stage(number, f'Evaluate {split}', CODE / 'evaluate_selector.py',
                  ['--checkpoint', checkpoint, '--split', split, '--output-dir', evaluation_dir], environment)
    figure_dir.mkdir(parents=True, exist_ok=False)
    plot_script = figure_dir / 'plot_metrics.py'
    shutil.copy2(PLOT_SOURCE, plot_script)
    run_stage(5, 'Generate figures', plot_script, [], environment)

    scores = {
        split: json.loads((evaluation_dir / f'{split}-score.json').read_text(encoding='utf-8'))
        for split in ('validation', 'test')
    }
    print(json.dumps({
        'completed': True,
        'output_directory': str(run_root),
        'checkpoint': str(checkpoint),
        'evaluation_directory': str(evaluation_dir),
        'figure_directory': str(figure_dir),
        'figure_count': len(list(figure_dir.glob('*.png'))),
        'scores': scores,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'Pipeline stopped because a stage failed with exit code {error.returncode}.', file=sys.stderr)
        raise SystemExit(error.returncode)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
