import os

for variable, value in {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1'}.items():
    os.environ[variable] = value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from selector_features import build_features, majority_choices
from selector_model import FORMAT, comparable_scores, hard_groups, input_matrix, metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def supervision(bank, split, data_dir):
    if split not in ('train', 'validation'):
        raise ValueError('Training can only open train and validation references')
    references = read_jsonl(Path(data_dir) / f'gsm8k_{split}.jsonl')
    if len(references) != len(bank):
        raise ValueError('Reference length mismatch')
    labels = []
    for row, reference in zip(bank, references):
        if row['id'] != reference['id'] or row['question'] != reference['question']:
            raise ValueError('Reference alignment mismatch')
        gold = parse_answer(reference['answer'])
        label = [gold is not None and parse_answer(candidate['text']) == gold for candidate in row['candidates']]
        if not any(label):
            raise ValueError('Question has no correct candidate under the official parser')
        labels.append(label)
    return np.asarray(labels, dtype=np.int64)


def source_hashes():
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if (root / 'model_bundle' / 'config.json').exists():
        raise FileExistsError('A frozen model already exists; use a new output directory for a new training run')
    root.mkdir(parents=True, exist_ok=True)
    candidates_dir = root / 'candidates'
    candidates_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    np.random.seed(args.seed)
    for split in ('train', 'validation'):
        check_data(args.data_dir, (split,))
    train_bank, train_arrays = load_inputs('train', args.data_dir)
    validation_bank, validation_arrays = load_inputs('validation', args.data_dir)
    train_labels = supervision(train_bank, 'train', args.data_dir)
    validation_labels = supervision(validation_bank, 'validation', args.data_dir)
    train_hard = hard_groups(train_bank)
    feature_started = time.perf_counter()
    train_features, names = build_features(train_bank, train_arrays)
    validation_features, validation_names = build_features(validation_bank, validation_arrays)
    if names != validation_names:
        raise ValueError('Feature schema mismatch')
    feature_seconds = time.perf_counter() - feature_started
    majority = np.zeros((len(validation_bank), 4), dtype=np.float32)
    majority[np.arange(len(validation_bank)), majority_choices(validation_bank)] = 1
    experiments = [{'name': 'majority_valid_answer', 'validation': metrics(validation_bank, validation_labels, majority), 'training_seconds': 0.0}]
    results = []
    matrices = {}

    def matrices_for(mode):
        if mode not in matrices:
            matrices[mode] = (
                input_matrix(train_arrays, train_features, names, mode),
                input_matrix(validation_arrays, validation_features, names, mode),
            )
        return matrices[mode]

    def add_result(name, component, scores, train_seconds, parameters):
        result = {'name': name, 'components': [component], 'validation': metrics(validation_bank, validation_labels, scores), 'training_seconds': train_seconds, 'parameters': parameters}
        experiments.append(result)
        results.append((result, scores))
        print(json.dumps({'experiment': name, **result['validation'], 'training_seconds': train_seconds}), flush=True)

    def train_linear(name, mode, c, hard_only, centered):
        matrix, val_matrix = matrices_for(mode)
        mask = train_hard if hard_only else np.ones(len(train_bank), dtype=bool)
        fit_matrix = matrix[mask].copy()
        scale = np.ones(fit_matrix.shape[2], dtype=np.float32)
        if mode != 'embedding':
            start = 0 if mode == 'text' else 1024
            scale[start:] = np.maximum(fit_matrix[:, :, start:].reshape(-1, fit_matrix.shape[2] - start).std(axis=0), 0.1)
        fit_matrix /= scale
        if centered:
            fit_matrix -= fit_matrix.mean(axis=1, keepdims=True)
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, tol=1e-4, random_state=args.seed)
        tick = time.perf_counter()
        with threadpool_limits(limits=1, user_api='blas'):
            model.fit(fit_matrix.reshape(-1, fit_matrix.shape[2]), train_labels[mask].reshape(-1))
        elapsed = time.perf_counter() - tick
        if np.any(model.n_iter_ >= model.max_iter):
            raise RuntimeError('Linear optimizer did not converge')
        coefficient = model.coef_[0] / scale
        file = name + '.npz'
        np.savez(candidates_dir / file, coefficient=coefficient)
        scores = (val_matrix.reshape(-1, val_matrix.shape[2]) @ coefficient).reshape(-1, 4)
        add_result(name, {'kind': 'linear', 'mode': mode, 'file': file, 'weight': 1.0}, scores, elapsed, {
            'C': c, 'hard_training_only': hard_only, 'center_training_groups': centered, 'training_questions': int(mask.sum()),
            'seed': args.seed, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-4, 'iterations': int(model.n_iter_[0]),
            'feature_scaling': 'none' if mode == 'embedding' else 'training-only standard deviation for engineered features',
        })

    train_linear('raw_embedding_all', 'embedding', 0.01, False, False)
    train_linear('centered_embedding_all', 'embedding', 0.1, False, True)
    for c in (0.01, 0.03, 0.1):
        train_linear('centered_embedding_hard_c' + str(c), 'embedding', c, True, True)
    for mode in ('local', 'full'):
        for c in (0.01, 0.03, 0.1):
            train_linear(mode + '_hard_c' + str(c), mode, c, True, True)
    train_linear('engineered_only_hard', 'text', 0.1, True, True)
    matrix, val_matrix = matrices_for('text')
    for leaves, minimum in ((7, 80), (15, 80), (15, 160)):
        name = f'engineered_boost_l{leaves}_m{minimum}'
        model = lgb.LGBMClassifier(
            objective='binary', n_estimators=400, learning_rate=0.035, num_leaves=leaves,
            min_child_samples=minimum, reg_lambda=10.0, colsample_bytree=0.9,
            random_state=args.seed, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True,
        )
        tick = time.perf_counter()
        model.fit(matrix[train_hard].reshape(-1, matrix.shape[2]), train_labels[train_hard].reshape(-1))
        elapsed = time.perf_counter() - tick
        file = name + '.txt'
        model.booster_.save_model(str(candidates_dir / file))
        scores = model.booster_.predict(val_matrix.reshape(-1, val_matrix.shape[2]), raw_score=True, num_threads=4).reshape(-1, 4)
        add_result(name, {'kind': 'lightgbm', 'mode': 'text', 'file': file, 'weight': 1.0}, scores, elapsed, model.get_params())
    ranker = lgb.LGBMRanker(
        objective='lambdarank', n_estimators=400, learning_rate=0.04, num_leaves=15,
        min_child_samples=50, reg_lambda=5.0, random_state=args.seed,
        n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True,
    )
    tick = time.perf_counter()
    ranker.fit(matrix.reshape(-1, matrix.shape[2]), train_labels.reshape(-1), group=[4] * len(train_bank))
    elapsed = time.perf_counter() - tick
    file = 'engineered_ranker.txt'
    ranker.booster_.save_model(str(candidates_dir / file))
    scores = ranker.booster_.predict(val_matrix.reshape(-1, val_matrix.shape[2]), raw_score=True, num_threads=4).reshape(-1, 4)
    add_result('engineered_ranker', {'kind': 'lightgbm', 'mode': 'text', 'file': file, 'weight': 1.0}, scores, elapsed, ranker.get_params())
    linear_results = [(result, scores) for result, scores in results if result['components'][0]['kind'] == 'linear']
    tree_results = [(result, scores) for result, scores in results if result['components'][0]['kind'] == 'lightgbm']
    key = lambda item: (item[0]['validation']['hard_correct'], item[0]['validation']['correct'])
    best_linear, linear_scores = max(linear_results, key=key)
    for tree_result, tree_scores in tree_results:
        for weight in (0.1, 0.2, 0.35, 0.5):
            components = [dict(best_linear['components'][0], weight=1.0 - weight), dict(tree_result['components'][0], weight=weight)]
            scores = (1.0 - weight) * comparable_scores(linear_scores) + weight * comparable_scores(tree_scores)
            result = {
                'name': best_linear['name'] + '_' + tree_result['name'] + '_fusion_' + str(weight), 'components': components,
                'validation': metrics(validation_bank, validation_labels, scores),
                'training_seconds': best_linear['training_seconds'] + tree_result['training_seconds'],
                'parameters': {'tree_weight': weight, 'normalize_components': True},
            }
            experiments.append(result)
            results.append((result, scores))
            print(json.dumps({'experiment': result['name'], **result['validation']}), flush=True)
    best, best_scores = max(results, key=key)
    bundle = root / 'model_bundle'
    bundle.mkdir()
    for component in best['components']:
        shutil.copyfile(candidates_dir / component['file'], bundle / component['file'])
    model_assets = {path.name: digest(path) for path in sorted(bundle.iterdir()) if path.is_file()}
    config = {
        'format': FORMAT, 'protocol': PROTOCOL, 'frozen': True,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
        'selected_experiment': best['name'], 'selection_rule': 'validation hard-group correct count, then full validation correct count; first experiment on a tie',
        'validation': best['validation'], 'components': best['components'],
        'normalize_components': len(best['components']) > 1,
        'feature_names': names if any(component['mode'] != 'embedding' for component in best['components']) else [],
        'asset_sha256': model_assets, 'source_sha256': source_hashes(),
    }
    write_json(bundle / 'config.json', config)
    asset_bytes = sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())
    if asset_bytes > 16 * 2 ** 20:
        raise RuntimeError('Frozen inference assets exceed 16 MiB')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'threadpoolctl', 'joblib', 'pip')}
    record = {
        'protocol': PROTOCOL, 'seed': args.seed, 'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'packages': versions, 'feature_count': len(names), 'feature_names': names,
        'train_questions': len(train_bank), 'train_hard_questions': int(train_hard.sum()),
        'validation_questions': len(validation_bank), 'validation_hard_questions': int(hard_groups(validation_bank).sum()),
        'feature_seconds': feature_seconds, 'total_training_search_seconds': time.perf_counter() - started,
        'experiments': experiments, 'selected': best, 'inference_asset_bytes': asset_bytes,
        'data_used_to_fit': ['train'], 'data_used_to_select': ['validation'], 'test_used_during_development': False,
        'research_question': 'Does centering candidate embeddings within each question and matching training to answer-disagreement groups improve selection?',
        'hypothesis': 'Removing shared question context and training on disagreement groups improves accuracy on ambiguous candidate sets.',
        'local_environment_note': 'User requested the existing Python 3.10.9 and installed packages; Python, pip and all package versions were left unchanged. Assignment README recommends Python 3.12 and numpy 2.5.3.',
        'ai_disclosure': 'Codex assisted with method design, implementation, validation experiments, error analysis and execution. No external pretrained weights or external datasets were used.',
        'reuse': ['Official unchanged bank, numeric parser and evaluation tools', 'NumPy, scikit-learn and LightGBM training implementations'],
        'commands': {
            'data_check': [sys.executable, 'check_data.py'],
            'training': [sys.executable, 'train_selector.py', '--output-dir', str(root), '--seed', str(args.seed)],
            'validation': [sys.executable, 'evaluate_selector.py', '--checkpoint', str(bundle), '--split', 'validation'],
            'test': [sys.executable, 'evaluate_selector.py', '--checkpoint', str(bundle), '--split', 'test'],
        },
        'source_sha256': config['source_sha256'],
    }
    write_json(root / 'experiments.json', record)
    write_json(root / 'environment.json', {'python': sys.version, 'executable': sys.executable, 'packages': versions})
    print(json.dumps({'checkpoint': str(bundle), 'selected': best['name'], 'validation': best['validation'], 'asset_bytes': asset_bytes, 'search_seconds': record['total_training_search_seconds']}), flush=True)


if __name__ == '__main__':
    main()
