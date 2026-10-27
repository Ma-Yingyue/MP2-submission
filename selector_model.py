import json
from pathlib import Path

import numpy as np

from common import parse_answer


FORMAT = 'mp2-selector-v1'


def parsed_answers(bank):
    return [[parse_answer(candidate['text']) for candidate in row['candidates']] for row in bank]


def hard_groups(bank):
    return np.asarray([len(set(answers) - {None}) >= 2 for answers in parsed_answers(bank)])


def input_matrix(arrays, features, names, mode):
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float32)
    if mode == 'embedding':
        return embeddings
    if mode == 'local':
        columns = [i for i, name in enumerate(names) if name.startswith('local_')]
        return np.concatenate((embeddings, features[:, :, columns]), axis=2)
    if mode == 'full':
        return np.concatenate((embeddings, features), axis=2)
    if mode == 'text':
        return features
    raise ValueError('Unknown feature mode: ' + str(mode))


def comparable_scores(scores):
    centered = np.asarray(scores, dtype=np.float64)
    centered = centered - centered.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.sqrt(np.mean(centered * centered, axis=1, keepdims=True)), 1e-8)


def choose(bank, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores of shape [N, 4]')
    if not bank:
        return []
    valid = np.asarray([[answer is not None for answer in row] for row in parsed_answers(bank)], dtype=bool)
    usable = valid.any(axis=1)
    if not usable.all():
        raise ValueError('A question has no candidate answer accepted by the official parser')
    scores = scores.copy()
    scores[usable[:, None] & ~valid] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]


def metrics(bank, labels, scores):
    choices = choose(bank, scores)
    if not choices:
        raise ValueError('Cannot evaluate an empty bank')
    correct = labels[np.arange(len(bank)), choices]
    hard = hard_groups(bank)
    return {
        'correct': int(correct.sum()),
        'n': len(bank),
        'accuracy': float(correct.mean() * 100),
        'hard_correct': int(correct[hard].sum()),
        'hard_n': int(hard.sum()),
        'hard_accuracy': float(correct[hard].mean() * 100) if hard.any() else None,
    }


def component_scores(component, bank, arrays, features, names, root):
    matrix = input_matrix(arrays, features, names, component['mode'])
    if component['kind'] == 'linear':
        with np.load(Path(root) / component['file'], allow_pickle=False) as weights:
            coefficient = weights['coefficient']
        if matrix.shape[2] != len(coefficient):
            raise ValueError('Checkpoint feature dimension differs from inputs')
        return (matrix.reshape(-1, matrix.shape[2]) @ coefficient).reshape(len(bank), 4)
    if component['kind'] == 'lightgbm':
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(Path(root) / component['file']))
        return model.predict(matrix.reshape(-1, matrix.shape[2]), raw_score=True, num_threads=4).reshape(len(bank), 4)
    raise ValueError('Unknown checkpoint component')


def predict_scores(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('A trained checkpoint directory is required')
    root = Path(checkpoint)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if config.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format')
    if not bank:
        return np.empty((0, 4), dtype=np.float64)
    embeddings = np.asarray(arrays['embeddings'])
    if embeddings.shape != (len(bank), 4, 1024) or not np.isfinite(embeddings).all():
        raise ValueError('Expected finite embeddings of shape [N, 4, 1024]')
    if any(len(row['candidates']) != 4 for row in bank):
        raise ValueError('Expected four candidates per question')
    if any(component['mode'] != 'embedding' for component in config['components']):
        from selector_features import build_features

        features, names = build_features(bank, arrays)
        if names != config['feature_names']:
            raise ValueError('Checkpoint feature schema differs from code')
    else:
        features, names = None, []
    scores = np.zeros((len(bank), 4), dtype=np.float64)
    for component in config['components']:
        component_result = component_scores(component, bank, arrays, features, names, root)
        if config.get('normalize_components', False):
            component_result = comparable_scores(component_result)
        scores += component['weight'] * component_result
    return scores
