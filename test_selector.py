import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from selector_features import build_features
from selector_model import FORMAT, choose, hard_groups
from submission import predict


def sample_bank():
    return [{
        'id': 'new-audit-question',
        'question': 'A box has 3 red balls and 5 blue balls. How many balls are there?',
        'candidates': [
            {'index': 0, 'text': '3 + 5 = 8. Therefore \\boxed{8}.'},
            {'index': 1, 'text': '3 * 5 = 15. Therefore \\boxed{15}.'},
            {'index': 2, 'text': 'The total is \\boxed{8}.'},
            {'index': 3, 'text': 'I cannot provide a final numeric answer.'},
        ],
    }]


class SelectorTests(unittest.TestCase):
    def test_official_parser_mask_and_numeric_equivalence(self):
        bank = sample_bank()
        bank[0]['candidates'][2]['text'] = '#### 16/2'
        self.assertEqual(choose(bank, [[0, 1, 2, 100]]), [2])
        self.assertEqual(hard_groups(bank).tolist(), [True])
        bank[0]['candidates'][1]['text'] = '#### 8.0'
        self.assertEqual(hard_groups(bank).tolist(), [False])

    def test_no_parseable_answer_is_an_explicit_error(self):
        bank = sample_bank()
        for candidate in bank[0]['candidates']:
            candidate['text'] = 'No numeric answer supplied.'
        with self.assertRaises(ValueError):
            choose(bank, [[0, 1, 2, 3]])

    def test_group_features_follow_candidate_permutation(self):
        bank = sample_bank()
        embeddings = np.random.default_rng(19).normal(size=(1, 4, 1024)).astype(np.float32)
        original, names = build_features(bank, {'embeddings': embeddings})
        order = [2, 0, 3, 1]
        permuted = copy.deepcopy(bank)
        permuted[0]['candidates'] = [permuted[0]['candidates'][i] for i in order]
        for index, candidate in enumerate(permuted[0]['candidates']):
            candidate['index'] = index
        actual, actual_names = build_features(permuted, {'embeddings': embeddings[:, order]})
        self.assertEqual(names, actual_names)
        np.testing.assert_allclose(actual, original[:, order], rtol=1e-5, atol=1e-5)

    def test_checkpoint_accepts_unknown_ids_and_sizes_without_references(self):
        rng = np.random.default_rng(23)
        coefficient = rng.normal(size=1024)
        with tempfile.TemporaryDirectory(prefix='mp2-selector-test-') as directory:
            root = Path(directory)
            np.savez(root / 'weights.npz', coefficient=coefficient)
            config = {'format': FORMAT, 'components': [{'kind': 'linear', 'mode': 'embedding', 'file': 'weights.npz', 'weight': 1.0}]}
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            for size in (1, 3, 7):
                bank = [copy.deepcopy(sample_bank()[0]) for _ in range(size)]
                for index, row in enumerate(bank):
                    row['id'] = 'unseen-' + str(index)
                arrays = {'embeddings': rng.normal(size=(size, 4, 1024)).astype(np.float32)}
                with patch('common.read_jsonl', side_effect=AssertionError('Inference opened references')):
                    actual = predict(bank, arrays, root)
                self.assertEqual(actual, choose(bank, (arrays['embeddings'] @ coefficient)))
                self.assertTrue(all(type(index) is int for index in actual))
            self.assertEqual(predict([], {'embeddings': np.empty((0, 4, 1024), np.float32)}, root), [])


if __name__ == '__main__':
    unittest.main()
