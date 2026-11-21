"""Scaled GRPO labels and epoch conversion, without training or GPUs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sgs.tasks import Sorting
from visualize.sort_curves import available_configs, grpo_rows, user_content


class ScaledCurvesTest(unittest.TestCase):
    def test_optional_scaled_labels(self):
        self.assertEqual(available_configs({'grpo-L8-128', 'sgs-onpolicy-grpo-reset-L8-128'}),
                         ['L8-128'])
        self.assertEqual(available_configs({'grpo-1.5x-all', 'grpo-2x-L8-128',
                                           'grpo-sgs-synth-L8-16-128'}), ['L8-128', 'all'])

    def test_epochs_and_sample_grouping(self):
        for size, steps in ((1500, 46), (2000, 62)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                (run / 'eval_samples').mkdir()
                # Deliberately evaluate more often than once per epoch.
                (run / 'args.json').write_text(json.dumps({
                    'dataset': 'train', 'val_dataset': 'val', 'train_batch_size': 32,
                    'eval_every': 2}))
                for step in (0, steps, 5 * steps):
                    samples = []
                    for p, scores in (([2, 1], [0, 0, 1, 0, 0]), ([4, 3], [0] * 5)):
                        samples += [{'input': 'user\n' + user_content(Sorting, p) + '\nassistant\n',
                                     'acc': score} for score in scores]
                    (run / 'eval_samples' / f'{step}.jsonl').write_text(
                        ''.join(json.dumps(r) + '\n' for r in samples))
                with patch.object(Sorting, 'load', side_effect=lambda p: [[2, 1]] * size if p == 'train' else [[4, 3]]):
                    rows = list(grpo_rows(run, Sorting))
                    self.assertEqual({r['epoch'] for r in rows}, {0, 1, 5})
                    self.assertTrue(all(r['pass'] == (1 if r['split'] == 'train' else 0) for r in rows))
                    rows = list(grpo_rows(run, Sorting, 'pass1'))
                    self.assertTrue(all(r['pass'] == (0.2 if r['split'] == 'train' else 0) for r in rows))


if __name__ == '__main__':
    unittest.main()
