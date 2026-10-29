"""Dataset label and history rules on a tiny synthetic token cache; CPU only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch
except ImportError:  # CI installs only numpy/Pillow; these tests need torch
    torch = None


@unittest.skipIf(torch is None, 'needs torch')
class TrackingDatasetTests(unittest.TestCase):
    def setUp(self):
        from tracking_dataset import DataConfig, JsonTrackingDataset, collate_batch
        self.DataConfig, self.Dataset, self.collate = DataConfig, JsonTrackingDataset, collate_batch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'frames/ep').mkdir(parents=True)
        cache = self.root / 'vision_cache/frames/ep'
        cache.mkdir(parents=True)
        # Frame i's tokens are filled with the value i so padding sources are visible.
        for i in range(12):
            torch.save(torch.full((4, 6), float(i)), cache / f'{i:03d}_vcoarse.pt')
            torch.save(torch.full((64, 6), float(i)), cache / f'{i:03d}_vfine.pt')

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, rows, history=5):
        path = self.root / 'train.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return self.Dataset(self.DataConfig(train_json=str(path), history=history))

    def test_short_history_is_left_padded_with_current_frame(self):
        item = self.dataset([dict(images=['frames/ep/001.jpg', 'frames/ep/002.jpg'],
                                  current='frames/ep/009.jpg', waypoints=[[0, 0, 0]] * 8)])[0]
        per_slot = item['coarse_tokens'].view(5, 4, 6)[:, 0, 0].tolist()
        self.assertEqual(per_slot, [9.0, 9.0, 9.0, 1.0, 2.0])
        self.assertTrue(torch.equal(item['fine_tokens'], torch.full((64, 6), 9.0)))
        self.assertEqual(item['fine_tidx'].unique().tolist(), [5])

    def test_action_labels_use_legacy_index_schedule(self):
        # Nine commands at 1 m/s give nine positions 0.1 s apart; eight are kept and index 4 is dropped.
        item = self.dataset([dict(images=[], current='frames/ep/000.jpg', actions=[[1.0, 0, 0]] * 9)])[0]
        self.assertTrue(np.allclose(item['waypoints'][:, 0].numpy(), [0, .1, .2, .3, .5, .6, .7, .8]))

    def test_valid_idx_becomes_mask_and_batch_stacks(self):
        ds = self.dataset([dict(images=[], current='frames/ep/000.jpg', waypoints=[[0, 0, 0]] * 8,
                                valid_idx=[0, 3]),
                           dict(images=[], current='frames/ep/001.jpg', waypoints=[[1, 0, 0]] * 8)])
        self.assertEqual(ds[0]['valid_mask'].nonzero().flatten().tolist(), [0, 3])
        batch = self.collate([ds[0], ds[1]])
        self.assertEqual(tuple(batch['coarse_tokens'].shape), (2, 20, 6))
        self.assertEqual(tuple(batch['waypoints'].shape), (2, 8, 3))


if __name__ == '__main__':
    unittest.main()
