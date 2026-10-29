"""The single DINOv3+SigLIP pooling recipe matches the inline code it replaced; CPU only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import transformers  # noqa: F401  (cache_gridpool imports it)
except ImportError:  # CI installs only numpy/Pillow
    torch = None


@unittest.skipIf(torch is None, 'needs torch and transformers')
class VisionRecipeTests(unittest.TestCase):
    def setUp(self):
        import cache_gridpool
        self.cg = cache_gridpool
        g = torch.Generator().manual_seed(0)
        self.Hp, self.Wp = 24, 24
        self.dino = torch.randn(2, self.Hp * self.Wp, 384, generator=g)
        self.siglip = torch.randn(2, self.Hp * self.Wp, 1152, generator=g)

    def inline_recipe(self, dino, siglip):
        # Exactly what inference_server, tracking_dataset, precache_frames, trained_agent
        # and cache_gridpool each spelled out before.
        cat = torch.cat([dino, siglip], dim=-1)
        return (self.cg.grid_pool_tokens(cat, self.Hp, self.Wp, out_tokens=64),
                self.cg.grid_pool_tokens(cat, self.Hp, self.Wp, out_tokens=4))

    def test_pool_fine_coarse_is_bit_identical(self):
        fine, coarse = self.cg.pool_fine_coarse(self.dino, self.siglip, self.Hp, self.Wp)
        ref_fine, ref_coarse = self.inline_recipe(self.dino, self.siglip)
        self.assertEqual(tuple(fine.shape), (2, 64, 1536))
        self.assertEqual(tuple(coarse.shape), (2, 4, 1536))
        self.assertTrue(torch.equal(fine, ref_fine) and torch.equal(coarse, ref_coarse))

    def test_pil_entry_point_uses_both_encoders_on_the_dino_grid(self):
        calls = []
        cacher = self.cg.VisionFeatureCacher.__new__(self.cg.VisionFeatureCacher)  # skip weight loading
        cacher._encode_dino = lambda pils: (calls.append('dino') or (self.dino[:1], self.Hp, self.Wp))
        cacher._encode_siglip = lambda pils, out_hw: (calls.append(('siglip', out_hw)) or self.siglip[:1])
        fine, coarse = cacher.encode_pil_fine_coarse(object())
        ref_fine, ref_coarse = self.inline_recipe(self.dino[:1], self.siglip[:1])
        self.assertEqual(calls, ['dino', ('siglip', (self.Hp, self.Wp))])
        self.assertTrue(torch.equal(fine, ref_fine) and torch.equal(coarse, ref_coarse))


if __name__ == '__main__':
    unittest.main()
