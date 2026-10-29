import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_pinned_models(self):
        revisions=json.loads((ROOT/'requirements/model-revisions.json').read_text())
        self.assertEqual(len(revisions),5)
        for value in revisions.values():self.assertRegex(value,r'^[0-9a-f]{40}$')
    def test_launcher_has_no_old_checkout_dependency(self):
        for name in ('run_local.sh','real_robot/start_omtrack_ridgeback.sh','real_robot/run_remote_omtrack_case.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('/home/robot/',text)
            self.assertNotIn('DyNAMO',text.replace('no DyNAMO',''))
    def test_live_preview_has_no_ros_publisher(self):
        tree=ast.parse((ROOT/'real_robot/live_omtrack_preview.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='create_publisher' for n in ast.walk(tree)))
    def test_repo_imports_point_at_existing_modules(self):
        # Scripts in training/, sim/ and scripts/ can't all be run here (Habitat, GPU),
        # so check statically that every import of a repository module resolves.
        local=('omtrack','training','sim','real_robot','scripts')
        sources=[p for d in local for p in (ROOT/d).rglob('*.py')]+[ROOT/'infer_sample.py']
        for path in sources:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
                    names=[node.module]
                elif isinstance(node,ast.Import):
                    names=[a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if name.split('.')[0] not in local:
                        continue
                    base=ROOT.joinpath(*name.split('.'))
                    with self.subTest(file=str(path.relative_to(ROOT)),module=name):
                        self.assertTrue(base.with_suffix('.py').is_file() or (base/'__init__.py').is_file())
    def test_core_inference_has_no_habitat_import(self):
        tree=ast.parse((ROOT/'infer_sample.py').read_text())
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('habitat','sim')) for n in names))
