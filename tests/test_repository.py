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
        for name in ('methods/omtrack_vla/run_local.sh','methods/omtrack_vla/ridgeback/start_omtrack_ridgeback.sh','methods/omtrack_vla/ridgeback/run_remote_omtrack_case.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('/home/robot/',text)
            self.assertNotIn('DyNAMO',text.replace('no DyNAMO',''))
    def test_live_preview_has_no_ros_publisher(self):
        tree=ast.parse((ROOT/'methods/omtrack_vla/ridgeback/live_omtrack_preview.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='create_publisher' for n in ast.walk(tree)))
    def test_repo_imports_point_at_existing_modules(self):
        # Most scripts can't be run here (ROS, GPU), so check statically
        # that every import of a repository module resolves.
        local=('leader_follow','robot','methods')
        sources=[p for d in local+('tests',) for p in (ROOT/d).rglob('*.py')]
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
    def test_leader_follow_does_not_depend_on_a_method(self):
        # Methods plug into the system's stages, never the other way round.
        for path in (ROOT/'leader_follow').rglob('*.py'):
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertNotIn('methods',path.read_text())
    def test_nothing_imports_habitat(self):
        for path in [p for d in ('leader_follow','robot','methods') for p in (ROOT/d).rglob('*.py')]:
            for node in ast.walk(ast.parse(path.read_text())):
                names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                with self.subTest(file=str(path.relative_to(ROOT))):
                    self.assertFalse(any(n.split('.')[0] in ('habitat','habitat_sim') for n in names))
