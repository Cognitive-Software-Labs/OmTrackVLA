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
        for name in ('run-local.sh','real_robot/start_omtrack_ridgeback.sh','real_robot/run_remote_omtrack_case.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('/home/robot/',text)
            self.assertNotIn('DyNAMO',text.replace('no DyNAMO',''))
    def test_live_preview_has_no_ros_publisher(self):
        tree=ast.parse((ROOT/'real_robot/live_omtrack_preview.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='create_publisher' for n in ast.walk(tree)))
    def test_core_inference_has_no_habitat_import(self):
        tree=ast.parse((ROOT/'infer-sample.py').read_text())
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('habitat','trained_agent')) for n in names))
