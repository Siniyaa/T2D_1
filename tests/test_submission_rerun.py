import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    'submission_rerun', Path(__file__).resolve().parents[1] / 'code' / 'submission_rerun.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PackageBoundaryTest(unittest.TestCase):
    def make_outputs(self, root):
        for directory, names in [('results', runner.RESULT_FILES), ('models', runner.MODEL_FILES)]:
            (root / directory).mkdir()
            for name in names:
                (root / directory / name).write_text('aggregate test fixture\n')

    def test_participant_and_unlisted_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_outputs(root)
            (root / 'results' / 'participant_predictions.csv').write_text('EID,probability\n101,0.2\n')
            (root / 'medical_history_3.tsv').write_text('participant extract\n')
            archive = runner.package_outputs(root, 'test-revision')
            with ZipFile(archive) as z:
                self.assertNotIn('results/participant_predictions.csv', z.namelist())
                self.assertNotIn('medical_history_3.tsv', z.namelist())
                self.assertEqual(len(z.namelist()), len(runner.RESULT_FILES) + len(runner.MODEL_FILES) + 1)
                manifest = json.loads(z.read('revision_manifest.json'))
                self.assertEqual(manifest['code_revision'], 'test-revision')

    def test_missing_expected_output_prevents_packaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_outputs(root)
            (root / 'results' / 'Calibration_sex_specific.csv').unlink()
            with self.assertRaises(ValueError):
                runner.package_outputs(root, 'test-revision')
            self.assertFalse((root / 'T2D_revision_outputs.zip').exists())


if __name__ == '__main__':
    unittest.main()
