"""Normal-control export must preserve real normal evidence and checker binding."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/hard_tier_expansion_artifacts.py'
spec = importlib.util.spec_from_file_location('hard_tier_controls', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HardControlsTest(unittest.TestCase):
    def record(self):
        return {'asset_id': 'PM_1', 'case_id': 'hp-a', 'files': {},
            'gold': {'reference_sha': 'normal', 'source': {'split': 'train'}},
            'checks': {'clean_check': {'code_correct': True,
                'candidate_sha': 'normal', 'reference_sha': 'normal'},
                'source_authenticity': {'passed': True}},
            'public': {'initial_view': 'front', 'supplementary_view': 'right',
                'states': ['rest', 'mid', 'limit'], 'native_root': 'native/hp-a'},
            'review': {'human_judgment': {'final_decision': None}}}

    def test_control_copies_normal_pixels_and_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            module.write(out / 'source.json', {'asset_id': 'PM_1'})
            record = self.record()
            for view in ('front', 'right'):
                for state in ('rest', 'mid', 'limit'):
                    path = out / 'native/hp-a' / view / ('reference_' + state + '.png')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((view + state).encode())
            module.pair_control(out, record, out / 'source.json', lambda _: SimpleNamespace(sha='normal'))
            paired = record['paired_control']
            for view in ('front', 'right'):
                for state in ('rest', 'mid', 'limit'):
                    directory = out / paired['public']['native_root'] / view
                    self.assertEqual((directory / ('reference_' + state + '.png')).read_bytes(),
                                     (directory / ('candidate_' + state + '.png')).read_bytes())
            self.assertEqual(paired['gold']['source']['split'], 'train')
            self.assertEqual(paired['review']['status'], 'pending_human_review')
            self.assertIsNone(paired['review']['human_judgment']['final_decision'])

    def test_official_archive_verification_is_pinned_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'verification.json'
            module.write(report, {'status': 'verified', 'kind': 'current_archive_verification',
                'archives': [{'asset_id': '46108', 'sha256': 'a' * 64,
                    'status': 'verified_against_official_lfs'}]})
            acquisitions, pins = {}, {}

            module.add_official_archive_verification(report, acquisitions, pins)

            assert acquisitions['46108'] == ('official_verification', module.read(report)['archives'][0])
            assert pins[str(report.resolve())] == module.sha(report)
            module.write(report, {'status': 'verified', 'kind': 'current_archive_verification',
                'archives': [{'asset_id': '46108', 'sha256': 'b' * 64,
                    'status': 'verified_against_official_lfs'}]})
            with self.assertRaises(ValueError):
                module.add_official_archive_verification(report, acquisitions, pins)

    def test_mismatched_source_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            module.write(out / 'source.json', {})
            with self.assertRaisesRegex(ValueError, 'binding failed'):
                module.pair_control(out, self.record(), out / 'source.json',
                                    lambda _: SimpleNamespace(sha='different'))
            self.assertFalse((out / 'native').exists())
