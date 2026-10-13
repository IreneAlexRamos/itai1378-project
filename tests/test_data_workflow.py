"""Offline checks for the costly experiment's review and leakage safeguards."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gnasher_data import GROUPS, build_manifest, draft_label_table, read_reviewed_labels, write_csv, FIELDS


class DataWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'Images'
        self.mapping = {}
        for index, group in enumerate(GROUPS):
            breed = f'Breed_{index}'
            self.mapping[breed] = group
            folder = self.images / f'n{index:08d}-{breed}'
            folder.mkdir(parents=True)
            for photo in range(30):
                (folder / f'{photo}.jpg').write_bytes(f'unique-image-{index}-{photo}'.encode())
        self.table = self.root / 'labels.csv'
        draft_label_table(self.images, self.mapping, {}, self.table)

    def review(self):
        with self.table.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row['review_status'] = 'reviewed'
            row['reviewed_by'] = 'Synthetic test reviewer'
        write_csv(self.table, rows, FIELDS)

    def test_pending_labels_stop_manifest_and_preserve_existing_edits(self):
        original = self.table.read_bytes()
        draft_label_table(self.images, {}, {}, self.table)
        self.assertEqual(original, self.table.read_bytes())
        with self.assertRaisesRegex(ValueError, 'student review'):
            build_manifest(self.images, self.table, self.root / 'experiment', cap=20)

    def test_capped_stratification_has_no_overlap_and_is_reproducible(self):
        self.review()
        # Exact duplicate must be removed, even if filenames differ.
        folder = next(self.images.iterdir())
        (folder / 'duplicate.jpg').write_bytes((folder / '0.jpg').read_bytes())
        first = build_manifest(self.images,self.table,self.root/'one',cap=20,seed=42)
        second = build_manifest(self.images,self.table,self.root/'two',cap=20,seed=42)
        self.assertEqual(first,second)
        self.assertEqual(len(first),140)
        for group in GROUPS:
            rows = [r for r in first if r['group']==group]
            self.assertEqual(sum(r['split']=='train' for r in rows),14)
            self.assertEqual(sum(r['split']=='validation' for r in rows),3)
            self.assertEqual(sum(r['split']=='test' for r in rows),3)
        split_hashes = {split:{r['sha256'] for r in first if r['split']==split}
                        for split in ['train','validation','test']}
        self.assertFalse(split_hashes['train'] & split_hashes['test'])
        self.assertFalse(split_hashes['train'] & split_hashes['validation'])
        self.assertFalse(split_hashes['validation'] & split_hashes['test'])

    def test_conflicting_duplicate_labels_stop_the_split(self):
        self.review()
        folders = sorted(self.images.iterdir())
        (folders[1]/'0.jpg').write_bytes((folders[0]/'0.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError,'conflicting group labels'):
            build_manifest(self.images,self.table,self.root/'experiment',cap=20)

    def test_changed_labels_or_changed_images_cannot_reuse_experiment(self):
        self.review()
        rows = build_manifest(self.images,self.table,self.root/'experiment',cap=20)
        image = self.images / rows[0]['relative_path']
        image.write_bytes(b'changed-photo')
        with self.assertRaisesRegex(ValueError,'image changed'):
            build_manifest(self.images,self.table,self.root/'experiment',cap=20)
        with self.assertRaisesRegex(ValueError,'different labels/settings'):
            build_manifest(self.images,self.table,self.root/'experiment',cap=25)

    def test_unknown_breed_stays_excluded_and_unreviewed(self):
        folder = self.images / 'n99999999-unknown_breed'
        folder.mkdir()
        self.table.unlink()
        draft_label_table(self.images,self.mapping,{},self.table)
        with self.table.open(newline='') as stream:
            unknown = next(r for r in csv.DictReader(stream) if r['stanford_breed_id']==folder.name)
        self.assertEqual(unknown['include'],'no')
        self.assertEqual(unknown['akc_group'],'')
        self.assertEqual(unknown['review_status'],'pending')


if __name__=='__main__':
    unittest.main()
