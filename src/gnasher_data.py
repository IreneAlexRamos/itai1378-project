"""Stanford Dogs inventory, student-reviewed labeling, and reproducible splits.

This module uses the standard library. It never marks labels as human reviewed.
"""
import csv
import hashlib
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

GROUPS = ['Sporting', 'Hound', 'Working', 'Terrier', 'Toy', 'Non-Sporting', 'Herding']
DATASET_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'
FIELDS = ['stanford_breed_id', 'source_breed_name', 'canonical_breed', 'akc_group',
          'include', 'source_url', 'notes', 'review_status', 'reviewed_by']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download_dataset(root):
    """Download original images; accept an already uploaded images.tar."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images = root / 'Images'
    if images.is_dir() and len(list(images.glob('*/*.jpg'))) == 20580:
        return images
    archive = root / 'images.tar'
    if not archive.exists():
        errors = []
        for base in [DATASET_URL.replace('http:', 'https:'), DATASET_URL]:
            partial = root / 'images.tar.partial'
            try:
                with urlopen(base + 'images.tar', timeout=120) as response, partial.open('wb') as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                partial.replace(archive)
                break
            except Exception as exc:
                errors.append(str(exc))
                partial.unlink(missing_ok=True)
        else:
            raise RuntimeError('Stanford download failed. Download images.tar from the official '
                               'dataset page and upload it to this folder. Details: ' + '; '.join(errors))
    # Colab supports the safe data filter; reject links and path traversal.
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    folders = [p for p in images.iterdir() if p.is_dir()]
    if len(folders) != 120 or len(list(images.glob('*/*.jpg'))) != 20580:
        raise ValueError('Expected 120 Stanford Dogs folders and 20,580 JPGs. Check the archive.')
    return images


def normalize(name):
    return re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()


def draft_label_table(images_root, breed_to_group, unsupported_labels, output):
    """Inventory real folder IDs; suggest labels without certifying them."""
    output = Path(output)
    if output.exists():
        return output  # Preserve Irene's edits; validate them before use.
    lookup = {normalize(name): (name, group) for name, group in breed_to_group.items()}
    excluded = {normalize(name): reason for name, reason in unsupported_labels.items()}
    aliases = {'Boston bull': 'Boston Terrier', 'Japanese spaniel': 'Japanese Chin',
               'Blenheim spaniel': 'Cavalier King Charles Spaniel', 'Scotch terrier': 'Scottish Terrier',
               'Mexican hairless': 'Xoloitzcuintli', 'Brabancon griffon': 'Brussels Griffon',
               'English springer': 'English Springer Spaniel', 'basset': 'Basset Hound'}
    rows = []
    for folder in sorted(Path(images_root).iterdir()):
        if not folder.is_dir():
            continue
        source_name = folder.name.split('-', 1)[-1]
        matched = lookup.get(normalize(source_name))
        label, group = matched if matched else (source_name.replace('_', ' '), '')
        rows.append(dict(zip(FIELDS, [folder.name, source_name, aliases.get(label, label), group,
                    'yes' if matched else 'no',
                    'https://www.akc.org/dog-breeds/' + group.lower() + '/' if matched else DATASET_URL,
                    'Suggested from V1; verify breed and AKC group.' if matched else
                    excluded.get(normalize(source_name), 'No verified V1 match; exclude until reviewed.'),
                    'pending', ''])))
    write_csv(output, rows, FIELDS)
    return output


def read_reviewed_labels(path, images_root):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    folders = {p.name for p in Path(images_root).iterdir() if p.is_dir()}
    identifiers = [r['stanford_breed_id'] for r in rows]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != folders:
        raise ValueError('Label table must contain each source breed folder exactly once.')
    problems = []
    for row in rows:
        if row['review_status'].strip().lower() != 'reviewed' or not row['reviewed_by'].strip():
            problems.append(row['stanford_breed_id'])
        if row['include'].strip().lower() not in ('yes', 'no'):
            raise ValueError('include must be yes or no.')
        if not row['canonical_breed'].strip() or not row['source_url'].strip():
            raise ValueError('Every label needs a canonical name and source URL.')
        if row['include'].strip().lower() == 'yes' and row['akc_group'] not in GROUPS:
            raise ValueError('Included labels need one of the seven AKC groups.')
        if row['include'].strip().lower() == 'no' and not row['notes'].strip():
            raise ValueError('Excluded labels need a documented reason.')
    if problems:
        raise ValueError(f'{len(problems)} labels need student review. Edit the table, verify each '
                         'assignment/exclusion, set review_status=reviewed and reviewed_by=your name.')
    return rows


def build_manifest(images_root, label_path, output_dir, cap=1000, seed=42):
    """Deduplicate before stratified 70/15/15 splitting; never silently change a split."""
    if cap < 20:
        raise ValueError('Use a cap of at least 20 images per group.')
    images_root, output_dir = Path(images_root), Path(output_dir)
    labels = read_reviewed_labels(label_path, images_root)
    label_sha = sha256_file(label_path)
    manifest_path = output_dir / 'manifest.csv'
    metadata_path = output_dir / 'manifest_metadata.json'
    if manifest_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata['label_sha256'], metadata['cap'], metadata['seed']) != (label_sha, cap, seed):
            raise ValueError('Existing experiment uses different labels/settings. Use a new output folder.')
        if sha256_file(manifest_path) != metadata['manifest_sha256']:
            raise ValueError('Manifest was changed after creation. Use a new experiment folder.')
        with manifest_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if sha256_file(images_root / row['relative_path']) != row['sha256']:
                raise ValueError('An experiment image changed after split creation.')
        return rows
    candidates = []
    for label in labels:
        if label['include'].strip().lower() == 'yes':
            for image in sorted((images_root / label['stanford_breed_id']).glob('*.jpg')):
                candidates.append({'relative_path': image.relative_to(images_root).as_posix(),
                                   'breed_id': label['stanford_breed_id'], 'group': label['akc_group'],
                                   'sha256': sha256_file(image)})
    hashes = defaultdict(list)
    for row in candidates:
        hashes[row['sha256']].append(row)
    conflicts = [h for h, members in hashes.items() if len({r['group'] for r in members}) > 1]
    if conflicts:
        raise ValueError(f'{len(conflicts)} identical images have conflicting group labels; review them first.')
    unique = [members[0] for members in hashes.values()]
    grouped = defaultdict(list)
    for row in unique:
        grouped[row['group']].append(row)
    rng = random.Random(seed)
    rows = []
    for group in GROUPS:
        pool = sorted(grouped[group], key=lambda r: r['relative_path'])
        rng.shuffle(pool)
        pool = pool[:cap]
        n = len(pool)
        if n < 20:
            raise ValueError(f'{group} has only {n} unique images. Review labels or dataset.')
        n_train, n_val = int(n * .70), int(n * .15)
        for index, row in enumerate(pool):
            rows.append({**row, 'split': 'train' if index < n_train else
                         'validation' if index < n_train + n_val else 'test'})
    assert len({r['sha256'] for r in rows}) == len(rows)
    fields = ['relative_path', 'breed_id', 'group', 'sha256', 'split']
    write_csv(manifest_path, rows, fields)
    metadata = {'seed': seed, 'cap': cap, 'label_sha256': label_sha,
                'manifest_sha256': sha256_file(manifest_path), 'class_names': GROUPS,
                'selected_images': len(rows), 'exact_duplicates_removed': len(candidates)-len(unique),
                'counts': {group: dict(Counter(r['split'] for r in rows if r['group']==group))
                           for group in GROUPS},
                'split_method': 'custom group-stratified 70/15/15; not Stanford official split',
                'limitations': 'Exact-byte deduplication only; near duplicates and ImageNet pretraining overlap remain possible.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return rows
