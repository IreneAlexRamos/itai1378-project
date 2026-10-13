"""Version 2: EfficientNet transfer learning and paired held-out evaluation.

Training uses validation for model/threshold selection. Test results are written
once per experiment. No accuracy or runtime results are supplied in advance.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import InterpolationMode

from gnasher_data import GROUPS, sha256_file, write_csv


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PhotoDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['relative_path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, GROUPS.index(row['group']), index


def make_loaders(images_root, rows, batch_size=32, seed=42):
    weights = EfficientNet_B0_Weights.DEFAULT
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.70, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ['train', 'validation', 'test']:
        split_rows = [r for r in rows if r['split'] == split]
        transform = train_transform if split == 'train' else weights.transforms()
        dataset = PhotoDataset(images_root, split_rows, transform)
        # num_workers=0 is intentional for predictable Colab execution.
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split=='train',
                                   num_workers=0, generator=generator,
                                   pin_memory=torch.cuda.is_available())
    return loaders


def build_model(pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(GROUPS))
    return model


def configure_stage(model, fine_tune):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if fine_tune:
        for block in model.features[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None, fine_tune=False):
    training = optimizer is not None
    model.train(training)
    if training:
        # Keep frozen BatchNorm statistics and stochastic layers frozen too.
        model.features.eval()
        if fine_tune:
            model.features[-3:].train()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)
    return {'loss': total_loss / total, 'accuracy': correct / total}


def train_model(loaders, output_dir, manifest_metadata, seed=42, head_epochs=3, fine_epochs=5):
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'history.csv').exists() or (output / 'best.pt').exists():
        raise ValueError('This experiment already has training artifacts. Use a new experiment name '
                         'to train again, or load its saved checkpoint for evaluation.')
    if head_epochs < 1 or fine_epochs < 1:
        raise ValueError('Run at least one epoch for each training stage.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    train_counts = np.bincount([GROUPS.index(r['group']) for r in loaders['train'].dataset.rows],
                              minlength=len(GROUPS))
    class_weights = torch.tensor(train_counts.sum() / (len(GROUPS) * train_counts),
                                 dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history, best_accuracy = [], -1.0
    common = {'class_names': GROUPS, 'seed': seed, 'manifest_metadata': manifest_metadata,
              'torch_version': str(torch.__version__), 'torchvision_version': str(torchvision.__version__),
              'device': str(device), 'pretrained_weights': 'EfficientNet_B0_Weights.IMAGENET1K_V1',
              'head_epochs_requested': head_epochs, 'fine_epochs_requested': fine_epochs}
    for stage, epochs, lr in [('head', head_epochs, 1e-3), ('fine_tune', fine_epochs, 1e-4)]:
        fine_tune = stage == 'fine_tune'
        configure_stage(model, fine_tune)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=lr, weight_decay=1e-4)
        stale = 0
        stage_best = -1.0
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train = run_epoch(model, loaders['train'], criterion, device, optimizer, fine_tune)
            validation = run_epoch(model, loaders['validation'], criterion, device)
            row = {'stage': stage, 'epoch': epoch, 'learning_rate': lr,
                   'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
                   'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy'],
                   'epoch_seconds': time.perf_counter() - started}
            history.append(row)
            write_csv(output / 'history.csv', history, list(row))
            checkpoint = {**common, 'stage': stage, 'epoch': epoch,
                          'validation_accuracy': validation['accuracy'], 'state_dict': model.state_dict()}
            torch.save(checkpoint, output / 'last.pt')
            if validation['accuracy'] > best_accuracy:
                best_accuracy = validation['accuracy']
                torch.save(checkpoint, output / 'best.pt')
            if validation['accuracy'] > stage_best:
                stage_best, stale = validation['accuracy'], 0
            else:
                stale += 1
            print(f"{stage} {epoch}/{epochs}: train {train['accuracy']:.1%}, "
                  f"validation {validation['accuracy']:.1%}, {row['epoch_seconds']:.0f}s")
            if stale >= 3:
                print('Early stopping based on validation only.')
                break
        # Begin the next stage from the best validation checkpoint so far.
        model.load_state_dict(torch.load(output / 'best.pt', map_location=device,
                                        weights_only=True)['state_dict'])
    return model, device


def load_best(output_dir, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(Path(output_dir) / 'best.pt', map_location=device, weights_only=True)
    if checkpoint['class_names'] != GROUPS:
        raise ValueError('Checkpoint class order does not match the project.')
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, device, checkpoint


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for images, labels, _ in loader:
        probabilities.extend(model(images.to(device)).softmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    return np.array(probabilities), np.array(targets)


def select_threshold(model, validation_loader, device, output_dir):
    """Choose on validation, never on test; retain rejected examples in metrics."""
    saved = Path(output_dir) / 'threshold.json'
    if saved.exists():
        selection = json.loads(saved.read_text())
        if selection['checkpoint_sha256'] != sha256_file(Path(output_dir) / 'best.pt'):
            raise ValueError('Saved threshold belongs to another checkpoint; use a new experiment.')
        return selection
    probabilities, targets = predict_loader(model, validation_loader, device)
    confidence, prediction = probabilities.max(1), probabilities.argmax(1)
    candidates = []
    for threshold in np.arange(.50, .96, .05):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        accuracy = float((prediction[accepted] == targets[accepted]).mean()) if accepted.any() else None
        candidates.append({'threshold': round(float(threshold), 2), 'coverage': coverage,
                           'accepted_accuracy': accuracy})
    qualifying = [c for c in candidates if c['coverage'] >= .80 and c['accepted_accuracy'] >= .85]
    selected = qualifying[0]['threshold'] if qualifying else .50
    selection = {'threshold': selected, 'selection_data': 'validation only',
                 'rule': 'smallest threshold >=0.50 with >=80% validation coverage and >=85% '
                         'accepted accuracy; fallback 0.50 if none qualify',
                 'criterion_met': bool(qualifying), 'candidates': candidates,
                 'checkpoint_sha256': sha256_file(Path(output_dir) / 'best.pt')}
    (Path(output_dir) / 'threshold.json').write_text(json.dumps(selection, indent=2))
    return selection


def summarize_predictions(probabilities, targets, threshold):
    prediction = probabilities.argmax(1)
    confidence = probabilities.max(1)
    accepted = confidence >= threshold
    confusion = np.zeros((len(GROUPS), len(GROUPS)), dtype=int)
    for truth, guessed in zip(targets, prediction):
        confusion[truth, guessed] += 1
    accepted_confusion = np.zeros((len(GROUPS), len(GROUPS)+1), dtype=int)
    for truth, guessed, keep in zip(targets, prediction, accepted):
        accepted_confusion[truth, guessed if keep else len(GROUPS)] += 1
    return {'n': len(targets), 'top1_accuracy': float((prediction == targets).mean()),
            'coverage': float(accepted.mean()), 'abstentions': int((~accepted).sum()),
            'accepted_accuracy': float((prediction[accepted] == targets[accepted]).mean())
                                 if accepted.any() else None,
            'correct_with_abstentions_counted_in_denominator': float(((prediction == targets)&accepted).mean()),
            'confusion_matrix': confusion.tolist(), 'decision_matrix_with_uncertain': accepted_confusion.tolist(),
            'per_group_accuracy': {g: float((prediction[targets==i]==i).mean())
                                   for i,g in enumerate(GROUPS)}}


@torch.inference_mode()
def evaluate_paired(model, test_loader, device, output_dir, aggregate_group_scores):
    """Compare V1 and V2 on identical test photographs; lock report to checkpoint."""
    output = Path(output_dir)
    checkpoint_sha = sha256_file(output / 'best.pt')
    metadata = json.loads((output / 'manifest_metadata.json').read_text())
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['manifest_metadata'] != metadata:
        raise ValueError('Training checkpoint and test manifest do not match.')
    selection = json.loads((output / 'threshold.json').read_text())
    threshold_sha = sha256_file(output / 'threshold.json')
    if selection['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Threshold was chosen for a different checkpoint.')
    report_path = output / 'test_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report['checkpoint_sha256'] != checkpoint_sha or
            report['manifest_sha256'] != metadata['manifest_sha256'] or
            report['threshold_sha256'] != threshold_sha):
            raise ValueError('A different model or split has already been tested in this experiment.')
        print('Returning the already recorded test report; do not tune using these results.')
        return report
    probabilities, targets = predict_loader(model, test_loader, device)
    weights = EfficientNet_B0_Weights.DEFAULT
    baseline = efficientnet_b0(weights=weights).to(device).eval()
    old_probabilities = []
    for images, _, _ in test_loader:
        for raw in baseline(images.to(device)).softmax(1).cpu().tolist():
            result = aggregate_group_scores(weights.meta['categories'], raw)
            old_probabilities.append([result['groups'][group] for group in GROUPS])
    old_probabilities = np.array(old_probabilities)
    threshold = selection['threshold']
    report = {'checkpoint_sha256': checkpoint_sha, 'manifest_sha256': metadata['manifest_sha256'],
              'threshold_sha256': threshold_sha,
              'class_names': GROUPS, 'device': str(device), 'v2_threshold': threshold,
              'prototype_v1': summarize_predictions(old_probabilities, targets, .50),
              'trained_v2': summarize_predictions(probabilities, targets, threshold),
              'limitations': 'Held out from project fine-tuning only; ImageNet pretraining overlap '
                             'and near duplicates are possible. No reliable non-dog detection established.'}
    rows = []
    for index, row in enumerate(test_loader.dataset.rows):
        ranking = np.argsort(-probabilities[index])
        first, second = int(ranking[0]), int(ranking[1])
        v1 = int(old_probabilities[index].argmax())
        rows.append({**row, 'expected_group': GROUPS[int(targets[index])],
                     'v1_leading_group': GROUPS[v1], 'v1_score': float(old_probabilities[index,v1]),
                     'v1_decision': GROUPS[v1] if old_probabilities[index,v1]>=.50 else 'Uncertain',
                     'v2_top1': GROUPS[first], 'v2_top1_score': float(probabilities[index,first]),
                     'v2_top2': GROUPS[second], 'v2_top2_score': float(probabilities[index,second]),
                     'v2_decision': GROUPS[first] if probabilities[index,first]>=threshold else 'Uncertain'})
    write_csv(output / 'test_predictions.csv', rows, list(rows[0]))
    report_path.write_text(json.dumps(report, indent=2))
    return report


@torch.inference_mode()
def benchmark_model(model, image_path, device, runs=30, warmups=5):
    model.eval()
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    with Image.open(image_path) as image:
        batch = transform(image.convert('RGB')).unsqueeze(0).to(device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for _ in range(warmups):
        model(batch)
    sync()
    times = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        model(batch)
        sync()
        times.append(time.perf_counter()-start)
    return {'device': str(device), 'runs': runs, 'warmups': warmups,
            'mean_forward_seconds': float(np.mean(times)),
            'median_forward_seconds': float(np.median(times)),
            'scope': 'preloaded batch, model forward only; excludes image loading and preprocessing'}


@torch.inference_mode()
def predict_photo(model, image_path, device, threshold):
    with Image.open(image_path) as image:
        batch = EfficientNet_B0_Weights.DEFAULT.transforms()(image.convert('RGB')).unsqueeze(0).to(device)
    model.eval()
    values, indices = model(batch).softmax(1)[0].topk(2)
    candidates = [{'group': GROUPS[int(i)], 'score': float(s)} for s,i in zip(values,indices)]
    return {'decision': candidates[0]['group'] if candidates[0]['score']>=threshold else 'Uncertain',
            'top_two': candidates, 'note': 'Seven-group closed-set model; this is not a non-dog detector.'}
