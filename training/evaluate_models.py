import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from models.baseline_cnn import BaselineCNN
from models.siamese import SiameseNetwork
from datasets.base_dataset import get_all_subjects, split_subjects
from datasets.single_dataset import SingleDataset
from datasets.pair_dataset import PairDataset


def compute_metrics(labels, scores, threshold=None, positive_label=1):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    if threshold is None:
        threshold = 0.5

    preds = (scores > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(labels, scores)
    except ValueError:
        metrics['roc_auc'] = None

    return metrics


def find_best_threshold(labels, scores, num_thresholds=500):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    min_score, max_score = float(scores.min()), float(scores.max())
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    best_threshold = thresholds[0]
    best_f1 = -1.0
    best_metrics = None

    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = compute_metrics(labels, scores, threshold=threshold)

    return best_threshold, best_metrics


def evaluate_baseline(model_path, dataset, device, batch_size=16, num_workers=0):
    model = BaselineCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    criterion = nn.BCEWithLogitsLoss()
    all_labels = []
    all_scores = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels.unsqueeze(1))
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    average_loss = total_loss / total_samples
    metrics = compute_metrics(all_labels, all_scores, threshold=0.5)
    best_threshold, best_metrics = find_best_threshold(all_labels, all_scores)

    return {
        'loss': average_loss,
        'threshold': 0.5,
        'metrics': metrics,
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
    }


def evaluate_siamese(model_path, dataset, device, batch_size=16, num_workers=0, threshold=0.5):
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for img_a, img_b, labels in loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)
            total_loss += loss.item() * logits.size(0)
            total_samples += logits.size(0)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.squeeze(1).cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    average_loss = total_loss / total_samples

    default_metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    best_threshold, best_metrics = find_best_threshold(all_labels, all_probs)

    return {
        'loss': average_loss,
        'threshold': threshold,
        'metrics': default_metrics,
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
    }


def print_report(name, stats):
    print(f"--- {name} ---")
    for key, value in stats.items():
        if key == 'metrics' or key == 'best_metrics':
            print(f"{key}:")
            for metric_name, metric_value in value.items():
                print(f"  {metric_name}: {metric_value}")
        else:
            print(f"{key}: {value}")
    print()


def main(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    subjects = get_all_subjects()
    train_subjects, val_subjects, test_subjects = split_subjects(subjects)

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)

    if args.model in ('baseline', 'all'):
        print('Evaluating baseline model...')
        baseline_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'baseline_cnn.pth')
        )
        baseline_model = BaselineCNN()  # just to verify import works

        for split_name, subject_list in [('val', val_subjects), ('test', test_subjects)]:
            dataset = SingleDataset(subject_list, augment=False)
            stats = evaluate_baseline(
                baseline_model_path,
                dataset,
                device,
                batch_size=args.batch_size,
                num_workers=num_workers,
            )
            print(f"Baseline {split_name} set: {len(dataset)} images")
            print_report(f"Baseline {split_name}", stats)

    if args.model in ('siamese', 'all'):
        print('Evaluating Siamese model...')
        siamese_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'siamese.pth')
        )
        siamese_model = SiameseNetwork()  # verify import

        for split_name, subject_list in [('val', val_subjects), ('test', test_subjects)]:
            dataset = PairDataset(subject_list, augment=False)
            stats = evaluate_siamese(
                siamese_model_path,
                dataset,
                device,
                batch_size=args.batch_size,
                num_workers=num_workers,
                threshold=args.threshold,
            )
            print(f"Siamese {split_name} set: {len(dataset)} pairs")
            print_report(f"Siamese {split_name}", stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baseline and Siamese signature models.')
    parser.add_argument('--model', choices=['baseline', 'siamese', 'all'], default='all')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda or cpu. Defaults to cuda if available.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Default probability threshold for Siamese BCE evaluation.')
    args = parser.parse_args()
    main(args)
