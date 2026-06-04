import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.siamese import SiameseNetwork
from datasets.base_dataset import get_all_subjects, split_subjects
from datasets.pair_dataset import PairDataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    'epochs':        50,
    'batch_size':    16,
    'learning_rate': 3e-4,
    'patience':      10,
    'margin':        0.5,
    'save_path':     os.path.abspath(
                         os.path.join(os.path.dirname(__file__),
                                      '..', 'saved_models', 'siamese.pth')
                     ),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix random seeds for reproducibility when comparing runs.
torch.manual_seed(42)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity-based metric learning."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distances, labels):
        labels = labels.view(-1)
        distances = distances.view(-1)
        similar_loss = (1 - labels) * distances.pow(2)
        dissimilar_loss = labels * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return torch.mean(similar_loss + dissimilar_loss)


def train_one_epoch(model, loader, optimizer, criterion):
    """
    One full training pass over all pairs.
    Each batch contains (img_a, img_b, label) triplets.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for img_a, img_b, labels in loader:
        img_a  = img_a.to(DEVICE)
        img_b  = img_b.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)  # (batch,) → (batch, 1)

        preds = model(img_a, img_b)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predicted   = (preds > CONFIG['margin'] / 2).float()
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    """
    One full evaluation pass over all validation pairs.
    No gradient computation.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for img_a, img_b, labels in loader:
            img_a  = img_a.to(DEVICE)
            img_b  = img_b.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            preds  = model(img_a, img_b)
            loss   = criterion(preds, labels)

            total_loss += loss.item()
            predicted   = (preds > CONFIG['margin'] / 2).float()
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


def train():
    print("=" * 55)
    print("TRAINING — Siamese Network")
    print(f"Device: {DEVICE}")
    print("=" * 55 + "\n")

    # ── Data ──────────────────────────────────────────────
    subjects                        = get_all_subjects()
    train_subjects, val_subjects, _ = split_subjects(subjects)

    train_ds = PairDataset(train_subjects, augment=True)
    val_ds   = PairDataset(val_subjects, augment=False)

    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)

    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == 'cuda')
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == 'cuda')
    )

    print(f"\nTrain pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,}")
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}\n")

    # ── Model ─────────────────────────────────────────────
    model = SiameseNetwork().to(DEVICE)

    # ── Loss ──────────────────────────────────────────────
    label_counts = train_ds.get_label_counts()
    print(f"  PairDataset label counts: {label_counts} | margin={CONFIG['margin']:.2f}")

    criterion = ContrastiveLoss(margin=CONFIG['margin'])

    # ── Optimizer ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-5
    )

    # ── Scheduler ─────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Training Loop ─────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(1, CONFIG['epochs'] + 1):

        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_dl, criterion)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         val_loss,
                'val_acc':          val_acc,
            }, CONFIG['save_path'])
            print(f"           ✅ Best model saved (val_loss: {val_loss:.4f})")

        else:
            patience_count += 1
            print(f"           ⏳ No improvement ({patience_count}/{CONFIG['patience']})")

            if patience_count >= CONFIG['patience']:
                print(f"\n⛔ Early stopping at epoch {epoch}")
                break

    print(f"\n{'=' * 55}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CONFIG['save_path']}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    train()