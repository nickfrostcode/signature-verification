import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline_cnn import BaselineCNN
from datasets.base_dataset import get_all_subjects, split_subjects
from datasets.single_dataset import SingleDataset

# ─────────────────────────────────────────────
# CONFIG
# All training hyperparameters in one place.
# Change these without touching the training logic.
# ─────────────────────────────────────────────
CONFIG = {
    'epochs':        50,
    'batch_size':    16,      # smaller batch for CPU — fits in RAM comfortably
    'learning_rate': 3e-4,
    'patience':      10,       # increased patience per audit
    'save_path':     os.path.abspath(
                         os.path.join(os.path.dirname(__file__),
                                      '..', 'saved_models', 'baseline_cnn.pth')
                     ),
    # Class weights are computed dynamically from the train split
    # when this value is None. Otherwise, a fixed weight is used.
    'pos_weight':    None,
}

# Use GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix random seeds for reproducibility when comparing runs.
torch.manual_seed(42)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Run one full pass over the training data.
    Updates model weights via backpropagation.

    Returns average loss and accuracy for the epoch.
    """
    model.train()   # enables dropout and batchnorm training behavior
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)  # (batch,) → (batch, 1)

        # Forward pass
        preds = model(imgs)
        loss  = criterion(preds, labels)

        # Backward pass — compute gradients and update weights
        optimizer.zero_grad()   # clear gradients from previous batch
        loss.backward()         # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        # update weights

        total_loss += loss.item()
        predicted   = (torch.sigmoid(preds) > 0.5).float()
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    """
    Run one full pass over validation data.
    No weight updates — evaluation only.

    Returns average loss and accuracy.
    """
    model.eval()    # disables dropout and uses running batchnorm stats
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():   # disable gradient computation for speed
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            preds  = model(imgs)
            loss   = criterion(preds, labels)

            total_loss += loss.item()
            predicted   = (torch.sigmoid(preds) > 0.5).float()
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    print("=" * 55)
    print("TRAINING — Baseline CNN")
    print(f"Device: {DEVICE}")
    print("=" * 55 + "\n")

    # ── Data ──────────────────────────────────────────────
    subjects                             = get_all_subjects()
    train_subjects, val_subjects, _      = split_subjects(subjects)

    train_ds = SingleDataset(train_subjects, augment=True)
    val_ds   = SingleDataset(val_subjects, augment=False)

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

    print(f"\nTrain batches: {len(train_dl)} | Val batches: {len(val_dl)}\n")

    # ── Model ─────────────────────────────────────────────
    model = BaselineCNN().to(DEVICE)

    # ── Loss ──────────────────────────────────────────────
    # BCELoss with pos_weight penalizes missing forged signatures more
    # pos_weight=2.0 means forged errors cost twice as much
    # This compensates for the 2:1 genuine/forged class imbalance
    label_counts = train_ds.get_label_counts()
    pos_weight = CONFIG['pos_weight']
    if pos_weight is None:
        pos_weight = label_counts[0] / max(label_counts[1], 1)
    print(f"  SingleDataset label counts: {label_counts} | pos_weight={pos_weight:.4f}")

    # Use logits + stable binary loss with class imbalance support.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(DEVICE)
    )

    # ── Optimizer ─────────────────────────────────────────
    # AdamW adds decoupled weight decay as a regularizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-5
    )

    # ── Scheduler ─────────────────────────────────────────
    # Reduce LR by 50% if val loss doesn't improve for 3 epochs
    # Helps escape plateaus in the loss landscape
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Training Loop ─────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0            # tracks epochs without improvement

    for epoch in range(1, CONFIG['epochs'] + 1):

        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_dl, criterion)

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ── Save best model ────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':   val_loss,
                'val_acc':    val_acc,
            }, CONFIG['save_path'])
            print(f"           ✅ Best model saved (val_loss: {val_loss:.4f})")

        else:
            patience_count += 1
            print(f"           ⏳ No improvement ({patience_count}/{CONFIG['patience']})")

            # ── Early stopping ─────────────────────────────
            if patience_count >= CONFIG['patience']:
                print(f"\n⛔ Early stopping at epoch {epoch}")
                break

    print(f"\n{'=' * 55}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CONFIG['save_path']}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    train()