from datasets.base_dataset import get_all_subjects, split_subjects
from datasets.pair_dataset import PairDataset
from datasets.single_dataset import SingleDataset
from torch.utils.data import DataLoader

subjects = get_all_subjects()
train_subjects, val_subjects, test_subjects = split_subjects(subjects)

print(f"Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}\n")

# Test PairDataset
pair_ds = PairDataset(train_subjects)
pair_dl = DataLoader(pair_ds, batch_size=8, shuffle=True)
img_a, img_b, label = next(iter(pair_dl))
print(f"\nPair batch — img_a: {img_a.shape} | img_b: {img_b.shape} | label: {label.shape}")

# Test SingleDataset
single_ds = SingleDataset(train_subjects)
single_dl = DataLoader(single_ds, batch_size=8, shuffle=True)
img, label = next(iter(single_dl))
print(f"Single batch — img: {img.shape} | label: {label.shape}")