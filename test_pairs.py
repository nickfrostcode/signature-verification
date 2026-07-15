import torch
from inference import load_model, predict_siamese

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('siamese', device)
import os
import torch
from inference import load_model, predict_siamese

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('siamese', device)
threshold = 0.345

print("Searching for True Positive (Genuine correctly predicted as > 0.345)...")
found_tp = False
for sig_id in range(1, 10):
    folder = f"SIG_{sig_id:04d}"
    if not os.path.exists(f"data_processed/{folder}/genuine/001.png"): continue
    for j in range(2, 6):
        path_b = f"data_processed/{folder}/genuine/00{j}.png"
        if not os.path.exists(path_b): continue
        p, pred = predict_siamese(model, f"data_processed/{folder}/genuine/001.png", path_b, threshold, device)
        if p >= threshold:
            print(f"✅ Found Genuine Match: {folder}/genuine/001.png vs {folder}/genuine/00{j}.png (Prob: {p:.4f})")
            found_tp = True
            break
    if found_tp: break

print("\nSearching for True Negative (Forged correctly predicted as < 0.345)...")
found_tn = False
for sig_id in range(1, 10):
    folder = f"SIG_{sig_id:04d}"
    if not os.path.exists(f"data_processed/{folder}/genuine/001.png"): continue
    for j in range(1, 6):
        path_b = f"data_processed/{folder}/forged/00{j}.png"
        if not os.path.exists(path_b): continue
        p, pred = predict_siamese(model, f"data_processed/{folder}/genuine/001.png", path_b, threshold, device)
        if p < threshold:
            print(f"✅ Found Forgery Detected: {folder}/genuine/001.png vs {folder}/forged/00{j}.png (Prob: {p:.4f})")
            found_tn = True
            break
    if found_tn: break
