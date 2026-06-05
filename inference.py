"""Signature verification CLI with explainability.

This script provides a command-line inference interface for the
signature verification project. It supports two modes:

- baseline: classify a single signature image as genuine or forged
- siamese: compare two signatures and decide whether they match

The script also supports explainability by generating saliency heatmaps
and overlay images for the input signatures.

Example usages:
    python inference.py --model baseline --image 002.jpg --explain
    python inference.py --model siamese --image-a 001.jpg --image-b 002.jpg --threshold 0.345 --explain
"""

import argparse
import os
import torch
from PIL import Image

from models.baseline_cnn import BaselineCNN
from models.siamese import SiameseNetwork
from datasets.base_dataset import load_stats, load_image
from explainability.saliency import (
    compute_input_saliency,
    compute_siamese_saliency,
    save_raw_saliency,
    save_saliency_overlay,
)

# Paths to saved model weights. These files are created when training is complete.
BASELINE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models', 'baseline_cnn.pth'))
SIAMESE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models', 'siamese.pth'))

# Default decision thresholds found through evaluation.
DEFAULT_BASELINE_THRESHOLD = 0.5
DEFAULT_SIAMESE_THRESHOLD = 0.345


def load_model(model_name: str, device: torch.device):
    """Load the selected model weights and return a ready-to-use PyTorch model."""
    print(f'Loading {model_name} model on {device}...')
    if model_name == 'baseline':
        model = BaselineCNN()
        checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        print('Baseline model loaded successfully.')
        return model
    if model_name == 'siamese':
        model = SiameseNetwork()
        checkpoint = torch.load(SIAMESE_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        print('Siamese model loaded successfully.')
        return model
    raise ValueError(f'Unknown model: {model_name}')


def predict_baseline(model, image_path: str, threshold: float, device: torch.device):
    mean, std = load_stats()
    image = load_image(image_path, mean, std, transform=None).to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
        prob = torch.sigmoid(logits).item()
    prediction = 'genuine' if prob >= threshold else 'forged'
    return prob, prediction


def predict_siamese(model, image_a_path: str, image_b_path: str, threshold: float, device: torch.device):
    mean, std = load_stats()
    image_a = load_image(image_a_path, mean, std, transform=None).to(device).unsqueeze(0)
    image_b = load_image(image_b_path, mean, std, transform=None).to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(image_a, image_b)
        prob = torch.sigmoid(logits).item()
    prediction = 'match' if prob >= threshold else 'different'
    return prob, prediction


def build_explanation(model_name: str, model, image_path: str, image_a_path: str = None, image_b_path: str = None, output_dir: str = 'explanations', device: torch.device = torch.device('cpu')):
    os.makedirs(output_dir, exist_ok=True)
    if model_name == 'baseline':
        mean, std = load_stats()
        image_tensor = load_image(image_path, mean, std, transform=None).to(device)
        saliency = compute_input_saliency(model, image_tensor)
        overlay_path = os.path.join(output_dir, 'baseline_explanation_overlay.png')
        heatmap_path = os.path.join(output_dir, 'baseline_explanation_heatmap.png')
        save_saliency_overlay(image_path, saliency, overlay_path)
        save_raw_saliency(saliency, heatmap_path)
        return {'overlay': overlay_path, 'heatmap': heatmap_path}

    if model_name == 'siamese':
        mean, std = load_stats()
        image_a_tensor = load_image(image_a_path, mean, std, transform=None).to(device)
        image_b_tensor = load_image(image_b_path, mean, std, transform=None).to(device)
        saliency_a, saliency_b = compute_siamese_saliency(model, image_a_tensor, image_b_tensor)
        overlay_a = os.path.join(output_dir, 'siamese_explanation_A_overlay.png')
        overlay_b = os.path.join(output_dir, 'siamese_explanation_B_overlay.png')
        heatmap_a = os.path.join(output_dir, 'siamese_explanation_A_heatmap.png')
        heatmap_b = os.path.join(output_dir, 'siamese_explanation_B_heatmap.png')
        save_saliency_overlay(image_a_path, saliency_a, overlay_a)
        save_raw_saliency(saliency_a, heatmap_a)
        save_saliency_overlay(image_b_path, saliency_b, overlay_b)
        save_raw_saliency(saliency_b, heatmap_b)
        return {
            'overlay_a': overlay_a,
            'overlay_b': overlay_b,
            'heatmap_a': heatmap_a,
            'heatmap_b': heatmap_b,
        }

    raise ValueError(f'Unsupported explainability model: {model_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Signature verification CLI with explainability.')
    parser.add_argument('--model', choices=['baseline', 'siamese'], required=True,
                        help='Select inference model: baseline for single signatures, siamese for pair comparison.')
    parser.add_argument('--image', type=str,
                        help='Path to the single signature image for baseline mode.')
    parser.add_argument('--image-a', type=str,
                        help='Path to the first signature image for siamese comparison.')
    parser.add_argument('--image-b', type=str,
                        help='Path to the second signature image for siamese comparison.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold for probability output.')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explainability heatmaps for the prediction.')
    parser.add_argument('--output-dir', type=str, default='explanations',
                        help='Directory where explanation images are saved.')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device: cuda or cpu. Defaults to cuda if available.')
    return parser.parse_args()


def main():
    args = parse_args()
    print('Starting signature verification CLI...')
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')
    model = load_model(args.model, device)

    if args.model == 'baseline':
        if not args.image:
            raise ValueError('Baseline mode requires --image <path>.')
        print('Running baseline single-image inference...')
        threshold = args.threshold if args.threshold is not None else DEFAULT_BASELINE_THRESHOLD
        prob, prediction = predict_baseline(model, args.image, threshold, device)
        print("\n" + "="*55)
        print("🔍 SIGNATURE VERIFICATION ANALYSIS (BASELINE)")
        print("="*55)
        print(f"File analyzed   : {args.image}")
        print(f"Raw Probability : {prob:.4f} (ranges from 0 to 1)")
        print(f"Threshold       : {threshold:.3f}")
        print("-" * 55)
        
        confidence = prob * 100 if prediction == 'genuine' else (1 - prob) * 100
        print(f"💡 FINAL VERDICT : {prediction.upper()}")
        print(f"🧠 EXPLANATION   : The model is {confidence:.2f}% confident that the signature is {prediction}.")
        print(f"                 (A probability above {threshold:.3f} is considered genuine)")
        print("="*55 + "\n")
        if args.explain:
            print('Generating explainability outputs...')
            outputs = build_explanation('baseline', model, args.image, output_dir=args.output_dir, device=device)
            print('Explanation saved:')
            for name, path in outputs.items():
                print(f'  {name}: {path}')
        print('Baseline inference complete.')
        return

    if args.model == 'siamese':
        if not args.image_a or not args.image_b:
            raise ValueError('Siamese mode requires --image-a <path> and --image-b <path>.')
        print('Running siamese pair comparison inference...')
        threshold = args.threshold if args.threshold is not None else DEFAULT_SIAMESE_THRESHOLD
        prob, prediction = predict_siamese(model, args.image_a, args.image_b, threshold, device)
        print("\n" + "="*55)
        print("🔍 SIGNATURE COMPARISON ANALYSIS (SIAMESE)")
        print("="*55)
        print(f"Image A (Ref)   : {args.image_a}")
        print(f"Image B (Test)  : {args.image_b}")
        print(f"Similarity Score: {prob:.4f} (ranges from 0 to 1)")
        print(f"Threshold       : {threshold:.3f}")
        print("-" * 55)
        
        confidence = prob * 100 if prediction == 'match' else (1 - prob) * 100
        print(f"💡 FINAL VERDICT : {prediction.upper()}")
        print(f"🧠 EXPLANATION   : The model is {confidence:.2f}% confident that the signatures {prediction}.")
        print(f"                 (A score above {threshold:.3f} means they belong to the same person)")
        print("="*55 + "\n")
        if args.explain:
            print('Generating explainability outputs...')
            outputs = build_explanation('siamese', model, None, args.image_a, args.image_b,
                                        output_dir=args.output_dir, device=device)
            print('Explanation saved:')
            for name, path in outputs.items():
                print(f'  {name}: {path}')
        print('Siamese inference complete.')
        return

    raise ValueError(f'Unknown mode: {args.model}')


if __name__ == '__main__':
    main()
