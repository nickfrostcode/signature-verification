<div align="center">
  <h1>🖋️ Signature Verification Pipeline</h1>
  <p><i>An end-to-end PyTorch machine learning pipeline for detecting forged signatures.</i></p>
</div>

---

## 📖 Overview

This repository provides a complete machine learning pipeline to verify handwritten signatures. It helps you preprocess raw signature scans, train different neural network architectures (**Baseline CNN** and **Siamese Networks**), evaluate their performance, and run inference with visual explainability outputs so you can see *exactly why* the model made its decision.

---

## ☁️ Google Colab & Datasets

If you prefer to run the training and experiments in the cloud, you can use the provided Google Colab notebook. The datasets are hosted on Google Drive.

- 📓 **[Open in Google Colab](https://colab.research.google.com/drive/1-cX1dBbmkpQriQod3UHSPiwg7rrAe4_U?usp=sharing)** — Run the training pipeline directly in your browser.
- 📁 **[Google Drive Dataset Folder](https://drive.google.com/drive/folders/1hNfeATb-EoUGEViQSKAvbN8zMx6v7ke_?usp=drive_link)** — Contains the necessary data for this project:
  1. **Raw Data Folder:** The raw signature image files as originally collected.
  2. **Normalized Data (`.zip`):** The dataset with normalized file extensions and properly formatted folder structures.
  3. **Processed Data (`.zip`):** The cleaned, standardized data ready for immediate model training and inference.

> [!TIP]
> To use the data locally without running the preprocessing pipeline, simply download the **Processed Data** zip and extract it into a `data_processed/` folder in the root of this project.

---

## ⚙️ Architecture and Pipeline Process

Understanding the codebase is easiest when you follow the data pipeline from raw images to final predictions:

### 1. Data Ingestion & Preprocessing (`preprocessing/`)

- **Raw Data**: Placed in the `data/` directory.
- **Normalization**: Scripts like `fix_extensions.py` and `fix_structure.py` ensure all files have the correct extensions and reside in the expected `SIG_XXXX/genuine` and `SIG_XXXX/forged` folder layout.
- **Validation**: `audit.py` checks for inconsistencies in the dataset.
- **Cleaning**: `preprocess.py` handles the heavy lifting—resizing, thresholding, and standardizing the images, saving the results to `data_processed/` for the models to consume.

### 2. Model Architectures (`models/`)

The project explores two distinct approaches to signature verification:

- **Baseline CNN (`models/baseline_cnn.py`)**: A straightforward Convolutional Neural Network that looks at a single signature and classifies it directly as *genuine* or *forged*.
- **Siamese Network (`models/siamese.py`)**: A more advanced pairwise model. It takes two signature images (e.g., a known genuine reference and a test sample), passes both through identical sub-networks to extract feature embeddings, and computes a similarity score. This is highly effective for learning what makes a specific person's signature unique.

### 3. Training & Evaluation (`training/`)

- **Training**: `train_baseline.py` and `train_siamese.py` manage the PyTorch training loops, loss calculations, and saving the final model weights to the `saved_models/` directory.
- **Evaluation**: `evaluate_models.py` runs test sets against the trained models to compute metrics like Accuracy, Precision, Recall, and F1-score, helping you choose the best threshold.

### 4. Inference & Explainability (`inference.py` & `explainability/`)

- **CLI (`inference.py`)**: Provides a user-friendly command-line interface to test new signatures against either model. It outputs human-readable verdicts and confidence percentages.
- **Explainability (`explainability/saliency.py`)**: To prevent the models from being "black boxes", the project includes saliency mapping. This generates visual heatmaps highlighting exactly which pen strokes or features the model focused on when making its decision.

---

## 🚀 Quick Start (Local Setup)

1. Clone the repository and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Download the dataset from the [Google Drive Link](https://drive.google.com/drive/folders/1hNfeATb-EoUGEViQSKAvbN8zMx6v7ke_?usp=drive_link) and place the contents into the `data/` directory.
3. Verify the dataset structure and run preprocessing if needed:

```bash
python preprocessing/audit.py
python preprocessing/preprocess.py
```

---

## 🖥️ Web Interface

If you prefer a graphical user interface instead of the command line, this project includes a beautiful, responsive React + FastAPI web application. 

You can upload signatures directly from your computer or phone, adjust sensitivity thresholds, and view the forensic heatmaps interactively.

👉 **[Click here to read the Web App Quick Start Guide](app/README.md)** to get the interface running locally.

---

## 🔍 How to use the CLI

### Baseline Model Inference

Run a single image through the baseline classifier to determine if it is genuine or forged:

```bash
python inference.py --model baseline --image 001.jpg --explain
```

*This will load the model, evaluate the image, print a human-readable verdict, and generate an explanation heatmap in the `explanations/` folder.*

### Siamese Model Inference

Compare a pair of signatures to see if they belong to the same person:

```bash
python inference.py --model siamese --image-a 001.jpg --image-b 002.jpg --explain
```

*This will extract features from both images, compute a similarity score against an optimal threshold, and generate explanation heatmaps for both signatures.*

---

## 🛠️ Troubleshooting

- If `inference.py` cannot find `models/` or `data/`, make sure you are running commands from the repository root.
- If the dataset layout is wrong, `preprocessing/audit.py` will describe what needs fixing.
- If you encounter CUDA out-of-memory errors, you can force the CLI to use CPU by appending `--device cpu` to your commands.
