# ReciHack ♻️

**ReciHack** is a computer vision project focused on the classification and detection of urban waste to support and automate recycling processes. By leveraging state-of-the-art deep learning models—such as **YOLO** and **DINO**—the project aims to identify waste items from images and assign them to well-defined recycling categories.

> **Note**  
> This repository contains source code, notebooks, and configuration files only. Due to GitHub file size limits, **datasets** and **trained model weights** are not included. Please follow the instructions in the [Getting Started & Setup](#-getting-started--setup) section to prepare the environment and obtain the required data.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Waste Classes](#-waste-classes)
- [Why Are the Datasets and Models Missing?](#-why-are-the-datasets-and-models-missing)
- [Project Structure](#-project-structure)
- [Getting Started & Setup](#-getting-started--setup)
- [Data Sources](#-data-sources)
- [License](#-license)

---

## 🧐 Overview

Proper waste segregation is a key factor in environmental sustainability and efficient recycling systems. **ReciHack** explores how machine learning and computer vision techniques can be applied to automatically recognize and classify waste items from images.

The project integrates multiple public datasets and experiments with modern object detection and classification approaches to handle a wide range of waste types—from organic food scraps to electronic waste. The repository is designed as a **research and prototyping environment**, emphasizing reproducibility, clarity, and extensibility rather than deployment as a production system.

---

## 🗑️ Waste Classes

The models and experiments in this project are configured to detect and classify waste into the following categories:

- **Organic**  
  Food scraps and biodegradable waste.

- **Paper / Cardboard**  
  Clean and dry paper products such as boxes, newspapers, and cardboard.

- **Restos (General Waste)**  
  Non-recyclable or contaminated items, including used masks, diapers, and dirty paper or tissues.

- **Plastic / Packaging**  
  Tetra briks, metal cans, plastic bottles, food containers, and coffee capsules.

- **Glass**  
  Glass bottles and jars.

- **Punto Limpio (Clean Point)**  
  Special waste streams such as electronics, batteries, clothing, X-rays, used oil, cork, and medicines.

---

## ⚠️ Why Are the Datasets and Models Missing?

Git repositories are optimized for source code version control, not for large binary assets.

1. **Datasets**  
   The combined image datasets used in this project span several gigabytes.

2. **Models**  
   Trained model weights (e.g. `.pt` or `.pth` files) often exceed GitHub’s 100 MB per-file limit.

As a result, datasets and trained models are intentionally excluded from the repository. To use or reproduce the experiments, you must download the datasets from the sources listed below and train the models locally (or obtain pretrained weights from external links when available).

---

## 📂 Project Structure

```
ReciHack/
├── DINO/             # Source code and experiments related to the DINO model architecture
├── data/             # Dataset directory (empty by default; populated locally)
├── src/              # Main source code, Jupyter notebooks, and training scripts
├── requirements.md   # List of dependencies and dataset references
└── README.md         # Project documentation
```

---

## 🛠️ Getting Started & Setup

### 1. Prerequisites

- Python 3.8 or newer  
- A virtual environment is strongly recommended  

```bash
git clone https://github.com/aghidalgo04/ReciHack.git
cd ReciHack

python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.md
```

> **Note**  
> If you plan to use YOLO-based models, ensure that the `ultralytics` package is installed and properly configured.

### 3. Data Preparation

```text
data/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Use the notebooks in `src/` to preprocess and merge datasets.

### 4. Training

```bash
jupyter notebook
```

Open the training notebooks in `src/` and execute them sequentially.

---

## 📊 Data Sources

   - https://www.ultralytics.com/es/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models 
   - https://github.com/garythung/trashnet/tree/master/data
   - https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data
   - https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
   - https://huggingface.co/datasets/keremberke/garbage-object-detection
   - https://huggingface.co/datasets/ethanwan/trash_classification
   - https://www.kaggle.com/code/danielaco/recyclephoto/input?select=train
   - https://universe.roboflow.com/pruebas-a7ivv/clasificacion-basura/dataset/1
   - https://www.kaggle.com/datasets/nandinibagga/glass-images-for-waste-segregation/data
   - https://www.kaggle.com/datasets/aminizahra/recycling2
   - https://www.kaggle.com/datasets/naidusurajvardhan/recycling-waste
   - https://www.kaggle.com/datasets/yashkangale20/garbage-classification?resource=download
   - https://www.kaggle.com/datasets/brikwerk/ikeafs
   - https://www.kaggle.com/datasets/vijaysingh888/knife-detection?resource=download
   - https://www.kaggle.com/datasets/shank885/knife-dataset
   - https://www.kaggle.com/datasets/rohitganji13/kitchen-cutlery?resource=download
