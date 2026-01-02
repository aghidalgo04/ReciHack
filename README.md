# ReciHack ♻️

**ReciHack** is a computer vision project designed to classify and detect different types of waste to facilitate recycling processes. By leveraging state-of-the-art deep learning models (such as YOLO and DINO), this project aims to automate the segregation of garbage into specific categories.

> **⚠️ Important Note**: This repository contains the source code, notebooks, and configuration files. Due to GitHub's file size limits, the **datasets** and **trained model weights** are not hosted directly here. Please refer to the Setup & Data Preparation section to learn how to prepare the environment.

## Overview

Proper waste segregation is crucial for environmental sustainability. ReciHack uses machine learning to identify waste items from images and classify them into actionable categories. The project integrates various datasets and utilizes advanced object detection techniques to recognize items ranging from organic food scraps to electronic waste.

## Waste Classes

The model is configured to detect and classify waste into the following specific categories:

* **Organic**: Food scraps, biodegradable waste.
* **Paper / Cardboard**: Clean paper, boxes, newspapers (must be clean/dry).
* **Restos (General Waste)**: Non-recyclable items, used masks, diapers, dirty paper/tissues.
* **Plastic / Packaging**: Tetra briks, metal cans, plastic bottles, food containers, coffee capsules.
* **Glass**: Glass bottles and jars.
* **Punto Limpio (Clean Point)**: Special waste including electronics, batteries, clothing, X-rays, used oil, cork, and medicines.

## Project Structure

```bash
ReciHack/
├── DINO/             # Source code and experiments related to DINO model architecture
├── data/             # Folder structure for datasets (requires population)
├── src/              # Main source code, notebooks, and training scripts
└── README.md         # Project documentation
```

## Dependencies & Installation

This project relies on **Python 3.8+**. Please install the necessary libraries directly using pip.

**Core Dependencies:**
* **Data & Vis**: `numpy`, `pandas`, `matplotlib`, `seaborn`
* **CV & AI**: `opencv-python`, `Pillow`, `torch`, `torchvision`, `ultralytics`
* **Utils**: `jupyter`, `notebook`, `tqdm`, `scikit-learn`

### Quick Install Command
```bash
pip install numpy pandas matplotlib seaborn opencv-python Pillow torch torchvision ultralytics jupyter notebook scikit-learn tqdm
```

## Grounding DINO Setup

If you plan to use the Grounding DINO model, specific installation steps are required.

**1. Clone and Install**
```bash
git clone [https://github.com/IDEA-Research/GroundingDINO.git](https://github.com/IDEA-Research/GroundingDINO.git)
cd GroundingDINO
pip install -e .
cd ..
```

**2. Get Model Weights**
Download the `groundingdino_swint_ogc.pth` file:
```bash
wget -q [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
```
*> **Note**: Windows users without `wget` can download the file manually from the link above or use PowerShell.*

**3. Create DINO Directories**
```bash
mkdir DINO
cd DINO
mkdir annotations
```

## Setup & Data Preparation

### 1. Why is data missing?
Git repositories are designed for code. The combined image datasets exceed several Gigabytes, and trained model weights often exceed GitHub's 100MB file limit.

### 2. Download & Organization
To use this project, you must populate the `data/` folder:

1.  **Download**: Visit the links in the [Data Sources](#-data-sources) section.
2.  **Organize**: Extract the images. The project expects a structure compatible with YOLO/DINO training formats (usually split into `train` and `val` directories).
    ```text
    data/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
    │   ├── images/
    │   └── labels/
    ```
3.  **Preprocessing**: Check the `src/` folder for Jupyter Notebooks designed to ingest and format these raw datasets.

## Data Sources

We utilize a compilation of open-source datasets. To reproduce our results, please download the data from the following sources:

* **Ultralytics YOLO**: [Evolution of Object Detection](https://www.ultralytics.com/es/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models)
* **TrashNet**: [Gary Thung's TrashNet](https://github.com/garythung/trashnet/tree/master/data)
* **Kaggle Datasets**:
    * [Garbage Classification (Mostafa Abla)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data)
    * [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
    * [Recycle Photo](https://www.kaggle.com/code/danielaco/recyclephoto/input?select=train)
    * [Glass Images](https://www.kaggle.com/datasets/nandinibagga/glass-images-for-waste-segregation/data)
    * [Recycling 2](https://www.kaggle.com/datasets/aminizahra/recycling2)
    * [Recycling Waste](https://www.kaggle.com/datasets/naidusurajvardhan/recycling-waste)
    * [IKEA Furniture](https://www.kaggle.com/datasets/brikwerk/ikeafs)
    * [Knife/Cutlery Detection](https://www.kaggle.com/datasets/vijaysingh888/knife-detection?resource=download)
* **Hugging Face**:
    * [Garbage Object Detection](https://huggingface.co/datasets/keremberke/garbage-object-detection)
    * [Trash Classification](https://huggingface.co/datasets/ethanwan/trash_classification)
* **Roboflow**:
    * [Clasificación Basura](https://universe.roboflow.com/pruebas-a7ivv/clasificacion-basura/dataset/1)

## Usage

Most of the analysis and training pipelines are contained within Jupyter Notebooks.

1.  Ensure your environment is active and dependencies are installed.
2.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
3.  Navigate to the `src/` folder.
4.  Open the relevant notebooks to explore data processing, model training, and evaluation.
