# k-Nearest Neighbors (k-NN) Image Classification

This repository contains an assignment for exploring the k-Nearest Neighbors (k-NN) algorithm for image classification, based on the Stanford CS231n course materials. The goal of this exercise is to implement and evaluate a k-NN classifier on the CIFAR-10 dataset.

## Project Structure

- `notebooks/knn.ipynb`: Jupyter notebook containing the main exercise, visualizations, and questions.
- `notebooks/cs231n/`: Supporting Python modules containing the classifier implementation and data utilities.
- `data/get_datasets.sh` / `data/get_datasets.ps1`: Scripts to download the CIFAR-10 dataset.
- `TAREFA.md`: Original assignment instructions.

## Requirements

- Python >= 3.13
- A package manager like `uv` (recommended) or `pip`

## Setup Instructions

1. **Install Dependencies**
   The project uses `uv` for dependency management. You can install the required packages (such as `numpy`, `matplotlib`, and `ipykernel`) by running:
   ```bash
   uv sync
   ```

2. **Download the Dataset**
   Before running the notebook, you need to download the CIFAR-10 dataset. You can do this directly from within the Jupyter notebook (`knn.ipynb`), or manually by running the download script:
   - **Windows:** Run `.\data\get_datasets.ps1` from a PowerShell console.
   - **Linux/macOS:** Run `bash data/get_datasets.sh`.

3. **Run the Notebook**
   Start your Jupyter Notebook environment and open `notebooks/knn.ipynb`:
   ```bash
   uv run jupyter notebook
   ```
   Follow the instructions inside the notebook to complete the implementations and answer the inline questions.

## Assignment Goals

- Understand the basic Image Classification pipeline and cross-validation.
- Gain proficiency in writing efficient, vectorized Python code using NumPy.
- Implement distance matrix computations (with two loops, one loop, and zero loops).
- Predict labels based on $k$ nearest neighbors.
- Analyze the performance differences between implementations.
