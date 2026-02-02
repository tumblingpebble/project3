# DeepShell 2.0 / Project3 — Paper + Code (Preprint)

**DOI (Zenodo):** https://doi.org/10.5281/zenodo.18458799

This repository contains the paper (preprint) and accompanying code for **DeepShell 2.0**, an unsupervised learning pipeline that leverages pretrained vision representations (CLIP and DINOv2), an autoencoder for dimensionality reduction, and Gaussian Mixture Models (GMM) for clustering and evaluation.

---

## Paper

- **PDF:** `paper/NoLabelsNoProblem.pdf`
- **LaTeX source (Overleaf export):** `paper/source.zip`

> If you are viewing this on GitHub, you can open the PDF directly from the `paper/` folder.

---

## Summary

DeepShell 2.0 explores unsupervised clustering using representation learning and probabilistic clustering:
- **Representations:** CLIP ViT-L/14 and DINOv2 features
- **Dimensionality reduction:** autoencoder latent space
- **Clustering:** Gaussian Mixture Models (Expectation–Maximization)
- **Evaluation metrics:** Silhouette, NMI, ARI, clustering accuracy, and related diagnostics

---

## Repository Contents

- `precompute_representations.py` — feature extraction / representation precomputation  
- `precompute_labels.py` — label preprocessing for evaluation (when applicable)  
- `run_deepshell2.2.py` — main pipeline script  
- `utils.py` — shared utilities  
- `data/results/` — logs, plots, and metrics output artifacts  
- `paper/` — manuscript PDF and LaTeX source package  

---

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
If you want CUDA-enabled PyTorch, install PyTorch first (pick the correct command for your CUDA version), then install the remaining requirements.

Usage (Typical Workflow)
Precompute representations

python precompute_representations.py --dataset mnist --phis clipvitL14
python precompute_representations.py --dataset mnist --phis dinov2
Precompute labels (for evaluation)

python precompute_labels.py --dataset mnist
Run the pipeline

python run_deepshell2.2.py --use_gpu
Outputs (metrics, logs, plots) are written under:

data/results/

How to Cite
If you use this repository or the paper, cite the Zenodo archive:

DOI: https://doi.org/10.5281/zenodo.18458799

You can also use GitHub’s “Cite this repository” button if CITATION.cff is present.

License
Code: MIT License (see LICENSE if included)

Paper: Please see the paper text and/or repository notes for reuse terms

Acknowledgments
Developed as part of CSE 5160: Machine Learning at California State University, San Bernardino.
