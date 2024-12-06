# Project 3: DeepShell 2.0

## Overview
**DeepShell 2.0** is an advanced machine learning pipeline that transforms unsupervised clustering into classification for MNIST images. It leverages feature extraction, autoencoders, and Gaussian Mixture Models (GMMs) to achieve an impressive classification accuracy of **95.66%**.

## Features
- **Dataset:** MNIST handwritten digit dataset
- **Feature Extraction:** Pretrained representations from CLIP and DINOv2
- **Neural Network:** Autoencoder for dimensionality reduction
- **Clustering Algorithm:** Gaussian Mixture Models (GMMs)
- **Evaluation Metrics:** Silhouette Score, NMI, ARI, Clustering Accuracy, and more
- **Visualization:** t-SNE plots and correlation matrices for latent space analysis

## Project Objectives
1. Transform unsupervised clustering into a classification model.
2. Identify and classify clusters with high accuracy.
3. Leverage pretrained models and modular pipelines for flexibility and scalability.

## Installation
To set up the project environment, ensure you have Python 3.8+ installed. Use the provided `requirements.txt` to install all necessary dependencies:

pip install -r requirements.txt


> **Note:** For CUDA-enabled PyTorch installation, run the following command before installing other packages:
> ```
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

## Usage
1. **Precompute Representations and Ground Truth Labels**:
Run the following commands to precompute representations and ground truth labels:

> ```
> python precompute_representations.py --dataset mnist --phis clipvitL14
> ```
> ```
> python precompute_representations.py --dataset mnist --phis dinov2
> ```
> ```
> python precompute_labels.py --dataset mnist
> ``` 

2. **Prepare the Dataset**:
   - Ensure the MNIST dataset and precomputed feature representations are stored in the specified `data/representations` directory.
  Files should be mnist_train.npy and mnist_val.npy files in clipvitL14/ and dinov2/

2. **Run the Pipeline**:
   Execute the main script with default or custom arguments:

> ```
> python run_deepshell2.2.py --use_gpu
> ``` 


3. **Outputs**:
- Logs are saved in `data/results/run_deepshell2.1.log`.
- Visualizations and evaluation metrics are stored in the `data/results` directory.

## Key Results
- **Clustering Accuracy:** 95.66% achieved with 10 clusters.
- **Evaluation Metrics:**
- Silhouette Score: Indicates cluster cohesion.
- NMI (Normalized Mutual Information): Measures clustering consistency.
- ARI (Adjusted Rand Index): Evaluates cluster quality.

## Dependencies
The project uses the following libraries:
- **PyTorch**: For implementing the autoencoder.
- **scikit-learn**: For clustering and evaluation metrics.
- **Matplotlib & Seaborn**: For visualizations.
- **Numpy & Pandas**: For data processing.

See the complete list of dependencies in [requirements.txt](requirements.txt).

## File Structure
project3/ ├── data/ # Dataset and representations directory │ ├── representations/ # Precomputed feature files │ ├── labels/ # True labels for evaluation │ ├── results/ # Output results and logs ├── run_deepshell2.1.py # Main script ├── requirements.txt # Required Python libraries └── README.md # Project documentation


## Future Work
- Implement advanced neural network architectures, such as Variational Autoencoders (VAEs).
- Automate hyperparameter tuning and clustering method selection.
- Extend the pipeline to other datasets and tasks.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
This project was developed as part of **CSE 5160: Machine Learning** under the guidance of **Dr. Jennifer Jin** at **Cal State University San Bernardino**.
