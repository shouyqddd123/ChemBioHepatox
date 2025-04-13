# ChemBioHepatox

ChemBioHepatox is a novel multimodal deep-learning framework that integrates chemical structures with biological assay responses for accurate hepatotoxicity prediction. This repository contains the implementation of the complete framework as described in our paper.

## Overview

ChemBioHepatox consists of three integrated components:
1. **Hepatotoxicity Assay Response Spectrum**: A comprehensive set of 19 key assays capturing fundamental hepatotoxicity mechanisms
2. **Multimodal Integration Architecture**: A system that bridges chemical structures and biological assays through shared embedding space
3. **Interpretable Mechanism Analysis**: A linear output layer that quantifies each assay's contribution to prediction

The framework achieves high predictive performance (AUC 0.92, precision 0.88, recall 0.87) and significantly mitigates activity cliffs, with robust validation across pharmaceuticals, agricultural pesticides, and food additives.

## Repository Structure

- `DeepChem/ChemBERTa-77M-MTR/`: Pre-trained ChemBERTa model used for chemical structure encoding
- `saved_model/`: Trained model checkpoints for the multi-task learning component
- `best_model.pth`: Best performing model weights for the classifier
- `ChemBioHepatox Component 2 Multimodal Integration.py`: Implementation of the multimodal integration architecture
- `ChemBioHepatox Component 3 Interpretable Mechanism Analysis.py`: Implementation of the linear interpretable layer
- `Prediction example.ipynb`: Jupyter notebook with usage examples
- `cti.csv`: Dataset for training the multi-task learning model
- `data3.csv`: Dataset for training the hepatotoxicity classifier

## Installation

```bash
# Clone the repository
git clone https://github.com/shouyqddd123/ChemBioHepatox.git
cd ChemBioHepatox

# Create a conda environment (recommended)
conda create -n chembiohepatox python=3.8
conda activate chembiohepatox

# Install required packages
pip install torch torchvision
pip install transformers
pip install pandas numpy scikit-learn matplotlib
```

## Usage

We provide example code for using ChemBioHepatox in the `Prediction example.ipynb` Jupyter notebook, which includes:

1. **Predicting Hepatotoxicity for New Compounds**: Code to predict the hepatotoxicity probability of any chemical compound provided as a SMILES string
2. **Visualizing Assay Responses**: Code to generate and visualize the biological assay response spectrum for mechanism interpretation

To use the models with your own data:

1. Open `Prediction example.ipynb` in Jupyter Notebook or Jupyter Lab
2. Replace the example SMILES strings with your own chemical structures
3. Run the notebook cells to get predictions and visualizations

The notebook provides detailed comments and explanations for each step of the prediction process.

## Training

To retrain the models, follow these steps:

1. **Multi-task Learning Model**:
   Run `ChemBioHepatox Component 2 Multimodal Integration.py` with the appropriate dataset. This will train the model to predict the 19 assay responses from chemical structures.

2. **Hepatotoxicity Classifier**:
   After training the multi-task model, run `ChemBioHepatox Component 3 Interpretable Mechanism Analysis.py` to train the linear classifier for hepatotoxicity prediction.

## Web Platform

The ChemBioHepatox framework is also available as a web platform at [http://exposomex.cn:58080/](http://exposomex.cn:58080/), where you can input SMILES structures and get hepatotoxicity predictions along with mechanistic insights.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China and SEU Innovation Capability Enhancement Plan for Doctoral Students (CXJH_SEU 25).
