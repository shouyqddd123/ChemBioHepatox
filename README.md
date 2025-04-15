# ChemBioHepatox

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-Coming%20Soon-orange)](http://exposomex.cn:58080/)

## Multimodal Integration of Chemical Structure and Biological Fingerprint for Robust and Interpretable Hepatotoxicity Prediction

ChemBioHepatox is a novel deep learning framework that combines chemical structures with biological assay responses to accurately predict hepatotoxicity while providing mechanistic insights into toxicity pathways.


## Overview

Hepatotoxicity evaluation is vital in drug development and chemical safety assessment. Our framework addresses three key challenges in computational toxicity prediction:

1. **Activity Cliffs**: Successfully predicting toxicity for structurally similar compounds with different biological effects
2. **Interpretability**: Providing mechanistic insights into predictions
3. **Broad Applicability**: Working across diverse chemical domains (pharmaceuticals, pesticides, food additives)

ChemBioHepatox achieves superior predictive performance (AUC 0.92, precision 0.88, recall 0.87) while maintaining mechanistic interpretability through its innovative three-component architecture.

## Key Components

### 1. Hepatotoxicity Assay Response Spectrum
A comprehensive panel of 19 key assays covering all 12 fundamental hepatotoxicity mechanisms, established through a rigorous three-step selection process.

### 2. Multimodal Integration Architecture
A deep learning approach that bridges chemical structures and biological assays within a shared embedding space, employing:
- Masked processing for handling missing values
- Dynamic α-focal loss for addressing class imbalance
- Probabilistic soft label method for multi-task learning

### 3. Interpretable Mechanism Analysis
A linear output layer that quantifies each feature's contribution to the prediction, providing transparent insight into the biological mechanisms underlying toxicity.

## Results

ChemBioHepatox demonstrates exceptional performance across multiple domains:

- **Predictive Accuracy**: AUC of 0.92, precision 0.88, recall 0.87
- **Activity Cliff Resolution**: Successfully classified 26.9% of structurally similar compounds with different toxicity profiles
- **Clinical Relevance**: Predictions correlate with clinical severity grades in the LiverTox database
- **Broad Applicability**: Successfully validated across pharmaceuticals, agricultural pesticides, and food additives

## Installation

```bash
# Clone the repository
git clone https://github.com/shouyqddd123/ChemBioHepatox.git
cd ChemBioHepatox

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
