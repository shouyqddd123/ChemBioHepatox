# ChemBioHepatox Methodology

This document provides a detailed description of the ChemBioHepatox framework methodology, its components, and the scientific principles behind it.

## 1. Framework Overview

ChemBioHepatox is a novel multimodal deep-learning framework that integrates chemical structures with biological assay responses for accurate hepatotoxicity prediction. The framework consists of three key components:

1. **Hepatotoxicity Assay Response Spectrum**: A comprehensive panel of 19 key assays covering all 12 fundamental hepatotoxicity mechanisms.
2. **Multimodal Integration Architecture**: A deep learning approach that bridges chemical structures and biological assays.
3. **Interpretable Mechanism Analysis**: A linear output layer that quantifies feature contributions.

## 2. Component 1: Hepatotoxicity Assay Response Spectrum

### 2.1 Three-step Process for Assay Selection

Our assay selection follows a systematic three-step process:

1. **Drug-Assay Matrix Construction**: We constructed a drug-assay matrix (1,170 compounds × 1,510 assays) by extracting all available HTS data for DILIst compounds from PubChem.

2. **Statistical Assay Selection**: We employed two complementary statistical approaches:
   - **Chi-squared test**: Identified 50 assays with p < 0.05
   - **Weighted Partial Correlation (WPC)**: Identified 44 assays with WPC > 0.6
   
   The intersection of these methods yielded 38 statistically significant assays.

3. **Mechanism Mapping**: We refined the assay panel by eliminating redundancies and mapping to the 12 fundamental hepatotoxicity mechanisms established by Rusyn et al. This resulted in our final panel of 19 key assays.

### 2.2 Mathematical Formulation

For the Weighted Partial Correlation (WPC) method, we calculate:

$$w_i = \frac{n_i}{N}$$

Where $w_i$ is the ratio of valid data and total compounds, reflecting data completeness at each assay. $n_i$ is the number of non-missing values, and $N$ equals 1170, representing the total number of compounds.

The weighted correlation score is computed as:

$$\text{WPC}_i = -\log_{10}(p_i) \cdot w_i$$

Where $p_i$ is the Fisher's exact test p-value for association between assay $i$ and hepatotoxicity labels.

## 3. Component 2: Multimodal Integration Architecture

### 3.1 Chemical Structure Processing

We use the pre-trained ChemBERTa-77M-MTR model to generate 364-dimensional embeddings from the [CLS] token of compound SMILES representations.

### 3.2 Biological Assay Processing

For biological data, we implement:

1. **Masked Self-attention Mechanism**: Handles sparse data while preserving informative signals.
2. **Probabilistic Approach with Soft Labels**: Better captures the nuanced nature of biological responses.
3. **Dynamic α-focal Loss**: Addresses class imbalance with the formula:

$$L = -\sum_{i,j} m_{i,j} \cdot \alpha_{i,j} \cdot (1-p_{i,j})^\gamma \cdot \log(p_{i,j})$$

Where:
- $m_{i,j}$ represents the missing data mask
- $\alpha_{i,j}$ is dynamically computed based on class imbalance
- $\gamma$ modulates focus on difficult-to-predict samples
- $p_{i,j}$ is the prediction probability

### 3.3 Integration Approach

The trained model generates 19-dimensional assay probabilities that concatenate with 364-dimensional structure embeddings, creating 383-dimensional enhanced representations.

## 4. Component 3: Interpretable Mechanism Analysis

We implement a linear layer that transforms the 383-dimensional enhanced representations into hepatotoxicity probabilities through a simple weighted sum:

$$P(\text{Hepatotoxicity}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

Where:
- $\sigma$ is the sigmoid activation function
- $\mathbf{w}$ is the weight vector
- $\mathbf{x}$ is the enhanced representation
- $b$ is the bias term

This transparent design allows direct interpretation of how each assay and structural feature contributes to the prediction.

## 5. Experimental Validation

### 5.1 Activity Cliff Analysis

We defined activity cliffs as compound pairs with high structural similarity (MACCS fingerprint similarity > 0.9) but opposing hepatotoxicity labels. Our framework successfully differentiated 26.9% of these challenging cases.

### 5.2 Clinical Relevance Assessment

We validated clinical relevance using the LiverTox database, which employs a standardized severity grading system (Grades A-E). Our model showed strong correlation with clinical severity grades.

### 5.3 External Domain Validation

We applied our framework to screen:
- 401 agricultural pesticides
- 62,271 food additives

Experimental validation confirmed our model's predictions aligned with published toxicity reports and in vitro experiments.

## 6. Conclusion

The ChemBioHepatox framework represents a significant advancement in hepatotoxicity prediction by combining:
1. Comprehensive biological mechanistic coverage
2. Innovative deep learning architecture for multimodal integration
3. Interpretable prediction mechanism

This approach not only achieves superior predictive performance but also provides mechanistic insights into the biological processes underlying toxicity.
