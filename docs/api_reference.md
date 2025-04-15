# ChemBioHepatox API Reference

This document provides detailed information about the functions and classes in the ChemBioHepatox framework.

## Data Processing Module

### `assay_selection.py`

#### `chi_square_test(assay_results, labels)`
Calculates the statistical significance of correlation between an assay and hepatotoxicity using chi-square test.

**Parameters:**
- `assay_results` (pandas.Series): Series containing assay results
- `labels` (pandas.Series): Series containing binary hepatotoxicity labels (0 or 1)

**Returns:**
- float: p-value indicating the statistical significance

#### `weighted_partial_correlation(assay_results, labels)`
Calculates weighted partial correlation between assay results and hepatotoxicity labels.

**Parameters:**
- `assay_results` (pandas.Series): Series containing assay results
- `labels` (pandas.Series): Series containing binary hepatotoxicity labels (0 or 1)

**Returns:**
- float: Weighted correlation value between 0 and 1

#### `run_assay_selection(data_path, output_dir)`
Runs both assay selection methods and saves results.

**Parameters:**
- `data_path` (str): Path to the input Excel file
- `output_dir` (str): Directory to save output files

**Returns:**
- tuple: (chi_results, wpc_results) DataFrames containing results from both methods

## Model Module

### `multitask_learning.py`

#### `class MultiTaskDataset(Dataset)`
Dataset class for multi-task learning with masked labels.

**Methods:**
- `__init__(inputs, labels, label_masks)`: Initialize the dataset
- `__len__()`: Return the number of samples
- `__getitem__(idx)`: Get a sample by index

#### `focal_loss_with_dynamic_alpha(outputs, labels, label_masks, gamma=2)`
Implementation of Focal Loss with dynamic alpha adjustment based on class imbalance.

**Parameters:**
- `outputs` (torch.Tensor): Model prediction logits
- `labels` (torch.Tensor): Ground truth labels
- `label_masks` (torch.Tensor): Masks indicating valid (non-missing) labels
- `gamma` (float): Focusing parameter for difficult-to-classify examples

**Returns:**
- torch.Tensor: Calculated loss value

#### `train_multitask_model(data_path, model_path, output_dir, num_epochs=100, batch_size=128, learning_rate=1e-5, seed=42)`
Train a multi-task model to predict assay responses from molecular structures.

**Parameters:**
- `data_path` (str): Path to the CSV file containing SMILES strings and assay data
- `model_path` (str): Path to the pretrained language model
- `output_dir` (str): Directory to save the trained model and results
- `num_epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `learning_rate` (float): Learning rate for optimization
- `seed` (int): Random seed for reproducibility

**Returns:**
- tuple: (model, tokenizer, enhanced_embeddings, predictions)

### `downstream_classifier.py`

#### `class SimpleClassifier(nn.Module)`
Simple linear classifier for interpretable predictions.

**Methods:**
- `__init__(input_dim)`: Initialize the classifier
- `forward(x)`: Perform forward pass
- `get_feature_importance()`: Extract feature importance weights

#### `train_downstream_model(data_path, model_path, output_dir, num_epochs=50, batch_size=64, learning_rate=0.00026, test_size=0.1, seed=3407)`
Train a downstream classification model using enhanced embeddings.

**Parameters:**
- `data_path` (str): Path to the CSV file containing SMILES and labels
- `model_path` (str): Path to the pre-trained multitask model
- `output_dir` (str): Directory to save the trained model and results
- `num_epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `learning_rate` (float): Learning rate for optimization
- `test_size` (float): Proportion of data to use for testing
- `seed` (int): Random seed for reproducibility

**Returns:**
- tuple: (classifier, best_test_auc, feature_importance)

## Prediction Module

### `predict_hepatotoxicity.py`

#### `predict_hepatotoxicity(smiles, tokenizer, model, classifier)`
Predict the hepatotoxicity probability of a chemical compound.

**Parameters:**
- `smiles` (str): SMILES string representation of a chemical compound
- `tokenizer`: Tokenizer for the model
- `model`: Pre-trained model for generating embeddings
- `classifier`: Linear classifier for hepatotoxicity prediction

**Returns:**
- float: Predicted probability of hepatotoxicity

#### `get_assay_predictions(smiles, tokenizer, model, assay_names)`
Get prediction probabilities for all assays.

**Parameters:**
- `smiles` (str): SMILES string representation of a chemical compound
- `tokenizer`: Tokenizer for the model
- `model`: Pre-trained model for generating embeddings
- `assay_names` (list): List of assay names

**Returns:**
- dict: Dictionary mapping assay names to prediction probabilities

## Web Demo Module

### `app.py`

#### `app`
Flask application for web interface.

**Routes:**
- `/`: Home page
- `/predict`: API endpoint for prediction

## Utility Functions

Various utility functions for data processing, visualization, and evaluation can be found in the respective modules.
