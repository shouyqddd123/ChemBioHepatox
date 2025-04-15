# src/model/multitask_learning.py

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTaskDataset(Dataset):
    """
    Dataset class for multi-task learning with masked labels.
    
    This dataset handles multiple classification tasks simultaneously,
    with support for missing values through label masks.
    """
    def __init__(self, inputs, labels, label_masks):
        self.inputs = inputs
        self.labels = labels
        self.label_masks = label_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        item['label_masks'] = self.label_masks[idx]
        return item

def focal_loss_with_dynamic_alpha(outputs, labels, label_masks, gamma=2):
    """
    Implementation of Focal Loss with dynamic alpha adjustment based on class imbalance.
    
    Parameters:
    -----------
    outputs : torch.Tensor
        Model prediction logits
    labels : torch.Tensor
        Ground truth labels
    label_masks : torch.Tensor
        Masks indicating valid (non-missing) labels
    gamma : float
        Focusing parameter for difficult-to-classify examples
        
    Returns:
    --------
    torch.Tensor
        Calculated loss value
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    active_loss = labels != -1
    active_loss = active_loss & (label_masks > 0)  # Apply mask

    # Calculate the number of positive and negative samples
    num_positive = torch.sum(labels[active_loss] == 1, dim=0).float()
    num_negative = torch.sum(labels[active_loss] == 0, dim=0).float()

    # Dynamically calculate α value, higher α for fewer positive samples
    alpha = num_negative / (num_positive + num_negative + 1e-8)
    
    # Dynamic sample-level adjustment of alpha
    alpha_factor = labels[active_loss].float() * alpha + (1 - labels[active_loss].float()) * (1 - alpha)

    # Calculate basic cross-entropy loss
    losses = loss_fn(outputs[active_loss], labels[active_loss].float())

    # Calculate prediction probability p_t
    probas = torch.sigmoid(outputs[active_loss])

    # Adjust probability based on true labels
    pt = probas * labels[active_loss].float() + (1 - probas) * (1 - labels[active_loss].float())

    # Calculate Focal Loss term
    focal_weight = (1 - pt) ** gamma

    # Apply dynamically adjusted α value and focal weight
    focal_loss = alpha_factor * focal_weight * losses

    # Apply mask
    masked_losses = focal_loss * label_masks[active_loss]

    # Return average loss
    return masked_losses.sum() / label_masks.sum()

def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on the provided dataloader.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation data
    device : torch.device
        Device to run evaluation on
        
    Returns:
    --------
    tuple
        (average_loss, accuracy, auc) metrics
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_masks = batch['label_masks'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = focal_loss_with_dynamic_alpha(outputs, labels, label_masks)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()  # Get prediction probabilities
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate accuracy
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels[all_labels != -1], binary_preds[all_labels != -1])

    # Calculate AUC-ROC
    auc = roc_auc_score(all_labels[all_labels != -1], all_preds[all_labels != -1])

    return avg_loss, accuracy, auc

def train_multitask_model(data_path, model_path, output_dir, num_epochs=100, batch_size=128, learning_rate=1e-5, seed=42):
    """
    Train a multi-task model to predict assay responses from molecular structures.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing SMILES strings and assay data
    model_path : str
        Path to the pretrained language model
    output_dir : str
        Directory to save the trained model and results
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimization
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (model, tokenizer, enhanced_embeddings, predictions) trained model and outputs
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info(f"Loading data from {data_path}")
    multi_task_df = pd.read_csv(data_path)
    
    # Convert non-numeric data to NaN and extract labels
    multi_task_df.iloc[:, 1:] = multi_task_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    labels = multi_task_df.iloc[:, 1:].values
    
    # Create label masks and process labels
    label_masks = ~pd.isna(labels)
    labels = np.where(pd.isna(labels), -1, labels)
    labels = labels.astype(int)
    labels = torch.tensor(labels, dtype=torch.long)
    label_masks = torch.tensor(label_masks, dtype=torch.float)
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = labels.shape[1]
    config.output_hidden_states = True
    config.output_attentions = True
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    # Tokenize input data
    logger.info("Tokenizing input data")
    inputs = tokenizer(list(multi_task_df.iloc[:, 0]), padding=True, truncation=True, return_tensors="pt")
    
    # Split data into training and validation sets
    logger.info("Splitting data into training and validation sets")
    train_input_ids, test_input_ids, train_attention_mask, test_attention_mask, train_labels, test_labels, train_label_masks, test_label_masks = train_test_split(
        inputs['input_ids'], inputs['attention_mask'], labels, label_masks, test_size=0.2, random_state=seed
    )
    
    # Create dataset objects and data loaders
    train_inputs = {'input_ids': train_input_ids, 'attention_mask': train_attention_mask}
    test_inputs = {'input_ids': test_input_ids, 'attention_mask': test_attention_mask}
    
    train_dataset = MultiTaskDataset(train_inputs, train_labels, train_label_masks)
    test_dataset = MultiTaskDataset(test_inputs, test_labels, test_label_masks)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to device
    model.to(device)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=32)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    best_auc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training step
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            batch_label_masks = batch['label_masks'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = focal_loss_with_dynamic_alpha(outputs, batch_labels, batch_label_masks)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation step
        avg_val_loss, val_accuracy, val_auc = evaluate_model(model, test_dataloader, device)
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        # Save the best model based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
    
    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "multitask_model")
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    # Generate enhanced embeddings
    logger.info("Generating enhanced embeddings")
    model.eval()
    
    # Move inputs to device
    device_inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**device_inputs)
        embeddings = outputs.hidden_states[-1][:, 0, :]  # Get representation of [CLS] token
        predictions = torch.sigmoid(outputs.logits)  # Get prediction probability for each label
    
    # Concatenate embeddings and predictions
    enhanced_embeddings = torch.cat((embeddings, predictions), dim=1)
    
    # Save enhanced embeddings
    embeddings_path = os.path.join(output_dir, "enhanced_embeddings.pt")
    torch.save(enhanced_embeddings, embeddings_path)
    logger.info(f"Enhanced embeddings saved to {embeddings_path}")
    
    return model, tokenizer, enhanced_embeddings, predictions

# Example usage
if __name__ == "__main__":
    data_path = "../../data/processed/drug_assay_matrix.csv"
    model_path = "DeepChem/ChemBERTa-77M-MTR"
    output_dir = "../../models/multitask"
    
    model, tokenizer, embeddings, predictions = train_multitask_model(
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        num_epochs=100,
        batch_size=128,
        learning_rate=1e-5
    )
