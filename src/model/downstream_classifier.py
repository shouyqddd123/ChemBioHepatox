# src/model/downstream_classifier.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    roc_curve, 
    auc, 
    precision_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Set seed for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def generate_enhanced_embeddings(texts, tokenizer, model, device):
    """
    Generate enhanced embeddings using the pre-trained model.
    
    Parameters:
    -----------
    texts : list
        List of text strings to encode
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    model : transformers.PreTrainedModel
        Pre-trained model for generating embeddings
    device : torch.device
        Device to run the model on
        
    Returns:
    --------
    torch.Tensor
        Enhanced embeddings that combine [CLS] token embeddings with task predictions
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.hidden_states[-1][:, 0, :]
        predictions = torch.sigmoid(outputs.logits)
        enhanced_embeddings = torch.cat((embeddings, predictions), dim=1)
    return enhanced_embeddings

class TextDataset(Dataset):
    """
    Custom dataset for text classification with pre-computed embeddings.
    
    This dataset pre-computes embeddings for efficiency rather than 
    generating them on-the-fly, which improves training speed.
    """
    def __init__(self, texts, labels, tokenizer, model, device):
        """
        Initialize the dataset with texts and labels.
        
        Parameters:
        -----------
        texts : list
            List of text strings
        labels : list
            List of corresponding labels (0 or 1)
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer for the model
        model : transformers.PreTrainedModel
            Pre-trained model for generating embeddings
        device : torch.device
            Device to run the model on
        """
        self.texts = texts
        self.labels = labels
        self.embeddings = []
        
        # Pre-compute all embeddings in batches
        batch_size = 32
        model.eval()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = generate_enhanced_embeddings(batch_texts, tokenizer, model, device)
            self.embeddings.append(batch_embeddings)
            
            # Log progress for large datasets
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(texts):
                logger.info(f"Pre-computed embeddings for {min(i + batch_size, len(texts))}/{len(texts)} samples")
                
        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SimpleClassifier(nn.Module):
    """
    Simple linear classifier for interpretable predictions.
    
    This single linear layer enables direct interpretation of feature importance
    through the examination of the learned weights.
    """
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_feature_importance(self):
        """
        Extract feature importance weights from the linear layer.
        
        Returns:
        --------
        torch.Tensor
            Weights of the linear layer, indicating feature importance
        """
        return self.linear.weight.data.squeeze()

def train_downstream_model(
    data_path, 
    model_path, 
    output_dir, 
    num_epochs=50, 
    batch_size=64, 
    learning_rate=0.00026, 
    test_size=0.1, 
    seed=3407
):
    """
    Train a downstream classification model using enhanced embeddings.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing SMILES and labels
    model_path : str
        Path to the pre-trained multitask model
    output_dir : str
        Directory to save the trained model and results
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimization
    test_size : float
        Proportion of data to use for testing
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (classifier, best_test_auc, feature_importance) model and performance metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data_df = pd.read_csv(data_path)
    
    # Split data into training and test sets
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=seed)
    logger.info(f"Data split: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Load pre-trained model and tokenizer
    logger.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)
    model.eval()  # Set model to evaluation mode for embedding generation
    
    # Calculate input dimension for classifier (hidden size + number of assay predictions)
    input_dim = config.hidden_size + config.num_labels
    logger.info(f"Classifier input dimension: {input_dim}")
    
    # Initialize classifier
    classifier = SimpleClassifier(input_dim).to(device)
    
    # Create datasets and data loaders
    logger.info("Creating datasets and dataloaders")
    train_dataset = TextDataset(
        train_df['compound'].tolist(), 
        train_df['label'].tolist(), 
        tokenizer, 
        model, 
        device
    )
    test_dataset = TextDataset(
        test_df['compound'].tolist(), 
        test_df['label'].tolist(), 
        tokenizer, 
        model, 
        device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam([
        {'params': classifier.parameters(), 'lr': learning_rate}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Use GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Track best model
    best_test_auc = 0.0
    best_epoch = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_model.pth")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0.0
        
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = classifier(embeddings).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        
        # Evaluation phase
        classifier.eval()
        test_outputs = []
        test_labels = []
        test_predictions = []
        test_loss = 0.0
        
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = classifier(embeddings).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Store outputs and labels for metrics calculation
                predictions = torch.sigmoid(outputs) > 0.5
                test_outputs.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(predictions.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(test_labels, test_outputs)
        test_auc = auc(fpr, tpr)
        test_precision = precision_score(test_labels, test_predictions)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Test Loss: {test_loss:.4f}, "
                   f"Test AUC: {test_auc:.4f}, "
                   f"Test Precision: {test_precision:.4f}")
        
        # Save best model
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch + 1
            
            # Save the model
            torch.save({
                'classifier_state_dict': classifier.state_dict(),
                'test_auc': best_test_auc,
                'epoch': best_epoch
            }, model_save_path)
            
            logger.info(f"New best model saved with AUC: {best_test_auc:.4f}")
    
    logger.info(f"Training completed. Best Test AUC: {best_test_auc:.4f} at epoch {best_epoch}")
    
    # Load the best model for final evaluation
    checkpoint = torch.load(model_save_path)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Final evaluation
    classifier.eval()
    all_test_labels = []
    all_test_predictions = []
    all_test_outputs = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = classifier(embeddings).squeeze()
            predictions = torch.sigmoid(outputs) > 0.5
            
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_outputs.extend(outputs.cpu().numpy())
    
    # Calculate final metrics
    accuracy = accuracy_score(all_test_labels, all_test_predictions)
    conf_matrix = confusion_matrix(all_test_labels, all_test_predictions)
    class_report = classification_report(all_test_labels, all_test_predictions, 
                                         target_names=['Non-hepatotoxic', 'Hepatotoxic'])
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_outputs)
    roc_auc = auc(fpr, tpr)
    
    # Log final results
    logger.info(f"\nFinal Test Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"\nClassification Report:\n{class_report}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Save results to file
    results_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        f.write(f"Classification Report:\n{class_report}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
    
    # Extract feature importance from the linear layer
    feature_importance = classifier.get_feature_importance().cpu().numpy()
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'Feature_Index': range(len(feature_importance)),
        'Importance': feature_importance
    })
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    
    # Create visualization of feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(feature_importance)), 
        feature_importance, 
        color=['r' if x > 0 else 'b' for x in feature_importance]
    )
    plt.xlabel('Feature Index')
    plt.ylabel('Weight (Importance)')
    plt.title('Feature Importance from Linear Layer Weights')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    
    return classifier, best_test_auc, feature_importance

# Example usage
if __name__ == "__main__":
    data_path = "../../data/processed/downstream_data.csv"
    model_path = "../../models/multitask"
    output_dir = "../../models/downstream"
    
    classifier, auc, importance = train_downstream_model(
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        num_epochs=50,
        batch_size=64,
        learning_rate=0.00026
    )
