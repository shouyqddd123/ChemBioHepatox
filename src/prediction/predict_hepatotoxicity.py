# src/prediction/predict_hepatotoxicity.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleClassifier(nn.Module):
    """
    Simple linear classifier for interpretable predictions.
    """
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def load_models(model_path, classifier_path):
    """
    Load the pre-trained model and classifier.
    
    Parameters:
    -----------
    model_path : str
        Path to the pre-trained model directory
    classifier_path : str
        Path to the saved classifier checkpoint
        
    Returns:
    --------
    tuple
        (tokenizer, model, classifier) loaded model components
    """
    # Load tokenizer and model configuration
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    # Create classifier with correct input dimension
    classifier = SimpleClassifier(config.hidden_size + config.num_labels)
    
    # Load saved weights
    checkpoint = torch.load(classifier_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Set models to evaluation mode
    model.eval()
    classifier.eval()
    
    return tokenizer, model, classifier

def generate_enhanced_embeddings(smiles, tokenizer, model):
    """
    Generate enhanced embeddings by combining structure embeddings with assay predictions.
    
    Parameters:
    -----------
    smiles : str
        SMILES string representation of a chemical compound
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    model : transformers.PreTrainedModel
        Pre-trained model for generating embeddings
        
    Returns:
    --------
    tuple
        (enhanced_embeddings, assay_predictions) generated embeddings and predictions
    """
    # Tokenize input
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract [CLS] token embeddings from the last hidden layer
        embeddings = outputs.hidden_states[-1][:, 0, :]
        
        # Get assay prediction probabilities
        predictions = torch.sigmoid(outputs.logits)
        
        # Concatenate embeddings and prediction probabilities
        enhanced_embeddings = torch.cat((embeddings, predictions), dim=-1)
    
    return enhanced_embeddings, predictions

def predict_hepatotoxicity(smiles, tokenizer, model, classifier):
    """
    Predict the hepatotoxicity probability of a chemical compound.
    
    Parameters:
    -----------
    smiles : str
        SMILES string representation of a chemical compound
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    model : transformers.PreTrainedModel
        Pre-trained model for generating embeddings
    classifier : torch.nn.Module
        Linear classifier for hepatotoxicity prediction
        
    Returns:
    --------
    float
        Predicted probability of hepatotoxicity
    """
    # Generate enhanced embeddings
    enhanced_embeddings, _ = generate_enhanced_embeddings(smiles, tokenizer, model)
    
    # Get hepatotoxicity prediction
    with torch.no_grad():
        classifier_predictions = classifier(enhanced_embeddings)
        final_prediction = torch.sigmoid(classifier_predictions).item()
    
    return round(final_prediction, 2)

def get_assay_predictions(smiles, tokenizer, model, assay_names):
    """
    Get prediction probabilities for all assays.
    
    Parameters:
    -----------
    smiles : str
        SMILES string representation of a chemical compound
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    model : transformers.PreTrainedModel
        Pre-trained model for generating embeddings
    assay_names : list
        List of assay names
        
    Returns:
    --------
    dict
        Dictionary mapping assay names to prediction probabilities
    """
    # Tokenize input
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).numpy().flatten()
    
    # Create dictionary of assay predictions
    predicted_probabilities = {assay_names[i]: float(predictions[i]) for i in range(len(assay_names))}
    
    return predicted_probabilities

def visualize_assay_predictions(predictions, output_path=None):
    """
    Create a visualization of assay prediction probabilities.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary mapping assay names to prediction probabilities
    output_path : str, optional
        Path to save the visualization, if None, the plot is displayed
        
    Returns:
    --------
    None
    """
    # Sort predictions by value
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    assays, values = zip(*sorted_predictions)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(assays)), values, color=['r' if v > 0.5 else 'b' for v in values])
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(max(v + 0.02, 0.1), i, f"{v:.2f}", va='center')
    
    # Configure plot
    plt.yticks(range(len(assays)), assays)
    plt.xlabel('Probability')
    plt.title('Assay Response Prediction Probabilities')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    """
    Main function to run the prediction pipeline from command line arguments.
    """
    parser = argparse.ArgumentParser(description="Predict hepatotoxicity of a chemical compound")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the compound")
    parser.add_argument("--model_path", type=str, default="saved_model", help="Path to the pre-trained model directory")
    parser.add_argument("--classifier_path", type=str, default="best_model.pth", help="Path to the saved classifier checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of assay predictions")
    args = parser.parse_args()
    
    # Define assay names
    assay_names = [
        "Caspase-3/7 HepG2 qHTS", "CYP1A2 Antag qHTS", "CYP2C19 Antag qHTS",
        "CYP2C9 Antag qHTS", "CYP3A4 Antag Reporter qHTS", "CYP3A7 Antag Cell qHTS",
        "ARE Agon qHTS", "MMP qHTS", "ER Stress", "ER-beta Agon qHTS: Summary",
        "PPARg Agon qHTS: Summary", "RAR Agon qHTS", "ERR Antag qHTS", "GR Antag qHTS",
        "PPARd Antag qHTS", "PPARg Antag Summary qHTS", "TR Antag Summary qHTS", "MDR-1",
        "HPGD Inhib qHTS"
    ]
    
    # Create output directory if needed
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    tokenizer, model, classifier = load_models(args.model_path, args.classifier_path)
    
    # Get hepatotoxicity prediction
    hepatotox_prob = predict_hepatotoxicity(args.smiles, tokenizer, model, classifier)
    print(f"Predicted hepatotoxicity probability: {hepatotox_prob}")
    
    # Get and display assay predictions
    assay_predictions = get_assay_predictions(args.smiles, tokenizer, model, assay_names)
    print("\nAssay Response Predictions:")
    for assay, prob in assay_predictions.items():
        print(f"{assay}: {prob:.2f}")
    
    # Generate visualization if requested
    if args.visualize:
        output_path = os.path.join(args.output_dir, "assay_predictions.png")
        visualize_assay_predictions(assay_predictions, output_path)
        
        # Also save as CSV
        df = pd.DataFrame([assay_predictions])
        df.insert(0, 'SMILES', args.smiles)
        df.insert(1, 'Hepatotoxicity_Probability', hepatotox_prob)
        csv_path = os.path.join(args.output_dir, "prediction_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
