"""
Single compound hepatotoxicity prediction example.

This script demonstrates how to use the ChemBioHepatox framework
to predict the hepatotoxicity of a single chemical compound.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import argparse

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.predict_hepatotoxicity import (
    predict_hepatotoxicity,
    get_assay_predictions,
    load_models,
    visualize_assay_predictions
)

# Define assay names
ASSAY_NAMES = [
    "Caspase-3/7 HepG2 qHTS", "CYP1A2 Antag qHTS", "CYP2C19 Antag qHTS",
    "CYP2C9 Antag qHTS", "CYP3A4 Antag Reporter qHTS", "CYP3A7 Antag Cell qHTS",
    "ARE Agon qHTS", "MMP qHTS", "ER Stress", "ER-beta Agon qHTS: Summary",
    "PPARg Agon qHTS: Summary", "RAR Agon qHTS", "ERR Antag qHTS", "GR Antag qHTS",
    "PPARd Antag qHTS", "PPARg Antag Summary qHTS", "TR Antag Summary qHTS", "MDR-1",
    "HPGD Inhib qHTS"
]

def main():
    """Run hepatotoxicity prediction for a single compound."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict hepatotoxicity for a single compound")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the compound")
    parser.add_argument("--model_path", type=str, default="../saved_model", help="Path to the pretrained model")
    parser.add_argument("--checkpoint_path", type=str, default="../best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print(f"Loading models from {args.model_path} and {args.checkpoint_path}...")
    tokenizer, model, classifier = load_models(args.model_path, args.checkpoint_path)
    
    # Predict hepatotoxicity
    print(f"Predicting hepatotoxicity for: {args.smiles}")
    hepatotox_prob = predict_hepatotoxicity(args.smiles, tokenizer, model, classifier)
    print(f"Hepatotoxicity probability: {hepatotox_prob:.2f}")
    
    # Get assay predictions
    print("Generating assay response profile...")
    assay_predictions = get_assay_predictions(args.smiles, tokenizer, model, ASSAY_NAMES)
    
    # Display assay predictions
    print("\nAssay Response Profile:")
    for assay, prob in sorted(assay_predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"{assay}: {prob:.2f}")
    
    # Create visualization
    output_path = os.path.join(args.output_dir, "assay_predictions.png")
    visualize_assay_predictions(assay_predictions, output_path)
    print(f"\nVisualization saved to {output_path}")
    
    # Save results to a text file
    results_path = os.path.join(args.output_dir, "prediction_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Hepatotoxicity Probability: {hepatotox_prob:.4f}\n\n")
        f.write("Assay Response Profile:\n")
        for assay, prob in sorted(assay_predictions.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{assay}: {prob:.4f}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
