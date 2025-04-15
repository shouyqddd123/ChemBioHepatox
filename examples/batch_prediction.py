"""
Batch hepatotoxicity prediction example.

This script demonstrates how to use the ChemBioHepatox framework
to predict hepatotoxicity for multiple compounds in a batch.
"""

import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.predict_hepatotoxicity import (
    predict_hepatotoxicity,
    get_assay_predictions,
    load_models
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
    """Run hepatotoxicity prediction for multiple compounds."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict hepatotoxicity for multiple compounds")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with SMILES column")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="Name of SMILES column in CSV")
    parser.add_argument("--model_path", type=str, default="../saved_model", help="Path to the pretrained model")
    parser.add_argument("--checkpoint_path", type=str, default="../best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="batch_predictions.csv", help="Output CSV file path")
    parser.add_argument("--include_assays", action="store_true", help="Include assay predictions in output")
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading input data from {args.input}...")
    data = pd.read_csv(args.input)
    
    if args.smiles_col not in data.columns:
        print(f"Error: SMILES column '{args.smiles_col}' not found in input file.")
        return
    
    # Load models
    print(f"Loading models from {args.model_path} and {args.checkpoint_path}...")
    tokenizer, model, classifier = load_models(args.model_path, args.checkpoint_path)
    
    # Initialize results dataframe
    results = data.copy()
    results['Hepatotoxicity_Probability'] = None
    
    # Add assay columns if requested
    if args.include_assays:
        for assay in ASSAY_NAMES:
            results[assay] = None
    
    # Process each compound
    print(f"Processing {len(data)} compounds...")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        try:
            smiles = row[args.smiles_col]
            
            # Skip invalid SMILES
            if pd.isna(smiles) or not isinstance(smiles, str):
                continue
                
            # Predict hepatotoxicity
            hepatotox_prob = predict_hepatotoxicity(smiles, tokenizer, model, classifier)
            results.at[i, 'Hepatotoxicity_Probability'] = hepatotox_prob
            
            # Get assay predictions if requested
            if args.include_assays:
                assay_predictions = get_assay_predictions(smiles, tokenizer, model, ASSAY_NAMES)
                for assay, prob in assay_predictions.items():
                    results.at[i, assay] = prob
        
        except Exception as e:
            print(f"Error processing compound {i}: {e}")
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Generate summary statistics
    valid_results = results[~results['Hepatotoxicity_Probability'].isna()]
    num_compounds = len(valid_results)
    high_risk = (valid_results['Hepatotoxicity_Probability'] >= 0.7).sum()
    moderate_risk = ((valid_results['Hepatotoxicity_Probability'] >= 0.3) & 
                     (valid_results['Hepatotoxicity_Probability'] < 0.7)).sum()
    low_risk = (valid_results['Hepatotoxicity_Probability'] < 0.3).sum()
    
    print("\nSummary:")
    print(f"Total compounds processed: {num_compounds}")
    print(f"High risk compounds (≥0.7): {high_risk} ({high_risk/num_compounds*100:.1f}%)")
    print(f"Moderate risk compounds (0.3-0.7): {moderate_risk} ({moderate_risk/num_compounds*100:.1f}%)")
    print(f"Low risk compounds (<0.3): {low_risk} ({low_risk/num_compounds*100:.1f}%)")
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(valid_results['Hepatotoxicity_Probability'], bins=20, alpha=0.7)
    plt.axvline(x=0.7, color='r', linestyle='--', label='High Risk Threshold (0.7)')
    plt.axvline(x=0.3, color='orange', linestyle='--', label='Moderate Risk Threshold (0.3)')
    plt.xlabel('Hepatotoxicity Probability')
    plt.ylabel('Number of Compounds')
    plt.title('Distribution of Hepatotoxicity Predictions')
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.splitext(args.output)[0] + "_distribution.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Distribution plot saved to {plot_path}")

if __name__ == "__main__":
    main()
