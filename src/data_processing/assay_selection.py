# src/data_processing/assay_selection.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import os

def chi_square_test(assay_results, labels):
    """
    Calculate the chi-square test p-value for each assay.
    
    Parameters:
    -----------
    assay_results : pd.Series
        A series containing assay results
    labels : pd.Series
        A series containing labels (hepatotoxic/non-hepatotoxic)
        
    Returns:
    --------
    float
        Chi-square test p-value
    """
    valid_indices = ~assay_results.isna()
    if valid_indices.sum() == 0:
        return 1  # Return non-significant p-value if all values are missing
    table = pd.crosstab(assay_results[valid_indices], labels[valid_indices])
    if table.shape == (2, 2):
        _, p_value, _, _ = chi2_contingency(table)
    else:
        p_value = 1  # Return non-significant p-value if contingency table is not 2x2
    return p_value

def weighted_partial_correlation(assay_results, labels):
    """
    Calculate weighted partial correlation for assay and toxicity labels.
    
    Parameters:
    -----------
    assay_results : pd.Series
        A series containing assay results
    labels : pd.Series
        A series containing labels (hepatotoxic/non-hepatotoxic)
        
    Returns:
    --------
    float
        Weighted partial correlation value
    """
    valid_indices = ~assay_results.isna()
    if valid_indices.sum() == 0:
        return np.nan
    
    # Create 2x2 contingency table
    table = pd.crosstab(assay_results[valid_indices], labels[valid_indices])
    if table.shape == (2, 2):
        _, p_value = fisher_exact(table)
        # Local correlation
        partial_corr = -np.log10(p_value) if p_value > 0 else 0
    else:
        partial_corr = 0
    
    # Calculate weight (proportion of valid observations)
    weights = valid_indices.sum() / len(labels)
    
    # Weighted partial correlation
    weighted_corr = partial_corr * weights
    return weighted_corr

def run_assay_selection(data_path, output_dir):
    """
    Run both assay selection methods and save results.
    
    Parameters:
    -----------
    data_path : str
        Path to the input Excel file
    output_dir : str
        Directory to save output files
    """
    # Load data
    data = pd.read_excel(data_path, engine='openpyxl')
    
    # Extract features and labels
    X = data.iloc[:, 2:]  # Remaining columns are assays
    y = data.iloc[:, 1]   # Second column is the label
    
    # Method 1: Chi-square test
    chi_square_p_values = X.apply(lambda col: chi_square_test(col, y))
    chi_results = pd.DataFrame({
        'Assay': X.columns,
        'Chi-Square p-value': chi_square_p_values
    })
    
    # Method 2: Weighted partial correlation
    weighted_corr = X.apply(lambda col: weighted_partial_correlation(col, y))
    wpc_results = pd.DataFrame({
        'Assay': X.columns,
        'Weighted Partial Correlation': weighted_corr
    })
    
    # Limit correlation values between 0 and 1
    wpc_results['Weighted Partial Correlation'] = wpc_results['Weighted Partial Correlation'].apply(
        lambda x: min(max(x, 0), 1)
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    chi_results.to_csv(os.path.join(output_dir, "assay_chi_square_test.csv"), index=False)
    wpc_results.to_csv(os.path.join(output_dir, "assay_weighted_partial_correlation.csv"), index=False)
    
    return chi_results, wpc_results

# Example usage (can be commented out when importing as a module)
if __name__ == "__main__":
    data_path = "../../data/raw/assay_correlation.xlsx"
    output_dir = "../../data/processed/"
    chi_results, wpc_results = run_assay_selection(data_path, output_dir)
    
    # Display top 10 results for each method
    print("Top 10 assays by Chi-Square test:")
    print(chi_results.sort_values('Chi-Square p-value').head(10))
    
    print("\nTop 10 assays by Weighted Partial Correlation:")
    print(wpc_results.sort_values('Weighted Partial Correlation', ascending=False).head(10))
