import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set random seeds (consistent with training code)
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set font parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


def deduplicate_dataset(csv_file):
    """Remove duplicate molecules and return molecule objects with labels"""
    print("Loading and deduplicating dataset...")
    data = pd.read_csv(csv_file)
    smiles_list = data['compound'].tolist()
    labels_list = data['label'].tolist()

    print(f"Original dataset: {len(smiles_list)} SMILES")

    # Deduplication process
    seen_inchis = set()
    unique_molecules = []
    unique_labels = []
    unique_smiles = []  # Store corresponding SMILES for subsequent tokenization
    invalid_count = 0
    duplicate_count = 0

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                inchi = Chem.MolToInchi(mol)
                if inchi and inchi not in seen_inchis:
                    seen_inchis.add(inchi)
                    unique_molecules.append(mol)
                    unique_labels.append(labels_list[i])
                    # Use canonical SMILES for subsequent processing
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    unique_smiles.append(canonical_smiles)
                elif inchi in seen_inchis:
                    duplicate_count += 1
            except:
                invalid_count += 1
        else:
            invalid_count += 1

    print(f"Deduplication results:")
    print(f"  Original samples: {len(smiles_list)}")
    print(f"  Invalid SMILES: {invalid_count}")
    print(f"  Duplicates removed: {duplicate_count}")
    print(f"  Unique molecules: {len(unique_molecules)}")
    print(f"  Label distribution: {dict(zip(*np.unique(unique_labels, return_counts=True)))}")

    # Create DataFrame for data splitting
    deduplicated_df = pd.DataFrame({
        'compound': unique_smiles,
        'label': unique_labels
    })

    return deduplicated_df, unique_molecules, np.array(unique_labels)


def calculate_traditional_descriptors_from_mols(molecules):
    """Calculate traditional molecular descriptors from Mol objects (more efficient method)"""
    print("Calculating traditional molecular descriptors...")

    descriptors = {
        'ECFP': [],
        'MACCS': [],
        'Morgan': [],
        'RDKit': []
    }

    failed_count = 0

    for mol in molecules:
        try:
            if mol is not None:
                # ECFP (Morgan fingerprint as bit vector)
                ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                descriptors['ECFP'].append(np.array(ecfp))

                # MACCS keys
                maccs = MACCSkeys.GenMACCSKeys(mol)
                descriptors['MACCS'].append(np.array(maccs))

                # Morgan fingerprint (count-based)
                morgan = AllChem.GetMorganFingerprint(mol, radius=2)
                # Convert to fixed-length vector
                morgan_vec = np.zeros(1024)
                for key, value in morgan.GetNonzeroElements().items():
                    morgan_vec[key % 1024] += value
                descriptors['Morgan'].append(morgan_vec)

                # RDKit fingerprint
                rdkit_fp = RDKFingerprint(mol)
                descriptors['RDKit'].append(np.array(rdkit_fp))

            else:
                failed_count += 1
                # Add zero vectors
                descriptors['ECFP'].append(np.zeros(1024))
                descriptors['MACCS'].append(np.zeros(167))
                descriptors['Morgan'].append(np.zeros(1024))
                descriptors['RDKit'].append(np.zeros(2048))

        except Exception as e:
            print(f"Error processing molecule: {e}")
            failed_count += 1
            # Add zero vectors
            descriptors['ECFP'].append(np.zeros(1024))
            descriptors['MACCS'].append(np.zeros(167))
            descriptors['Morgan'].append(np.zeros(1024))
            descriptors['RDKit'].append(np.zeros(2048))

    # Convert to numpy arrays
    for key in descriptors:
        descriptors[key] = np.array(descriptors[key])

    if failed_count > 0:
        print(f"Number of molecules failed to process: {failed_count}")

    return descriptors


def load_best_model_and_generate_embeddings(smiles_list, model_path, best_model_path):
    """Load best model and generate enhanced representations"""
    print("Loading best model and generating enhanced representations...")

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)

    # Load best model weights (compatible with PyTorch 2.6+ and new numpy versions)
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load model with weights_only=False: {e}")
        # If still fails, try adding numpy safe global variables
        try:
            import torch.serialization
            # Use new numpy namespace
            try:
                import numpy._core.multiarray
                torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
            except ImportError:
                # Compatible with old numpy versions
                import numpy.core.multiarray
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

            checkpoint = torch.load(best_model_path, map_location=device)
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise RuntimeError("Unable to load model, please check PyTorch and numpy version compatibility")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Best model loaded (Test AUC: {checkpoint['test_auc']:.4f}, Epoch: {checkpoint['epoch']})")

    # Generate enhanced representations
    enhanced_embeddings = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]

            try:
                inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs)

                # Get CLS representations and prediction probabilities
                embeddings = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
                predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()

                # Concatenate enhanced representations
                batch_enhanced = np.concatenate([embeddings, predictions], axis=1)
                enhanced_embeddings.append(batch_enhanced)

                # Clear GPU memory
                del inputs, outputs
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                # Add zero vectors
                zero_embedding = np.zeros((len(batch_smiles), config.hidden_size + config.num_labels))
                enhanced_embeddings.append(zero_embedding)

    enhanced_embeddings = np.vstack(enhanced_embeddings)
    print(f"Enhanced representation dimensions: {enhanced_embeddings.shape}")

    return enhanced_embeddings


def prediction_uncertainty_confidence_assessment(train_data, test_data, train_embeddings, test_embeddings,
                                                 method_name, confidence_threshold=0.8, n_models=5):
    """Confidence assessment based on prediction uncertainty (using 0.8 threshold only)"""
    print(f"{method_name} - Performing prediction uncertainty assessment...")

    # Train multiple models
    predictions_all = []
    for i in range(n_models):
        model = RandomForestClassifier(n_estimators=100, random_state=seed + i, max_depth=10)
        model.fit(train_embeddings, train_data)
        pred_probs = model.predict_proba(test_embeddings)
        predictions_all.append(pred_probs[:, 1])

    predictions_all = np.array(predictions_all)
    mean_predictions = np.mean(predictions_all, axis=0)
    std_predictions = np.std(predictions_all, axis=0)

    # Calculate confidence
    prob_confidence = np.maximum(mean_predictions, 1 - mean_predictions)
    consistency_confidence = 1 - std_predictions
    combined_confidence = (prob_confidence + consistency_confidence) / 2

    # Calculate coverage rate (using 0.8 threshold only)
    combined_in_domain = combined_confidence >= confidence_threshold
    combined_coverage_rate = combined_in_domain.sum() / len(test_embeddings)

    result = {
        'coverage_rate': combined_coverage_rate,
        'covered_samples': combined_in_domain.sum(),
        'in_domain_mask': combined_in_domain,
        'confidence_scores': combined_confidence
    }

    print(f"{method_name} - Confidence threshold {confidence_threshold}: Coverage rate = {combined_coverage_rate:.1%}")

    return result


def uncertainty_based_applicability_domain_analysis():
    """Uncertainty-based applicability domain analysis (using 0.8 threshold only)"""

    # 1. Use your deduplication method
    print("=== Step 1: Data Deduplication ===")
    clean_df, unique_molecules, unique_labels = deduplicate_dataset('data3.csv')

    # 2. Data splitting (consistent with your training code)
    print("\n=== Step 2: Data Splitting ===")
    # Create indices for splitting
    indices = np.arange(len(unique_molecules))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=seed)

    # Split data
    train_df = clean_df.iloc[train_indices].reset_index(drop=True)
    test_df = clean_df.iloc[test_indices].reset_index(drop=True)
    train_molecules = [unique_molecules[i] for i in train_indices]
    test_molecules = [unique_molecules[i] for i in test_indices]
    train_labels = unique_labels[train_indices]
    test_labels = unique_labels[test_indices]

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # 3. Generate enhanced representations
    print("\n=== Step 3: Generate Enhanced Representations ===")
    model_path = "saved_model"
    best_model_path = "best_model.pth"

    train_enhanced = load_best_model_and_generate_embeddings(
        train_df['compound'].tolist(), model_path, best_model_path
    )
    test_enhanced = load_best_model_and_generate_embeddings(
        test_df['compound'].tolist(), model_path, best_model_path
    )

    # 4. Calculate traditional molecular descriptors from Mol objects (more efficient)
    print("\n=== Step 4: Calculate Traditional Molecular Descriptors ===")
    train_traditional = calculate_traditional_descriptors_from_mols(train_molecules)
    test_traditional = calculate_traditional_descriptors_from_mols(test_molecules)

    # 5. Uncertainty-based applicability domain coverage analysis (using 0.8 threshold only)
    print("\n=== Step 5: Uncertainty-based Applicability Domain Coverage Analysis ===")

    # Store all results
    uncertainty_results = {}

    # 5.1 Enhanced representation analysis
    print("\n--- Enhanced Representation Method ---")
    uncertainty_results['Enhanced'] = prediction_uncertainty_confidence_assessment(
        train_labels, test_labels, train_enhanced, test_enhanced, "Enhanced Representation"
    )

    # 5.2 Traditional method analysis
    traditional_methods = ['ECFP', 'MACCS', 'Morgan', 'RDKit']

    for method in traditional_methods:
        print(f"\n--- {method} Method ---")
        train_desc = train_traditional[method]
        test_desc = test_traditional[method]

        uncertainty_results[method] = prediction_uncertainty_confidence_assessment(
            train_labels, test_labels, train_desc, test_desc, method
        )

    # 6. Visualize results
    print("\n=== Step 6: Visualize Results ===")
    plot_uncertainty_coverage_comparison(uncertainty_results)

    # 7. Generate detailed report
    print("\n=== Step 7: Generate Analysis Report ===")
    generate_uncertainty_report(uncertainty_results)

    return uncertainty_results


def plot_uncertainty_coverage_comparison(uncertainty_results):
    """Plot uncertainty-based coverage comparison (using 0.8 threshold only)"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    methods = list(uncertainty_results.keys())
    colors = ['#5D99BE', '#EEC302', '#B7D9E4', '#E4BDCF', '#00656B']

    # 1. Coverage rate comparison (bar chart)
    coverage_rates = [uncertainty_results[method]['coverage_rate'] for method in methods]

    bars = axes[0, 0].bar(methods, coverage_rates, color=colors[:len(methods)])
    axes[0, 0].set_xlabel('Method')
    axes[0, 0].set_ylabel('Coverage Rate')
    axes[0, 0].set_title('Coverage Rate Comparison (Threshold = 0.8)')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3)

    # Add numerical labels
    for bar, rate in zip(bars, coverage_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., rate + 0.02, f'{rate:.1%}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Rotate x-axis labels to avoid overlap
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Applicability domain expansion ratio (using ECFP as baseline)
    baseline_method = 'ECFP'
    improvement_ratios = []
    method_names = []

    baseline_rate = uncertainty_results[baseline_method]['coverage_rate']

    for method in methods:
        if method != baseline_method:
            current_rate = uncertainty_results[method]['coverage_rate']
            improvement_ratios.append(current_rate / baseline_rate)
            method_names.append(method)

    x = np.arange(len(method_names))
    bars = axes[0, 1].bar(x, improvement_ratios, color=colors[1:len(method_names) + 1])
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (ECFP)')
    axes[0, 1].set_xlabel('Method')
    axes[0, 1].set_ylabel('Coverage Improvement Ratio')
    axes[0, 1].set_title('Applicability Domain Expansion\n(Relative to ECFP, Threshold=0.8)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(method_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Add numerical labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Confidence distribution box plot
    confidence_data = [uncertainty_results[method]['confidence_scores'] for method in methods]

    bp = axes[1, 0].boxplot(confidence_data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)

    axes[1, 0].set_xlabel('Method')
    axes[1, 0].set_ylabel('Confidence Score')
    axes[1, 0].set_title('Confidence Score Distribution (Threshold=0.8)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Add threshold line
    axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold=0.8')
    axes[1, 0].legend()

    # 4. Coverage rate distribution comparison (Enhanced vs Traditional)
    enhanced_rate = uncertainty_results['Enhanced']['coverage_rate']
    traditional_rates = [uncertainty_results[method]['coverage_rate']
                         for method in methods if method != 'Enhanced']

    box_data = [traditional_rates, [enhanced_rate]]
    box_labels = ['Traditional Methods', 'Enhanced Method']

    bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#EEC302')
    bp['boxes'][1].set_facecolor('#5D99BE')

    axes[1, 1].set_ylabel('Coverage Rate')
    axes[1, 1].set_title('Coverage Rate: Enhanced vs Traditional')
    axes[1, 1].grid(True, alpha=0.3)

    # Add specific numerical scatter points
    for i, rates in enumerate([traditional_rates, [enhanced_rate]]):
        y = rates
        x = [i + 1] * len(rates)
        axes[1, 1].scatter(x, y, alpha=0.8, s=50, c='red')
        for j, rate in enumerate(rates):
            axes[1, 1].text(i + 1, rate + 0.02, f'{rate:.1%}',
                            ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save figures
    plt.savefig('uncertainty_coverage_comparison_08.png', dpi=300, bbox_inches='tight')
    plt.savefig('uncertainty_coverage_comparison_08.pdf', dpi=300, bbox_inches='tight')
    print("Figures saved: uncertainty_coverage_comparison_08.png and uncertainty_coverage_comparison_08.pdf")

    plt.show()


def generate_uncertainty_report(uncertainty_results):
    """Generate detailed analysis report based on uncertainty (using 0.8 threshold only)"""

    print("\n" + "=" * 60)
    print("      Uncertainty-Based Applicability Domain Coverage Analysis Report (Threshold=0.8)")
    print("=" * 60)

    methods = list(uncertainty_results.keys())

    # 1. Summary of prediction uncertainty method
    print("\n1. Prediction Uncertainty-Based Confidence Assessment Results (Threshold=0.8):")
    print("-" * 50)
    print(f"{'Method':<12} {'Coverage Rate':<12} {'Covered Samples':<12}")
    print("-" * 50)

    for method in methods:
        rate = uncertainty_results[method]['coverage_rate']
        samples = uncertainty_results[method]['covered_samples']
        print(f"{method:<12} {rate:<12.1%} {samples:<12}")

    # 2. Enhanced representation vs traditional methods comparison
    print("\n2. Enhanced Representation vs Traditional Methods Improvement:")
    print("-" * 50)

    enhanced_rate = uncertainty_results['Enhanced']['coverage_rate']
    traditional_methods = [m for m in methods if m != 'Enhanced']

    for method in traditional_methods:
        traditional_rate = uncertainty_results[method]['coverage_rate']
        improvement = (enhanced_rate - traditional_rate) / traditional_rate * 100
        print(f"  vs {method}: {improvement:+.1f}% improvement ({enhanced_rate:.1%} vs {traditional_rate:.1%})")

    # 3. Ranking analysis
    print("\n3. Method Ranking (by coverage rate in descending order):")
    print("-" * 30)

    sorted_methods = sorted(methods, key=lambda x: uncertainty_results[x]['coverage_rate'], reverse=True)
    for i, method in enumerate(sorted_methods, 1):
        rate = uncertainty_results[method]['coverage_rate']
        print(f"{i}. {method}: {rate:.1%}")

    # 4. Key findings
    print("\n4. Key Findings:")
    print("-" * 30)

    # Find best traditional method
    best_traditional = max(traditional_methods,
                           key=lambda x: uncertainty_results[x]['coverage_rate'])
    best_traditional_rate = uncertainty_results[best_traditional]['coverage_rate']

    improvement = (enhanced_rate - best_traditional_rate) / best_traditional_rate * 100

    print(f"• Enhanced representation improves {improvement:.1f}% over best traditional method ({best_traditional})")
    print(f"• Enhanced representation coverage rate: {enhanced_rate:.1%}")
    print(f"• Best traditional method ({best_traditional}) coverage rate: {best_traditional_rate:.1%}")

    # Find worst method
    worst_method = min(methods, key=lambda x: uncertainty_results[x]['coverage_rate'])
    worst_rate = uncertainty_results[worst_method]['coverage_rate']
    print(f"• Lowest coverage rate method: {worst_method} ({worst_rate:.1%})")

    # 5. Confidence distribution statistics
    print("\n5. Confidence Distribution Statistics:")
    print("-" * 40)

    for method in methods:
        confidence_scores = uncertainty_results[method]['confidence_scores']
        print(f"{method}:")
        print(f"  Mean confidence: {np.mean(confidence_scores):.3f}")
        print(f"  Confidence std: {np.std(confidence_scores):.3f}")
        print(f"  Min confidence: {np.min(confidence_scores):.3f}")
        print(f"  Max confidence: {np.max(confidence_scores):.3f}")
        print(
            f"  High confidence samples (>0.8): {(confidence_scores > 0.8).sum()}/{len(confidence_scores)} ({(confidence_scores > 0.8).mean():.1%})")
        print()

    print("=" * 60)


# Main function
if __name__ == "__main__":
    uncertainty_results = uncertainty_based_applicability_domain_analysis()