import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def extract_maccs_fingerprints(mol_list):
    """Extract MACCS fingerprints from molecule objects"""
    fingerprints = []
    for mol in mol_list:
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.append(np.array(fp))
        except:
            fingerprints.append(np.zeros(167))  # MACCS fingerprint length is 167
    return np.array(fingerprints)


def extract_ecfp_fingerprints(mol_list, radius=2, nBits=2048):
    """Extract ECFP (Extended Connectivity Fingerprints) from molecule objects"""
    fingerprints = []
    for mol in mol_list:
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints.append(np.array(fp))
        except:
            fingerprints.append(np.zeros(nBits))
    return np.array(fingerprints)


def extract_rdkit_descriptors(mol_list):
    """Extract RDKit 2D molecular descriptors"""
    descriptor_names = [name for name, _ in Descriptors._descList]
    descriptors_matrix = []

    for mol in mol_list:
        descriptors = []
        for name, func in Descriptors._descList:
            try:
                value = func(mol)
                # Handle infinite and NaN values
                if np.isfinite(value):
                    descriptors.append(value)
                else:
                    descriptors.append(0.0)
            except:
                descriptors.append(0.0)
        descriptors_matrix.append(descriptors)

    return np.array(descriptors_matrix), descriptor_names


def extract_combined_descriptors(mol_list):
    """Extract combined descriptors (MACCS + ECFP + RDKit)"""
    print("Extracting MACCS fingerprints...")
    maccs = extract_maccs_fingerprints(mol_list)

    print("Extracting ECFP fingerprints...")
    ecfp = extract_ecfp_fingerprints(mol_list)

    print("Extracting RDKit descriptors...")
    rdkit, rdkit_names = extract_rdkit_descriptors(mol_list)

    # Combine all features
    combined = np.hstack([maccs, ecfp, rdkit])

    # Create feature names
    feature_names = []
    feature_names.extend([f'MACCS_{i}' for i in range(167)])
    feature_names.extend([f'ECFP_{i}' for i in range(2048)])
    feature_names.extend([f'RDKit_{name}' for name in rdkit_names])

    print(f"Combined descriptors shape: {combined.shape}")
    print(f"  MACCS: {maccs.shape[1]} features")
    print(f"  ECFP: {ecfp.shape[1]} features")
    print(f"  RDKit: {rdkit.shape[1]} features")
    print(f"  Total: {combined.shape[1]} features")

    return {
        'MACCS': maccs,
        'ECFP': ecfp,
        'RDKit': rdkit,
        'Combined': combined
    }, feature_names


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
    print(f"  Label distribution: {np.bincount(unique_labels)}")

    return unique_molecules, np.array(unique_labels)


def preprocess_features(X_train, X_test, feature_type):
    """Preprocess features (standardization, variance threshold, etc.)"""

    if feature_type == 'RDKit':
        # RDKit descriptors need standardization
        print("Standardizing RDKit descriptors...")
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)

        # Remove low variance features
        print("Removing low variance features...")
        selector = VarianceThreshold(threshold=0.01)
        X_train_processed = selector.fit_transform(X_train_processed)
        X_test_processed = selector.transform(X_test_processed)

        print(f"After preprocessing: {X_train_processed.shape[1]} features retained")
        return X_train_processed, X_test_processed

    elif feature_type == 'Combined':
        # Combined descriptors: only standardize RDKit part
        print("Preprocessing combined descriptors...")

        # Separate different feature types
        maccs_train = X_train[:, :167]
        ecfp_train = X_train[:, 167:167 + 2048]
        rdkit_train = X_train[:, 167 + 2048:]

        maccs_test = X_test[:, :167]
        ecfp_test = X_test[:, 167:167 + 2048]
        rdkit_test = X_test[:, 167 + 2048:]

        # Only standardize RDKit part
        scaler = StandardScaler()
        rdkit_train_scaled = scaler.fit_transform(rdkit_train)
        rdkit_test_scaled = scaler.transform(rdkit_test)

        # Recombine features
        X_train_processed = np.hstack([maccs_train, ecfp_train, rdkit_train_scaled])
        X_test_processed = np.hstack([maccs_test, ecfp_test, rdkit_test_scaled])

        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        X_train_processed = selector.fit_transform(X_train_processed)
        X_test_processed = selector.transform(X_test_processed)

        print(f"After preprocessing: {X_train_processed.shape[1]} features retained")
        return X_train_processed, X_test_processed

    else:
        # MACCS and ECFP fingerprints don't need special preprocessing
        return X_train, X_test


def comprehensive_ml_benchmark(X_train, X_test, y_train, y_test, feature_type):
    """Benchmark multiple machine learning methods on specified feature type"""
    print(f"\n=== ML Benchmark on {feature_type} Features ===")

    # Preprocess features
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test, feature_type)

    # Define models
    models = {
        # Linear models
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=3407),
        'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000,
                                                       random_state=3407),

        # Tree models
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=3407),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=3407),
        'Extra Trees': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=3407, bootstrap=False),

        # Gradient boosting
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=3407,
                                 eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                                        random_state=3407),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=3407),

        # Instance-based learning
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),

        # Support vector machines
        'SVM (RBF)': SVC(probability=True, kernel='rbf', random_state=3407),
        'SVM (Linear)': SVC(probability=True, kernel='linear', random_state=3407),

        # Neural networks
        'MLP (1 layer)': MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=3407),
        'MLP (2 layers)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=3407),

        # Naive Bayes
        'Naive Bayes': GaussianNB(),
    }

    results = {}
    detailed_results = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        try:
            # For certain models, additional standardization may be needed
            if model_name in ['SVM (RBF)', 'SVM (Linear)', 'MLP (1 layer)', 'MLP (2 layers)'] and feature_type in [
                'MACCS', 'ECFP']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_proc)
                X_test_scaled = scaler.transform(X_test_proc)

                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_proc, y_train)
                y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
                y_pred = model.predict(X_test_proc)

            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[model_name] = auc
            detailed_results[model_name] = {
                'AUC': auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }

            print(f"  AUC: {auc:.4f}")

        except Exception as e:
            print(f"  Failed: {e}")
            results[model_name] = 0.0
            detailed_results[model_name] = {
                'AUC': 0.0, 'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0
            }

    return results, detailed_results


def visualize_comprehensive_comparison(all_results, enhanced_performance=0.92):
    """Visualize comprehensive comparison across all feature types"""
    print("\n=== Creating Comprehensive Visualization ===")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Best performance comparison by feature type
    ax1 = axes[0, 0]
    feature_types = list(all_results.keys())
    best_scores = [max(all_results[ft].values()) for ft in feature_types]
    best_methods = [max(all_results[ft], key=all_results[ft].get) for ft in feature_types]

    bars = ax1.bar(feature_types, best_scores, alpha=0.7,
                   color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax1.axhline(y=enhanced_performance, color='red', linestyle='--', linewidth=2,
                label=f'Enhanced Representation ({enhanced_performance})')
    ax1.set_ylabel('Best AUC Score')
    ax1.set_title('Best Performance by Feature Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate best methods
    for i, (score, method) in enumerate(zip(best_scores, best_methods)):
        ax1.text(i, score + 0.01, f'{score:.3f}\n({method})', ha='center', va='bottom', fontsize=8)

    # 2. Heatmap: performance of all methods on all feature types
    ax2 = axes[0, 1]

    # Get all model names
    all_models = set()
    for results in all_results.values():
        all_models.update(results.keys())
    all_models = sorted(list(all_models))

    # Create heatmap data
    heatmap_data = []
    for model in all_models:
        row = []
        for ft in feature_types:
            score = all_results[ft].get(model, 0.0)
            row.append(score)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0.5, vmax=1.0)

    ax2.set_xticks(range(len(feature_types)))
    ax2.set_xticklabels(feature_types)
    ax2.set_yticks(range(len(all_models)))
    ax2.set_yticklabels(all_models)
    ax2.set_title('Performance Heatmap (AUC)')

    # Add numerical annotations
    for i in range(len(all_models)):
        for j in range(len(feature_types)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}', ha="center", va="center", color="black", fontsize=6)

    plt.colorbar(im, ax=ax2)

    # 3. Top 5 methods comparison for each feature type
    ax3 = axes[1, 0]

    colors = ['blue', 'green', 'orange', 'red']
    x_offset = np.array([-0.3, -0.1, 0.1, 0.3])

    for i, (ft, color) in enumerate(zip(feature_types, colors)):
        sorted_results = sorted(all_results[ft].items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_results[:5]
        methods = [item[0] for item in top_5]
        scores = [item[1] for item in top_5]

        x_pos = np.arange(len(methods)) + x_offset[i]
        ax3.bar(x_pos, scores, width=0.2, label=ft, alpha=0.7, color=color)

    ax3.axhline(y=enhanced_performance, color='red', linestyle='--', linewidth=2,
                label=f'Enhanced ({enhanced_performance})')
    ax3.set_xlabel('Methods (Top 5 for each feature type)')
    ax3.set_ylabel('AUC Score')
    ax3.set_title('Top 5 Methods Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Performance gain analysis
    ax4 = axes[1, 1]

    improvements = []
    labels = []
    for ft in feature_types:
        best_score = max(all_results[ft].values())
        improvement = enhanced_performance - best_score
        improvements.append(improvement)
        labels.append(f'{ft}\n({best_score:.3f})')

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax4.bar(labels, improvements, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Performance Gain vs Enhanced')
    ax4.set_title('Enhanced Representation Advantage')
    ax4.grid(True, alpha=0.3)

    # Annotate values
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax4.text(bar.get_x() + bar.get_width() / 2, imp + (0.005 if imp >= 0 else -0.005),
                 f'{imp:+.3f}', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig('traditional_vs_enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_comprehensive_report(all_results, enhanced_performance=0.92):
    """Generate comprehensive comparison report"""
    print("\n" + "=" * 80)
    print("           TRADITIONAL DESCRIPTORS vs ENHANCED REPRESENTATION")
    print("=" * 80)

    # Best performance by feature type
    print(f"\n📊 BEST PERFORMANCE BY FEATURE TYPE:")
    print(f"{'Feature Type':<20} {'Best Method':<20} {'AUC':<8} {'Gap vs Enhanced':<15}")
    print("-" * 70)

    for ft in all_results.keys():
        best_method = max(all_results[ft], key=all_results[ft].get)
        best_score = all_results[ft][best_method]
        gap = enhanced_performance - best_score
        print(f"{ft:<20} {best_method:<20} {best_score:<8.4f} {gap:<+15.4f}")

    # Find overall best traditional method
    overall_best_score = 0
    overall_best_method = ""
    overall_best_feature = ""

    for ft, results in all_results.items():
        for method, score in results.items():
            if score > overall_best_score:
                overall_best_score = score
                overall_best_method = method
                overall_best_feature = ft

    print(f"\n🏆 OVERALL BEST TRADITIONAL METHOD:")
    print(f"   Method: {overall_best_method} on {overall_best_feature}")
    print(f"   AUC: {overall_best_score:.4f}")
    print(f"   Gap vs Enhanced: {enhanced_performance - overall_best_score:+.4f}")
    print(f"   Relative Improvement: {((enhanced_performance / overall_best_score) - 1) * 100:+.2f}%")

    # Best performance by method category
    print(f"\n📈 BEST PERFORMANCE BY METHOD CATEGORY:")
    categories = {
        'Tree-based': ['Decision Tree', 'Random Forest', 'Extra Trees'],
        'Boosting': ['XGBoost', 'Gradient Boosting', 'AdaBoost'],
        'Linear': ['Logistic Regression', 'Logistic Regression (L1)'],
        'Instance-based': ['KNN (k=3)', 'KNN (k=5)', 'KNN (k=7)'],
        'SVM': ['SVM (RBF)', 'SVM (Linear)'],
        'Neural Network': ['MLP (1 layer)', 'MLP (2 layers)']
    }

    for category, methods in categories.items():
        best_score = 0
        best_feature = ""
        for ft, results in all_results.items():
            for method in methods:
                if method in results and results[method] > best_score:
                    best_score = results[method]
                    best_feature = ft

        if best_score > 0:
            gap = enhanced_performance - best_score
            print(f"   {category:<15}: {best_score:.4f} on {best_feature} (gap: {gap:+.4f})")

    print("\n" + "=" * 80)

    return {
        'overall_best_method': overall_best_method,
        'overall_best_feature': overall_best_feature,
        'overall_best_score': overall_best_score,
        'enhanced_advantage': enhanced_performance - overall_best_score
    }


def main():
    # Parameter settings
    csv_file = 'data3.csv'
    enhanced_performance = 0.92  # Your enhanced representation performance

    try:
        # 1. Deduplicate and get molecule objects
        unique_molecules, unique_labels = deduplicate_dataset(csv_file)

        # 2. Extract all types of traditional descriptors
        print("\n=== Extracting Traditional Descriptors ===")
        descriptor_dict, feature_names = extract_combined_descriptors(unique_molecules)

        # 3. Random split (use same split as enhanced representation)
        print("\n=== Performing Train-Test Split ===")
        # Use same random seed to ensure consistent splitting
        train_indices, test_indices = train_test_split(
            range(len(unique_labels)), test_size=0.1, random_state=3407,
            stratify=unique_labels
        )

        y_train = unique_labels[train_indices]
        y_test = unique_labels[test_indices]

        print(f"Split results:")
        print(f"  Training samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Train label distribution: {np.bincount(y_train)}")
        print(f"  Test label distribution: {np.bincount(y_test)}")

        # 4. Benchmark each feature type with machine learning methods
        all_results = {}

        for feature_type, features in descriptor_dict.items():
            print(f"\n{'=' * 50}")
            print(f"Testing {feature_type} features...")
            print(f"{'=' * 50}")

            X_train = features[train_indices]
            X_test = features[test_indices]

            results, detailed_results = comprehensive_ml_benchmark(
                X_train, X_test, y_train, y_test, feature_type
            )

            all_results[feature_type] = results

            # Display best result for current feature type
            best_method = max(results, key=results.get)
            best_score = results[best_method]
            print(f"\nBest {feature_type} result: {best_method} = {best_score:.4f}")

        # 5. Visualize comprehensive comparison
        visualize_comprehensive_comparison(all_results, enhanced_performance)

        # 6. Generate comprehensive report
        summary = generate_comprehensive_report(all_results, enhanced_performance)

        # 7. Save results
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv('traditional_descriptors_results.csv')

        # Save detailed results
        detailed_df = pd.DataFrame()
        for ft, results in all_results.items():
            temp_df = pd.DataFrame({
                'Feature_Type': ft,
                'Method': list(results.keys()),
                'AUC': list(results.values()),
                'Gap_vs_Enhanced': [enhanced_performance - score for score in results.values()]
            })
            detailed_df = pd.concat([detailed_df, temp_df], ignore_index=True)

        detailed_df.to_csv('traditional_detailed_results.csv', index=False)

        print(f"\n💾 Results saved:")
        print(f"   - traditional_descriptors_results.csv")
        print(f"   - traditional_detailed_results.csv")
        print(f"   - traditional_vs_enhanced_comparison.png")

        return summary

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    summary = main()