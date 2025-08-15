import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import warnings
from tqdm import tqdm
import pickle
from test_activation_after_transformation import leet_speak, get_model_probs
import plotly.graph_objects as go

from sae_utils import (
    get_device,
    load_model_and_tokenizer,
    load_sae,
    get_sae_checkpoint_path,
    load_jailbreak_dataset,
    get_activations_with_sae,
)

warnings.filterwarnings("ignore")

# --- Configuration ---
LAYER = 10  # Layer to analyze
FEATURE_IDX = 2337  # Specific SAE feature to use
SAE_BASE_DIR = "sae-test"
CACHE_DIR = "cache"
BATCH_SIZE = 8


def get_sae_feature_activations(texts, model, tokenizer, ae, layer, device, feature_idx, batch_size=8):
    """Get SAE feature activations for a specific feature."""
    all_feature_activations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting SAE activations"):
        batch_texts = texts[i : i + batch_size]
        
        # Get SAE activations
        _, sae_features = get_activations_with_sae(
            model, tokenizer, ae, batch_texts, layer, device, batch_size=batch_size
        )
        
        # Extract the specific feature and take mean over sequence length
        feature_activations = sae_features[:, :, feature_idx].mean(dim=1).cpu().numpy()  # [batch]
        all_feature_activations.append(feature_activations)
    
    return np.concatenate(all_feature_activations, axis=0)  # [N]


def create_mixed_data(texts, labels, transform_fn, random_state=42):
    """Create mixed data where only half of jailbreak samples are transformed."""
    import random
    random.seed(random_state)
    
    mixed_texts = []
    mixed_labels = []
    
    # Separate jailbreak and non-jailbreak samples
    jailbreak_indices = [i for i, label in enumerate(labels) if label == 1]
    non_jailbreak_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Transform half of jailbreak samples
    num_jailbreak = len(jailbreak_indices)
    num_to_transform = num_jailbreak // 2
    jailbreak_to_transform = random.sample(jailbreak_indices, num_to_transform)
    
    # Process all samples
    for i, (text, label) in enumerate(zip(texts, labels)):
        if i in jailbreak_to_transform:
            # Transform this jailbreak sample
            mixed_texts.append(transform_fn(text))
        else:
            # Keep original
            mixed_texts.append(text)
        mixed_labels.append(label)
    
    print(f"Mixed dataset: {num_to_transform}/{num_jailbreak} jailbreak samples transformed")
    return mixed_texts, mixed_labels


def load_or_compute_data(split, transform_fn=None, force_recompute=False, mixed=False):
    """Load cached data or compute and cache it."""
    if mixed:
        transform_suffix = "_mixed"
    else:
        transform_suffix = "_leetspeak" if transform_fn else "_original"
    
    cache_file = os.path.join(CACHE_DIR, f"{split}_data{transform_suffix}.pkl")
    
    if not force_recompute and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing data for {split} split{transform_suffix}...")
    
    # Load dataset
    dataset = load_jailbreak_dataset(split=split)
    dataset_list = list(dataset)
    texts = [ex["prompt"] for ex in dataset_list]
    labels = [1 if ex["type"] == "jailbreak" else 0 for ex in dataset_list]
    
    # Apply transformation based on type
    if mixed and transform_fn:
        texts, labels = create_mixed_data(texts, labels, transform_fn)
    elif transform_fn:
        texts = [transform_fn(t) for t in texts]
    
    # Get model probabilities
    probs = get_model_probs(texts, model, tokenizer, device, BATCH_SIZE)
    
    # Get SAE feature activations
    feature_activations = get_sae_feature_activations(
        texts, model, tokenizer, ae, LAYER, device, FEATURE_IDX, BATCH_SIZE
    )
    
    # Cache the results
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {
        'texts': texts,
        'labels': labels,
        'probs': probs,
        'feature_activations': feature_activations
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Cached data to {cache_file}")
    return data


def train_combined_classifier(train_data_orig, train_data_leetspeak, train_data_mixed=None):
    """Train a combined classifier using original, leetspeak, and optionally mixed data."""
    print("Training combined classifier...")
    
    # Combine original and leetspeak data
    X_orig = np.column_stack([
        train_data_orig['probs'],  # Original model probabilities [N, 3]
        train_data_orig['feature_activations'].reshape(-1, 1)  # SAE feature [N, 1]
    ])
    
    X_leetspeak = np.column_stack([
        train_data_leetspeak['probs'],  # Leetspeak model probabilities [N, 3]
        train_data_leetspeak['feature_activations'].reshape(-1, 1)  # SAE feature [N, 1]
    ])
    
    # Combine features
    X_combined = np.vstack([X_orig, X_leetspeak])  # [2N, 4]
    
    # Combine labels (same labels for both versions)
    y_combined = np.concatenate([
        train_data_orig['labels'],
        train_data_leetspeak['labels']
    ])
    
    # Add mixed data if provided
    if train_data_mixed is not None:
        X_mixed = np.column_stack([
            train_data_mixed['probs'],  # Mixed model probabilities [N, 3]
            train_data_mixed['feature_activations'].reshape(-1, 1)  # SAE feature [N, 1]
        ])
        X_combined = np.vstack([X_combined, X_mixed])  # [3N, 4]
        y_combined = np.concatenate([y_combined, train_data_mixed['labels']])
        print(f"Training with {len(X_combined)} samples (original + leetspeak + mixed)")
    else:
        print(f"Training with {len(X_combined)} samples (original + leetspeak)")
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_combined, y_combined)
    
    return clf


def evaluate_classifiers(clf_combined, test_data_orig, test_data_leetspeak, test_data_mixed=None):
    """Evaluate both the combined classifier and original model."""
    print("Evaluating classifiers...")
    
    # Prepare test data
    X_test_orig = np.column_stack([
        test_data_orig['probs'],
        test_data_orig['feature_activations'].reshape(-1, 1)
    ])
    
    X_test_leetspeak = np.column_stack([
        test_data_leetspeak['probs'],
        test_data_leetspeak['feature_activations'].reshape(-1, 1)
    ])
    
    # Original model predictions (using class 2 as jailbreak probability)
    orig_preds_orig = test_data_orig['probs'][:, 2] > 0.5
    orig_preds_leetspeak = test_data_leetspeak['probs'][:, 2] > 0.5
    
    # Combined classifier predictions
    combined_preds_orig = clf_combined.predict(X_test_orig)
    combined_preds_leetspeak = clf_combined.predict(X_test_leetspeak)
    
    # Calculate metrics
    results = {}
    
    # Original model on original data
    results['orig_model_orig'] = {
        'accuracy': accuracy_score(test_data_orig['labels'], orig_preds_orig),
        'auc': roc_auc_score(test_data_orig['labels'], test_data_orig['probs'][:, 2]),
        'report': classification_report(test_data_orig['labels'], orig_preds_orig)
    }
    
    # Original model on leetspeak data
    results['orig_model_leetspeak'] = {
        'accuracy': accuracy_score(test_data_leetspeak['labels'], orig_preds_leetspeak),
        'auc': roc_auc_score(test_data_leetspeak['labels'], test_data_leetspeak['probs'][:, 2]),
        'report': classification_report(test_data_leetspeak['labels'], orig_preds_leetspeak)
    }
    
    # Combined classifier on original data
    results['combined_orig'] = {
        'accuracy': accuracy_score(test_data_orig['labels'], combined_preds_orig),
        'auc': roc_auc_score(test_data_orig['labels'], clf_combined.predict_proba(X_test_orig)[:, 1]),
        'report': classification_report(test_data_orig['labels'], combined_preds_orig)
    }
    
    # Combined classifier on leetspeak data
    results['combined_leetspeak'] = {
        'accuracy': accuracy_score(test_data_leetspeak['labels'], combined_preds_leetspeak),
        'auc': roc_auc_score(test_data_leetspeak['labels'], clf_combined.predict_proba(X_test_leetspeak)[:, 1]),
        'report': classification_report(test_data_leetspeak['labels'], combined_preds_leetspeak)
    }
    
    # Add mixed dataset evaluation if provided
    if test_data_mixed is not None:
        X_test_mixed = np.column_stack([
            test_data_mixed['probs'],
            test_data_mixed['feature_activations'].reshape(-1, 1)
        ])
        
        orig_preds_mixed = test_data_mixed['probs'][:, 2] > 0.5
        combined_preds_mixed = clf_combined.predict(X_test_mixed)
        
        # Original model on mixed data
        results['orig_model_mixed'] = {
            'accuracy': accuracy_score(test_data_mixed['labels'], orig_preds_mixed),
            'auc': roc_auc_score(test_data_mixed['labels'], test_data_mixed['probs'][:, 2]),
            'report': classification_report(test_data_mixed['labels'], orig_preds_mixed)
        }
        
        # Combined classifier on mixed data
        results['combined_mixed'] = {
            'accuracy': accuracy_score(test_data_mixed['labels'], combined_preds_mixed),
            'auc': roc_auc_score(test_data_mixed['labels'], clf_combined.predict_proba(X_test_mixed)[:, 1]),
            'report': classification_report(test_data_mixed['labels'], combined_preds_mixed)
        }
    
    return results


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nORIGINAL MODEL:")
    print("-" * 40)
    print(f"Original data - Accuracy: {results['orig_model_orig']['accuracy']:.4f}, AUC: {results['orig_model_orig']['auc']:.4f}")
    print(f"Leetspeak data - Accuracy: {results['orig_model_leetspeak']['accuracy']:.4f}, AUC: {results['orig_model_leetspeak']['auc']:.4f}")
    if 'orig_model_mixed' in results:
        print(f"Mixed data - Accuracy: {results['orig_model_mixed']['accuracy']:.4f}, AUC: {results['orig_model_mixed']['auc']:.4f}")
    
    print("\nCOMBINED CLASSIFIER:")
    print("-" * 40)
    print(f"Original data - Accuracy: {results['combined_orig']['accuracy']:.4f}, AUC: {results['combined_orig']['auc']:.4f}")
    print(f"Leetspeak data - Accuracy: {results['combined_leetspeak']['accuracy']:.4f}, AUC: {results['combined_leetspeak']['auc']:.4f}")
    if 'combined_mixed' in results:
        print(f"Mixed data - Accuracy: {results['combined_mixed']['accuracy']:.4f}, AUC: {results['combined_mixed']['auc']:.4f}")
    
    print("\nIMPROVEMENTS:")
    print("-" * 40)
    orig_improvement = results['combined_orig']['accuracy'] - results['orig_model_orig']['accuracy']
    leetspeak_improvement = results['combined_leetspeak']['accuracy'] - results['orig_model_leetspeak']['accuracy']
    
    print(f"Original data improvement: {orig_improvement:+.4f}")
    print(f"Leetspeak data improvement: {leetspeak_improvement:+.4f}")
    if 'combined_mixed' in results:
        mixed_improvement = results['combined_mixed']['accuracy'] - results['orig_model_mixed']['accuracy']
        print(f"Mixed data improvement: {mixed_improvement:+.4f}")
    
    print("\nDETAILED CLASSIFICATION REPORTS:")
    print("-" * 40)
    print("\nOriginal Model on Original Data:")
    print(results['orig_model_orig']['report'])
    
    print("\nOriginal Model on Leetspeak Data:")
    print(results['orig_model_leetspeak']['report'])
    
    if 'orig_model_mixed' in results:
        print("\nOriginal Model on Mixed Data:")
        print(results['orig_model_mixed']['report'])
    
    print("\nCombined Classifier on Original Data:")
    print(results['combined_orig']['report'])
    
    print("\nCombined Classifier on Leetspeak Data:")
    print(results['combined_leetspeak']['report'])
    
    if 'combined_mixed' in results:
        print("\nCombined Classifier on Mixed Data:")
        print(results['combined_mixed']['report'])


def create_accuracy_plot(results):
    """Create a bar plot showing accuracies grouped by dataset."""
    print("Creating accuracy comparison plot...")
    
    # Determine which datasets are available
    has_mixed = 'orig_model_mixed' in results
    
    if has_mixed:
        datasets = ["Original Data", "Adversarial Data", "Mixed Data"]
        
        # Extract accuracies for each classifier on each dataset
        orig_model_acc = [
            results['orig_model_orig']['accuracy'] * 100,
            results['orig_model_leetspeak']['accuracy'] * 100,
            results['orig_model_mixed']['accuracy'] * 100
        ]
        
        combined_acc = [
            results['combined_orig']['accuracy'] * 100,
            results['combined_leetspeak']['accuracy'] * 100,
            results['combined_mixed']['accuracy'] * 100
        ]
    else:
        datasets = ["Original Data", "Adversarial Data"]
        
        # Extract accuracies for each classifier on each dataset
        orig_model_acc = [
            results['orig_model_orig']['accuracy'] * 100,
            results['orig_model_leetspeak']['accuracy'] * 100
        ]
        
        combined_acc = [
            results['combined_orig']['accuracy'] * 100,
            results['combined_leetspeak']['accuracy'] * 100
        ]
    
    fig = go.Figure()
    
    # Add bars for Original Model
    fig.add_trace(
        go.Bar(
            name="Prompt-Guard",
            x=datasets,
            y=orig_model_acc,
            marker_color="#1f77b4",  # Blue
            text=[f"{acc:.1f}%" for acc in orig_model_acc],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            width=0.35,
        )
    )
    
    # Add bars for Combined Classifier
    fig.add_trace(
        go.Bar(
            name="Prompt-Guard + SAE Feature",
            x=datasets,
            y=combined_acc,
            marker_color="#ff7f0e",  # Orange
            text=[f"{acc:.1f}%" for acc in combined_acc],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            width=0.35,
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Classifier Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        yaxis_title="Accuracy (%)",
        yaxis=dict(
            range=[0, 100],
            tickmode='linear',
            tick0=0,
            dtick=20,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=100, b=80),
    )
    
    # Save the plot
    os.makedirs("figures", exist_ok=True)
    plot_file = f"figures/combined_classifier_accuracy_comparison_layer{LAYER}_feature{FEATURE_IDX}.png"
    fig.write_image(plot_file, scale=3)
    print(f"Accuracy comparison plot saved to {plot_file}")
    
    return fig


def main():
    """Main function to run the combined classifier experiment."""
    print("=== Combined Classifier Experiment ===")
    print(f"Layer: {LAYER}, Feature: {FEATURE_IDX}")
    print(f"SAE Base Dir: {SAE_BASE_DIR}")
    
    # Setup
    global device, model, tokenizer, ae
    device = get_device()
    model_name = "meta-llama/Prompt-Guard-86M"
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Load SAE
    checkpoint_path = get_sae_checkpoint_path(LAYER, base_dir=SAE_BASE_DIR)
    ae = load_sae(checkpoint_path, device)
    
    print(f"Loaded SAE from {checkpoint_path}")
    
    # Get feature dimension by running a dummy forward pass
    dummy_input = torch.randn(1, 10, 768).to(device)  # [batch, seq_len, hidden_dim]
    with torch.no_grad():
        dummy_features = ae.encode(dummy_input)
    n_features = dummy_features.shape[-1]
    print(f"SAE feature dimension: {n_features}")
    
    if FEATURE_IDX >= n_features:
        raise ValueError(f"Feature index {FEATURE_IDX} out of range. SAE has {n_features} features.")
    
    # Load or compute data for both splits and transformations
    print("\nLoading/computing data...")
    train_data_orig = load_or_compute_data("train", transform_fn=None)
    train_data_leetspeak = load_or_compute_data("train", transform_fn=leet_speak)
    train_data_mixed = load_or_compute_data("train", transform_fn=leet_speak, mixed=True)
    test_data_orig = load_or_compute_data("test", transform_fn=None)
    test_data_leetspeak = load_or_compute_data("test", transform_fn=leet_speak)
    test_data_mixed = load_or_compute_data("test", transform_fn=leet_speak, mixed=True)
    
    print(f"Train original: {len(train_data_orig['texts'])} samples")
    print(f"Train leetspeak: {len(train_data_leetspeak['texts'])} samples")
    print(f"Train mixed: {len(train_data_mixed['texts'])} samples")
    print(f"Test original: {len(test_data_orig['texts'])} samples")
    print(f"Test leetspeak: {len(test_data_leetspeak['texts'])} samples")
    print(f"Test mixed: {len(test_data_mixed['texts'])} samples")
    
    # Train combined classifier
    clf_combined = train_combined_classifier(train_data_orig, train_data_leetspeak, train_data_mixed)
    
    # Evaluate
    results = evaluate_classifiers(clf_combined, test_data_orig, test_data_leetspeak, test_data_mixed)
    
    # Print results
    print_results(results)
    
    # Create and save accuracy comparison plot
    create_accuracy_plot(results)
    
    # Save results
    results_file = os.path.join(CACHE_DIR, f"combined_classifier_results_layer{LAYER}_feature{FEATURE_IDX}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
