import os
import torch
import numpy as np
import plotly.graph_objs as go
from sae_utils import (
    get_device,
    load_model_and_tokenizer,
    load_sae,
    get_sae_checkpoint_path,
    load_jailbreak_dataset,
    get_activations_with_sae,
)
import torch.nn.functional as F
import scipy.stats


def leet_speak(text):
    table = str.maketrans(
        {
            "a": "4",
            "A": "4",
            "e": "3",
            "E": "3",
            "i": "1",
            "I": "1",
            "o": "0",
            "O": "0",
            "t": "7",
            "T": "7",
            "s": "5",
            "S": "5",
            "b": "8",
            "B": "8",
            "g": "9",
            "G": "9",
        }
    )
    return text.translate(table)


def append_omega(text):
    return text + "$Ω$"


def append_omega_infared(text):
    return text + "$ΩINFAREDΩ$"


def append_infared(text):
    return text + "INFAREDINFAREDINFAREDINFAREDINFARED"
    # "Ω_{INFARED}=Ω" -> 26.2% mean activation change


# --- Configurable parameters ---
LAYER = 1  # Layer to analyze
FEATURE_IDX = 587  # Feature to highlight
N_SAMPLES = 500  # Number of samples to use
TRANSFORM_FN = append_infared  # Transformation function
SAE_BASE_DIR = "sae-test"

# --- Setup ---
device = get_device()
model_name = "meta-llama/Prompt-Guard-86M"
model, tokenizer = load_model_and_tokenizer(model_name, device)
checkpoint_path = get_sae_checkpoint_path(LAYER, base_dir=SAE_BASE_DIR)
ae = load_sae(checkpoint_path, device)

# --- Load data (benign only) ---
dataset = load_jailbreak_dataset(split="test")
dataset_list = list(dataset)
benign_samples = [ex for ex in dataset_list if ex.get("type") != "jailbreak"]
texts = [ex["prompt"] for ex in benign_samples[:N_SAMPLES]]
texts_transformed = [TRANSFORM_FN(t) for t in texts]


# --- Get SAE activations ---
def get_sae_activations(texts):
    # Returns activations, sae_features ([batch, seq_len, n_features])
    _, sae_features = get_activations_with_sae(
        model, tokenizer, ae, texts, LAYER, device, batch_size=8
    )
    return sae_features


if __name__ == "__main__":
    sae_features_orig = get_sae_activations(texts)  # [N, seq_len, n_features]
    sae_features_trans = get_sae_activations(texts_transformed)

    # --- Compute change in activation ---
    # We'll use mean activation per feature per sample (mean over seq_len)
    orig_means = sae_features_orig.mean(axis=1).cpu().numpy()  # [N, n_features]
    trans_means = sae_features_trans.mean(axis=1).cpu().numpy()
    delta = trans_means - orig_means  # [N, n_features]
    # Aggregate across samples (mean absolute change per feature)
    feature_deltas = np.abs(delta).mean(axis=0)  # [n_features]


# --- Compute change in Prompt-Guard model label prediction ---
def get_model_probs(texts, model, tokenizer, device, batch_size=8):
    all_probs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits  # [batch, 3]
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()  # [batch, 3]
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)  # [N, 3]


if __name__ == "__main__":
    probs_orig = get_model_probs(texts, model, tokenizer, device)
    probs_trans = get_model_probs(texts_transformed, model, tokenizer, device)

    # Compute mean and std absolute change in predicted class probabilities
    delta_probs = probs_trans - probs_orig  # [N, 3]
    label_delta_mean = delta_probs.mean(axis=0)  # [3]
    label_delta_std = delta_probs.std(axis=0)  # [3]
    mean_label_delta = delta_probs.mean()
    std_label_delta = delta_probs.std()

    print("\nAverage change in Prompt-Guard model class probabilities (per class):")
    for i, (mean_d, std_d) in enumerate(zip(label_delta_mean, label_delta_std)):
        print(f"  Class {i}: Mean |ΔProb| = {mean_d:.6f} | Std = {std_d:.6f}")
    print(
        f"Overall mean |ΔProb| across all classes and samples: {mean_label_delta:.6f} | Std = {std_label_delta:.6f}\n"
    )


# --- Identify samples with most significant classification changes ---
def get_classification_changes(probs_orig, probs_trans):
    """Compute various metrics of classification change for each sample"""
    # Get predicted classes
    pred_orig = np.argmax(probs_orig, axis=1)  # [N]
    pred_trans = np.argmax(probs_trans, axis=1)  # [N]

    # Check if classification changed
    class_changed = pred_orig != pred_trans  # [N]

    # Compute maximum probability change for each sample
    max_prob_change = np.max(np.abs(delta_probs), axis=1)  # [N]

    # Compute total absolute probability change
    total_prob_change = np.sum(np.abs(delta_probs), axis=1)  # [N]

    # Compute confidence change (max probability before vs after)
    conf_orig = np.max(probs_orig, axis=1)  # [N]
    conf_trans = np.max(probs_trans, axis=1)  # [N]
    conf_change = conf_trans - conf_orig  # [N]

    return {
        "class_changed": class_changed,
        "pred_orig": pred_orig,
        "pred_trans": pred_trans,
        "max_prob_change": max_prob_change,
        "total_prob_change": total_prob_change,
        "conf_orig": conf_orig,
        "conf_trans": conf_trans,
        "conf_change": conf_change,
    }


if __name__ == "__main__":
    changes = get_classification_changes(probs_orig, probs_trans)

    # Find samples with most significant changes
    n_top_samples = 10

    # 1. Samples where classification changed
    changed_indices = np.where(changes["class_changed"])[0]
    print(
        f"\n=== SAMPLES WHERE CLASSIFICATION CHANGED ({len(changed_indices)} samples) ==="
    )
    if len(changed_indices) > 0:
        # Sort by total probability change
        sorted_changed = sorted(
            changed_indices, key=lambda i: changes["total_prob_change"][i], reverse=True
        )
        for i, idx in enumerate(sorted_changed[:n_top_samples]):
            print(f"\n{i + 1}. Sample {idx}:")
            print(
                f"   Original text: {texts[idx][:100]}{'...' if len(texts[idx]) > 100 else ''}"
            )
            print(
                f"   Transformed text: {texts_transformed[idx][:100]}{'...' if len(texts_transformed[idx]) > 100 else ''}"
            )
            print(
                f"   Original prediction: Class {changes['pred_orig'][idx]} (conf: {changes['conf_orig'][idx]:.3f})"
            )
            print(
                f"   New prediction: Class {changes['pred_trans'][idx]} (conf: {changes['conf_trans'][idx]:.3f})"
            )
            print(f"   Max probability change: {changes['max_prob_change'][idx]:.3f}")
    else:
        print("No samples had classification changes.")

    # 2. Samples with largest probability changes (regardless of class change)
    print(f"\n=== SAMPLES WITH LARGEST PROBABILITY CHANGES (Top {n_top_samples}) ===")
    sorted_by_prob_change = np.argsort(changes["total_prob_change"])[::-1]
    for i, idx in enumerate(sorted_by_prob_change[:n_top_samples]):
        print(f"\n{i + 1}. Sample {idx}:")
        print(
            f"   Original text: {texts[idx][:100]}{'...' if len(texts[idx]) > 100 else ''}"
        )
        print(
            f"   Transformed text: {texts_transformed[idx][:100]}{'...' if len(texts_transformed[idx]) > 100 else ''}"
        )
        print(
            f"   Original prediction: Class {changes['pred_orig'][idx]} (conf: {changes['conf_orig'][idx]:.3f})"
        )
        print(
            f"   New prediction: Class {changes['pred_trans'][idx]} (conf: {changes['conf_trans'][idx]:.3f})"
        )
        print(
            f"   Classification changed: {'Yes' if changes['class_changed'][idx] else 'No'}"
        )

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total samples analyzed: {len(texts)}")
    print(
        f"Samples with classification changes: {np.sum(changes['class_changed'])} ({100 * np.mean(changes['class_changed']):.1f}%)"
    )
    print(f"Mean total probability change: {np.mean(changes['total_prob_change']):.3f}")
    print(f"Mean confidence change: {np.mean(changes['conf_change']):.3f}")
    print(f"Max total probability change: {np.max(changes['total_prob_change']):.3f}")
    print(f"Min confidence change: {np.min(changes['conf_change']):.3f}")

    # --- Print features with highest change and non-zero activations ---
    TOP_K = 20  # Number of top features to print

    # Top features by mean absolute change
    sorted_indices = np.argsort(feature_deltas)[::-1]
    print(f"Top {TOP_K} features by mean absolute change in activation:")
    for rank, idx in enumerate(sorted_indices[:TOP_K]):
        print(
            f"  Rank {rank + 1}: Feature {idx} | Mean |ΔActivation| = {feature_deltas[idx]:.6f}"
        )

    # --- Plot histogram of feature deltas ---

    highlight_value = (
        feature_deltas[FEATURE_IDX] if FEATURE_IDX < len(feature_deltas) else None
    )
    if highlight_value is not None:
        quantile = (
            scipy.stats.percentileofscore(feature_deltas, highlight_value, kind="rank")
            / 100.0
        )
    else:
        quantile = None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=feature_deltas,
            nbinsx=60,
            marker_color="#1f77b4",
            name="Mean |ΔActivation|",
        )
    )

    # Add vertical line at the highlighted feature's value
    # if highlight_value is not None:
    #     fig.add_vline(
    #         x=highlight_value,
    #         line_width=3,
    #         line_dash="dash",
    #         line_color="#ff7f0e",
    #         annotation_text=f"Feature {FEATURE_IDX}",
    #         annotation_position="top right",
    #         annotation_font_color="#ff7f0e",
    #         annotation_font_size=14,
    #     )

    # Add arrow annotation at the highlighted feature's value
    if highlight_value is not None:
        fig.add_annotation(
            x=highlight_value,
            y=1,  # y=1 to point to the bottom of the plot (log scale)
            xref="x",
            yref="y",
            text=f"Feature {FEATURE_IDX}",
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=-60,  # Arrow points downward
            font=dict(color="#ff7f0e", size=14),
            arrowcolor="#ff7f0e",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ff7f0e",
        )

    fig.update_yaxes(type="log")

    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="Mean |ΔActivation|",
        width=500,
        height=500,
        showlegend=False,
    )

    os.makedirs("figures", exist_ok=True)
    fig.write_image(
        f"figures/activation_change_histogram_layer{LAYER}.png", width=500, height=500
    )
