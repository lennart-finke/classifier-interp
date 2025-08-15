from sae_utils import (
    get_device,
    load_model_and_tokenizer,
    load_sae,
    load_jailbreak_dataset,
)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings
import scipy.stats as st
import os
import torch
import plotly.graph_objs as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Set up
model_name = "meta-llama/Prompt-Guard-86M"
device = get_device()
model, tokenizer = load_model_and_tokenizer(model_name, device)

# Discover available layers (0-12)
LAYER_RANGE = list(range(13))

# Prepare data
X_train_texts, X_test_texts, y_train, y_test = None, None, None, None


def prepare_data_once():
    global X_train_texts, X_test_texts, y_train, y_test
    if X_train_texts is None:
        dataset = load_jailbreak_dataset(split="train")
        dataset_list = list(dataset)
        texts = [ex["prompt"] for ex in dataset_list]
        labels = [1 if ex["type"] == "jailbreak" else 0 for ex in dataset_list]
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=1, stratify=labels
        )
    return X_train_texts, X_test_texts, y_train, y_test


# Extract all activations for all layers in one pass
def extract_all_layer_activations(texts, split_name):
    print(f"Getting activations for all layers for {split_name} set...")
    batch_size = 8
    all_activations = {layer: [] for layer in LAYER_RANGE}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer in LAYER_RANGE:
            activations = outputs.hidden_states[layer].mean(dim=1)
            all_activations[layer].append(activations.cpu())
    all_activations_cat = {}
    for layer in LAYER_RANGE:
        all_activations_cat[layer] = torch.cat(all_activations[layer], dim=0)
    return all_activations_cat


def create_layerwise_metrics_plot(layer_metrics):
    """Create the layerwise metrics plot from cached results."""
    print("Creating layerwise metrics plot from cached data...")

    layers = [m["layer"] for m in layer_metrics]
    accuracies = [m["accuracy"] for m in layer_metrics]
    aucs = [m["auc"] for m in layer_metrics]

    fig = go.Figure()

    # Add accuracy line
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=accuracies,
            mode="lines+markers",
            name="Test Accuracy",
            line=dict(color="royalblue", width=3),
            marker=dict(symbol="circle", size=8, color="royalblue"),
            hovertemplate="Layer: %{x}<br>Accuracy: %{y:.4f}<extra></extra>",
        )
    )

    # Add AUC line
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=aucs,
            mode="lines+markers",
            name="Test AUC",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(symbol="square", size=8, color="#ff7f0e"),
            hovertemplate="Layer: %{x}<br>AUC: %{y:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="Test Accuracy and AUC Across Layers",
        xaxis_title="Layer",
        yaxis_title="Score",
        width=900,
        height=600,
        showlegend=True,
        title_x=0.5,
        title_font_size=24,
        font=dict(size=16),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    # Update axes
    fig.update_xaxes(
        tickmode="array",
        tickvals=layers,
        range=[min(layers) - 0.5, max(layers) + 0.5],
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        tickfont=dict(size=14),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        range=[0, 1],
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        tickfont=dict(size=14),
        title_font=dict(size=18),
    )

    os.makedirs("figures", exist_ok=True)
    fig.write_html("figures/layerwise_metrics.html")
    fig.write_image("figures/layerwise_metrics.png", width=900, height=600, scale=2)
    print(
        "Layerwise metrics plot saved to figures/layerwise_metrics.html and figures/layerwise_metrics.png"
    )


# Main loop for all layers
def main():
    print(
        "=== SAE Feature-based Logistic Regression for Prompt Classification (All Layers) ==="
    )

    # Check if layer metrics cache exists
    layer_metrics_cache_file = "cache/layer_metrics.json"
    if os.path.exists(layer_metrics_cache_file):
        print("Loading layer metrics from cache...")
        import json

        with open(layer_metrics_cache_file, "r") as f:
            layer_metrics = json.load(f)
        print(f"Loaded {len(layer_metrics)} layer metrics from cache")

        # Create the layerwise metrics plot directly
        create_layerwise_metrics_plot(layer_metrics)
        return

    X_train_texts, X_test_texts, y_train, y_test = prepare_data_once()
    # Extract and cache activations for all layers
    X_train_acts = extract_all_layer_activations(X_train_texts, "train")
    X_test_acts = extract_all_layer_activations(X_test_texts, "test")
    # Collect metrics for each layer
    layer_metrics = []
    for layer in LAYER_RANGE:
        print(f"\n=== Processing Layer {layer} ===")
        # SAE checkpoint path
        checkpoint_path = f"sae-test/layer_{layer}/trainer_0/checkpoints/ae_100000.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for layer {layer}, skipping.")
            continue
        ae = load_sae(checkpoint_path, device)
        # Extract SAE features
        print("Encoding train features with SAE...")
        train_tensor = X_train_acts[layer]
        if not isinstance(train_tensor, torch.Tensor):
            train_tensor = torch.tensor(train_tensor)
        X_train_features = ae.encode(train_tensor.to(device)).detach().cpu().numpy()
        print("Encoding test features with SAE...")
        test_tensor = X_test_acts[layer]
        if not isinstance(test_tensor, torch.Tensor):
            test_tensor = torch.tensor(test_tensor)
        X_test_features = ae.encode(test_tensor.to(device)).detach().cpu().numpy()
        # Train model
        print("Training logistic regression model...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_features, y_train)
        # Evaluate model
        print("Evaluating model...")
        y_pred = lr_model.predict(X_test_features)
        y_pred_proba = lr_model.predict_proba(X_test_features)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        # Collect metrics
        layer_metrics.append({"layer": layer, "accuracy": accuracy, "auc": auc})
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Benign", "Harmful"]))
        # Analyze feature importance
        print("Analyzing feature importance...")
        coefficients = lr_model.coef_[0]
        feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        probs = lr_model.predict_proba(X_train_features)
        p = probs[:, 1]
        X_design = np.hstack(
            [np.ones((X_train_features.shape[0], 1)), X_train_features]
        )
        V = np.diag(p * (1 - p))
        C = lr_model.C if hasattr(lr_model, "C") else 1.0
        reg = 1.0 / C
        reg_matrix = np.eye(X_design.shape[1])
        reg_matrix[0, 0] = 0
        fisher_info = X_design.T @ V @ X_design
        fisher_info_reg = fisher_info + reg * reg_matrix
        cov = np.linalg.inv(fisher_info_reg)
        se = np.sqrt(np.diag(cov))[1:]
        z_scores = coefficients / se
        p_values = 2 * (1 - st.norm.cdf(np.abs(z_scores)))
        feature_importance = list(zip(feature_names, coefficients, p_values))
        feature_importance.sort(key=lambda x: x[2])
        # Save important features
        import json

        important_features = []
        for feature, coef, p_val in feature_importance[:10]:
            feature_idx = int(feature.split("_")[1]) if "_" in feature else int(feature)
            important_features.append(
                {
                    "index": feature_idx,
                    "name": feature,
                    "coefficient": float(coef),
                    "p_value": float(p_val),
                }
            )
        cache_data = {
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "target_layer": layer,
            "important_features": important_features,
        }
        os.makedirs(f"cache/layer_{layer}", exist_ok=True)
        with open(f"cache/layer_{layer}/important_features.json", "w") as f:
            json.dump(cache_data, f, indent=2)
        print(
            f"Saved {len(important_features)} important features to cache/layer_{layer}/important_features.json"
        )
        # Plot feature importance with Plotly
        top_features = feature_importance[:20]
        feature_names_top = [f[0] for f in top_features]
        p_values_top = [f[2] for f in top_features]
        coefficients_top = [f[1] for f in top_features]
        neg_log_p_values = [-np.log10(p) for p in p_values_top]

        # Create color mapping based on coefficients
        colors = ["#ff7f0e" if c > 0 else "#1f77b4" for c in coefficients_top]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=feature_names_top,
                x=neg_log_p_values,
                orientation="h",
                marker=dict(color=colors, opacity=0.8),
                hovertemplate="<b>%{y}</b><br>"
                + "Coefficient: %{customdata[0]:.4f}<br>"
                + "P-value: %{customdata[1]:.2e}<br>"
                + "-log10(p): %{x:.3f}<extra></extra>",
                customdata=list(zip(coefficients_top, p_values_top)),
            )
        )

        # Add significance threshold lines
        fig.add_hline(
            y=-np.log10(0.05),
            line_dash="dash",
            line_color="orange",
            annotation_text="p < 0.05",
            annotation_position="top right",
            annotation=dict(font_size=14),
        )
        fig.add_hline(
            y=-np.log10(0.01),
            line_dash="dash",
            line_color="red",
            annotation_text="p < 0.01",
            annotation_position="top right",
            annotation=dict(font_size=14),
        )
        fig.add_hline(
            y=-np.log10(0.001),
            line_dash="dash",
            line_color="darkred",
            annotation_text="p < 0.001",
            annotation_position="top right",
            annotation=dict(font_size=14),
        )

        fig.update_layout(
            title=f"Top 20 Feature Importance (Layer {layer})",
            xaxis_title="-log10(p-value)",
            yaxis_title="Features",
            width=900,
            height=600,
            showlegend=False,
            title_x=0.5,
            title_font_size=24,
            font=dict(size=16),
            margin=dict(l=60, r=60, t=80, b=60),
        )
        fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))

        outdir = f"figures/layer_{layer}"
        os.makedirs(outdir, exist_ok=True)
        fig.write_html(f"{outdir}/feature_importance_lr.html")
        fig.write_image(
            f"{outdir}/feature_importance_lr.png", width=900, height=600, scale=2
        )
        # Plot results with Plotly
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("ROC Curve", "Prediction Distributions"),
            specs=[[{"type": "scatter"}, {"type": "histogram"}]],
        )

        # ROC Curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (AUC = {auc:.3f})",
                line=dict(color="royalblue", width=3),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Random classifier line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random classifier",
                line=dict(color="gray", width=2, dash="dash"),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Prediction distributions
        y_test_np = np.array(y_test)
        benign_probs = y_pred_proba[y_test_np == 0]
        harmful_probs = y_pred_proba[y_test_np == 1]

        fig.add_trace(
            go.Histogram(
                x=benign_probs,
                name="Benign",
                nbinsx=30,
                opacity=0.7,
                marker_color="#1f77b4",
                hovertemplate="Probability: %{x:.3f}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                x=harmful_probs,
                name="Harmful",
                nbinsx=30,
                opacity=0.7,
                marker_color="#ff7f0e",
                hovertemplate="Probability: %{x:.3f}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Logistic Regression Results (Layer {layer})",
            width=600,
            height=400,
            showlegend=True,
            title_x=0.5,
            title_font_size=28,
            font=dict(size=16),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=14),
            ),
        )

        # Update axes
        fig.update_xaxes(
            title_text="False Positive Rate",
            row=1,
            col=1,
            range=[0, 1],
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )
        fig.update_yaxes(
            title_text="True Positive Rate",
            row=1,
            col=1,
            range=[0, 1],
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )
        fig.update_xaxes(
            title_text="Predicted Probability of Harmful",
            row=1,
            col=2,
            range=[0, 1],
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=2,
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )

        # Make subplot titles larger
        fig.update_annotations(font_size=16)

        # Add grid
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=1
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=2
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=2
        )

        fig.write_html(f"{outdir}/logistic_regression_results.html")
        fig.write_image(
            f"{outdir}/logistic_regression_results.png", width=1200, height=500, scale=2
        )
        print(
            f"Results saved to: {outdir}/feature_importance_lr.html, {outdir}/feature_importance_lr.png, {outdir}/logistic_regression_results.html, {outdir}/logistic_regression_results.png"
        )

    # Save layer metrics to cache
    if layer_metrics:
        os.makedirs("cache", exist_ok=True)
        layer_metrics_cache_file = "cache/layer_metrics.json"
        import json

        with open(layer_metrics_cache_file, "w") as f:
            json.dump(layer_metrics, f, indent=2)
        print(
            f"Saved {len(layer_metrics)} layer metrics to cache: {layer_metrics_cache_file}"
        )

    # Create layerwise metrics plot
    if layer_metrics:
        create_layerwise_metrics_plot(layer_metrics)
    print("\n=== All Layers Complete ===")


if __name__ == "__main__":
    main()
