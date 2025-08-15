from sae_utils import (
    get_device,
    load_model_and_tokenizer,
    load_sae,
    load_jailbreak_dataset,
    list_available_layers,
    get_sae_checkpoint_path,
)
import umap
import umap.aligned_umap
import plotly.express as px
import pandas as pd
import warnings
import torch
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objs as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Set up
model_name = "meta-llama/Prompt-Guard-86M"
device = get_device()
model, tokenizer = load_model_and_tokenizer(model_name, device)
# Discover available layers (0-12 or as found in cache/sae-test)
LAYER_RANGE = list_available_layers("sae-test")


train_dataset = load_jailbreak_dataset(split="train")
test_dataset = load_jailbreak_dataset(split="test")

# Cache file paths
CACHE_DIR = "cache"
# TRAIN_ACTIVATIONS_CACHE_FILE = os.path.join(
#     CACHE_DIR, f"train_activations_layer_{TARGET_LAYER}.pt"
# )
# TEST_ACTIVATIONS_CACHE_FILE = os.path.join(
#     CACHE_DIR, f"test_activations_layer_{TARGET_LAYER}.pt"
# )

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
for layer in LAYER_RANGE:
    os.makedirs(f"figures/layer_{layer}", exist_ok=True)


def prepare_train_data():
    """Prepare the train dataset and extract texts and labels."""
    # Convert dataset to list and extract texts and labels
    train_list = list(train_dataset)
    texts = [ex["prompt"] for ex in train_list]
    labels = [
        1 if ex["type"] == "jailbreak" else 0 for ex in train_list
    ]  # 1 for harmful, 0 for benign
    label_names = [
        "Benign" if ex["type"] != "jailbreak" else "Harmful" for ex in train_list
    ]

    print(f"Train dataset size: {len(texts)}")
    print(f"Number of benign examples: {labels.count(0)}")
    print(f"Number of jailbreak examples: {labels.count(1)}")

    return texts, labels, label_names


def prepare_test_data():
    """Prepare the test dataset and extract texts and labels."""
    # Convert dataset to list and extract texts and labels
    test_list = list(test_dataset)
    texts = [ex["prompt"] for ex in test_list]
    labels = [
        1 if ex["type"] == "jailbreak" else 0 for ex in test_list
    ]  # 1 for harmful, 0 for benign
    label_names = [
        "Benign" if ex["type"] != "jailbreak" else "Harmful" for ex in test_list
    ]

    print(f"Test dataset size: {len(texts)}")
    print(f"Number of benign examples: {labels.count(0)}")
    print(f"Number of jailbreak examples: {labels.count(1)}")

    return texts, labels, label_names


def extract_all_layer_activations(texts, split_name):
    import torch

    print(f"Getting activations for all layers for {split_name} set...")
    batch_size = 8
    all_activations_cat = {}
    for layer in LAYER_RANGE:
        cache_file = os.path.join(
            CACHE_DIR, f"{split_name}_activations_layer_{layer}.pt"
        )
        if os.path.exists(cache_file):
            print(f"Loading activations for layer {layer} from cache: {cache_file}")
            all_activations_cat[layer] = torch.load(cache_file, map_location="cpu")
        else:
            print(f"No cache for layer {layer}, computing activations...")
            activations = []
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
                acts = outputs.hidden_states[layer].mean(dim=1)
                activations.append(acts.cpu())
            activations_cat = torch.cat(activations, dim=0)
            all_activations_cat[layer] = activations_cat
            torch.save(activations_cat, cache_file)
            print(f"Saved activations for layer {layer} to cache: {cache_file}")
    return all_activations_cat


def apply_umap(features, n_components=2, n_neighbors=15, min_dist=0.5, random_state=42):
    """Apply UMAP dimensionality reduction."""
    print(f"Applying UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean",
    )

    embeddings = reducer.fit_transform(features)
    return embeddings


def create_umap_plot(
    embeddings, labels, label_names, texts, title="UMAP Clustering", is_3d=False
):
    """Create an interactive plotly plot of UMAP embeddings."""

    # Format texts for hover display by inserting line breaks every n characters
    formatted_texts = []
    chars_per_line = 80  # Insert line break every 80 characters
    for text in texts:
        # Insert line breaks every n characters
        formatted_text = ""
        for i in range(0, len(text), chars_per_line):
            formatted_text += text[i : i + chars_per_line] + "<br>"
        # Remove the last <br> if it exists
        if formatted_text.endswith("<br>"):
            formatted_text = formatted_text[:-4]
        formatted_texts.append(formatted_text)

    # Create DataFrame for plotly
    if is_3d:
        df = pd.DataFrame(
            {
                "UMAP1": embeddings[:, 0],
                "UMAP2": embeddings[:, 1],
                "UMAP3": embeddings[:, 2],
                "Label": label_names,
                "Label_Num": labels,
                "Sample_Prompt": formatted_texts,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "UMAP1": embeddings[:, 0],
                "UMAP2": embeddings[:, 1],
                "Label": label_names,
                "Label_Num": labels,
                "Sample_Prompt": formatted_texts,
            }
        )

    # Create the main scatter plot
    if is_3d:
        fig = px.scatter_3d(
            df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="Label",
            color_discrete_map={"Benign": "#1f77b4", "Harmful": "#ff7f0e"},
            title=title,
            labels={
                "UMAP1": "UMAP Component 1",
                "UMAP2": "UMAP Component 2",
                "UMAP3": "UMAP Component 3",
            },
            hover_data=["Label", "Sample_Prompt"],
        )

        # Update hover template for 3D
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "UMAP1: %{x:.3f}<br>"
            + "UMAP2: %{y:.3f}<br>"
            + "UMAP3: %{z:.3f}<br>"
            + "<b>Prompt:</b><br>%{customdata[1]}"
            + "<extra></extra>"
        )
    else:
        fig = px.scatter(
            df,
            color="Label",
            color_discrete_map={"Benign": "#1f77b4", "Harmful": "#ff7f0e"},
            title=title,
            labels={"UMAP1": "UMAP Component 1", "UMAP2": "UMAP Component 2"},
            hover_data=["Label", "Sample_Prompt"],
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        # Update hover template for 2D
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "UMAP1: %{x:.3f}<br>"
            + "UMAP2: %{y:.3f}<br>"
            + "<b>Prompt:</b><br>%{customdata[1]}"
            + "<extra></extra>"
        )

    # Update layout for better appearance
    if is_3d:
        fig.update_layout(
            width=900,
            height=700,
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            legend_title_text="Prompt Type",
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            ),
            scene=dict(
                xaxis_title="UMAP Component 1",
                yaxis_title="UMAP Component 2",
                zaxis_title="UMAP Component 3",
            ),
        )
    else:
        fig.update_layout(
            width=800,
            height=600,
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            legend_title_text="Prompt Type",
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            ),
        )

    # Update traces for better point appearance
    fig.update_traces(marker=dict(size=6, opacity=0.7), selector=dict(type="scatter"))

    return fig


def analyze_clustering_quality(embeddings, labels):
    """Analyze the quality of clustering by computing separation metrics."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # Compute clustering quality metrics
    silhouette = silhouette_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)

    print("\nClustering Quality Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")

    return silhouette, calinski_harabasz


def evaluate_knn_classification(
    train_features, train_labels, test_features, test_labels, feature_type="SAE"
):
    """Evaluate KNN classification accuracy on test set."""
    print(f"\n=== KNN Classification Evaluation ({feature_type}) ===")

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(train_features, train_labels)

    # Predict on test set
    test_predictions = knn.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, test_predictions)

    print(f"KNN Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_predictions, target_names=["Benign", "Harmful"]
        )
    )

    return accuracy


def main():
    print("=== AlignedUMAP (All Layers) and Cluster Quality Evolution ===")
    train_texts, train_labels, train_label_names = prepare_train_data()
    test_texts, test_labels, test_label_names = prepare_test_data()

    # Extract and cache activations for all layers
    train_acts = extract_all_layer_activations(train_texts, "train")
    test_acts = extract_all_layer_activations(test_texts, "test")

    # Extract SAE features for all layers
    feature_list = []
    test_feature_list = []
    for layer in LAYER_RANGE:
        checkpoint_path = get_sae_checkpoint_path(layer, base_dir="sae-test")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for layer {layer}, skipping.")
            continue
        ae = load_sae(checkpoint_path, device)
        train_tensor = train_acts[layer]
        if not isinstance(train_tensor, torch.Tensor):
            train_tensor = torch.tensor(train_tensor)
        features = ae.encode(train_tensor.to(device)).detach().cpu().numpy()
        feature_list.append(features)
        # Test SAE features
        test_tensor = test_acts[layer]
        if not isinstance(test_tensor, torch.Tensor):
            test_tensor = torch.tensor(test_tensor)
        test_features = ae.encode(test_tensor.to(device)).detach().cpu().numpy()
        test_feature_list.append(test_features)
    n_layers = len(feature_list)
    n_samples = feature_list[0].shape[0]

    # Build relations: identity mapping for all pairs
    relations = [{i: i for i in range(n_samples)} for _ in range(n_layers - 1)]

    # Run AlignedUMAP
    print("Fitting AlignedUMAP across all layers...")
    aligned_mapper = umap.aligned_umap.AlignedUMAP(
        n_neighbors=25, min_dist=1, n_components=2, random_state=42
    ).fit(feature_list, relations=relations)
    embeddings = (
        aligned_mapper.embeddings_
    )  # List of [n_samples, 2] arrays, one per layer

    # Compute cluster quality (silhouette) for each layer
    from sklearn.metrics import silhouette_score

    silhouette_scores = []
    umap_knn_accuracies = []
    for i, emb in enumerate(embeddings):
        score = silhouette_score(emb, train_labels)
        silhouette_scores.append(score)
        # Save embedding plot for this layer with prompt text in hover
        df = pd.DataFrame(
            {
                "UMAP1": emb[:, 0],
                "UMAP2": emb[:, 1],
                "Label": ["Benign" if l == 0 else "Harmful" for l in train_labels],
                "Prompt": train_texts,
            }
        )
        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color="Label",
            title=f"AlignedUMAP Embedding Layer {LAYER_RANGE[i]} (Silhouette: {score:.3f})",
            labels={"UMAP1": "UMAP1", "UMAP2": "UMAP2"},
            hover_data={"Label": True, "Prompt": True},
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br><b>Prompt:</b><br>%{customdata[1]}<extra></extra>"
        )
        # Also write as html
        fig.write_html(
            f"figures/layer_{LAYER_RANGE[i]}/aligned_umap_embedding_layer_{LAYER_RANGE[i]}.html"
        )
        fig.write_image(
            f"figures/layer_{LAYER_RANGE[i]}/aligned_umap_embedding_layer_{LAYER_RANGE[i]}.png",
            width=800,
            height=600,
        )

    # Per-layer UMAP + kNN test accuracy
    print("\n=== Per-layer UMAP kNN Test Accuracy ===")
    for i, layer in enumerate(LAYER_RANGE[: len(feature_list)]):
        train_features = feature_list[i]
        test_features = test_feature_list[i]
        # Fit UMAP on train, transform both train and test
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=1,
            random_state=42,
            metric="euclidean",
        )
        train_umap = reducer.fit_transform(train_features)
        test_umap = reducer.transform(test_features)
        # kNN
        acc = evaluate_knn_classification(
            train_umap,
            train_labels,
            test_umap,
            test_labels,
            feature_type=f"UMAP-SAE Layer {layer}",
        )
        umap_knn_accuracies.append(acc)
        print(f"Layer {layer}: UMAP kNN Test Accuracy = {acc:.4f}")

    # Plot cluster quality evolution (Plotly, dual y-axis)
    layers = LAYER_RANGE[: len(silhouette_scores)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="royalblue"),
            marker=dict(symbol="circle", size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=umap_knn_accuracies,
            mode="lines+markers",
            name="UMAP kNN Test Accuracy",
            line=dict(color="orange"),
            marker=dict(symbol="square", size=8),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Cluster Quality (Silhouette) and UMAP kNN Test Accuracy Across Layers",
        xaxis_title="Layer",
        width=900,
        height=600,
        legend=dict(
            title="Metric",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=60, b=60),
    )
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=False)
    fig.update_yaxes(
        title_text="UMAP kNN Test Accuracy", secondary_y=True, range=[0, 1]
    )
    fig.write_html("figures/aligned_umap_cluster_quality.html")
    fig.write_image("figures/aligned_umap_cluster_quality.png", width=900, height=600)


if __name__ == "__main__":
    main()
