from sae_utils import (
    get_device,
    load_model_and_tokenizer,
    get_feature_activations_for_texts,
    load_jailbreak_dataset,
    list_available_layers,
    load_layer_important_features,
    load_layer_sae,
)
import warnings
import pandas as pd
import torch
import time

# Flag to control inclusion of zero-activation examples in HTML
INCLUDE_ZERO_ACTIVATION = False

warnings.filterwarnings("ignore")

# Set up
model_name = "meta-llama/Prompt-Guard-86M"
device = get_device()
model, tokenizer = load_model_and_tokenizer(model_name, device)

# Get all available layers
available_layers = list_available_layers()
print(f"Available layers: {available_layers}")

dataset = load_jailbreak_dataset(split="train")


# Prepare examples (same as before)
def get_examples():
    benign_examples = list(
        dataset.filter(lambda x: x["type"] == "benign").shuffle(seed=42)
    )[:100]
    jailbreak_examples = list(
        dataset.filter(lambda x: x["type"] == "jailbreak").shuffle(seed=42)
    )[:100]
    custom_df = pd.read_csv("data/custom.csv")
    custom_examples = [
        {"prompt": row["prompt"], "type": "custom"} for _, row in custom_df.iterrows()
    ]
    return benign_examples, jailbreak_examples, custom_examples


benign_examples, jailbreak_examples, custom_examples = get_examples()
all_texts = [
    ex["prompt"] for ex in jailbreak_examples + custom_examples + benign_examples
]

# Precompute logits/probs for all texts (shared across layers)
logits, probs = None, None


def get_logits_probs():
    global logits, probs
    if logits is None or probs is None:
        print("Computing logits and probabilities for all texts...")
        l, p = get_logits_and_probs(model, tokenizer, all_texts, device)
        logits, probs = l, p
    return logits, probs


# Utility functions for visualization


def get_valid_max_activation(activations, offset_mapping):
    """Compute max activation only for valid tokens (excluding padding)."""
    valid_activations = []
    for (start, end), activation in zip(offset_mapping, activations):
        # Skip padding tokens (start == end == 0)
        if not (start == end == 0):
            valid_activations.append(abs(activation.item()))
    if not valid_activations:
        return 0.0
    return max(valid_activations)


def activation_to_color(activation, max_activation):
    """Convert activation value to a color."""
    if max_activation == 0:
        return "rgba(0, 0, 0, 0)"
    abs_activation = abs(activation)
    intensity = abs_activation / max_activation
    return f"rgba(255, 0, 0, {intensity})"


def visualize_activations(text, activations, offset_mapping, max_activation):
    """Create HTML visualization of token activations, grouping consecutive tokens with the same color."""
    html_parts = []
    current_pos = 0
    last_color = None
    span_buffer = ""
    for (start, end), activation in zip(offset_mapping, activations):
        if start == end == 0:
            continue
        if start > current_pos:
            # Flush any buffered span
            if span_buffer:
                html_parts.append(
                    f'<span style="background-color: {last_color}">{span_buffer}</span>'
                )
                span_buffer = ""
                last_color = None
            html_parts.append(text[current_pos:start])
        token = text[start:end]
        color = activation_to_color(activation.item(), max_activation)
        if color == last_color:
            span_buffer += token
        else:
            if span_buffer:
                html_parts.append(
                    f'<span style="background-color: {last_color}">{span_buffer}</span>'
                )
            span_buffer = token
            last_color = color
        current_pos = end
    # Flush any remaining buffered span
    if span_buffer:
        html_parts.append(
            f'<span style="background-color: {last_color}">{span_buffer}</span>'
        )
    if current_pos < len(text):
        html_parts.append(text[current_pos:])
    return "".join(html_parts)


def get_logits_and_probs(model, tokenizer, texts, device, batch_size=8):
    """Compute logits and probabilities for a list of texts."""
    all_logits = []
    all_probs = []
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
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            all_logits.append(logits)
            all_probs.append(probs)
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    return all_logits, all_probs


def generate_example_html(
    examples_with_activation,
    examples_without_activation,
    activations,
    offset_mappings,
    offset_start_idx,
    category_name,
    show_no_activation=True,
    logits=None,
    probs=None,
    logits_offset=0,
):
    """Generate HTML for examples with and without activation."""
    html_parts = []
    # Examples with activation
    html_parts.append(f"<h3>{category_name} Examples (with activation)</h3>")
    for j, (example, idx) in enumerate(examples_with_activation):
        text = example["prompt"]
        example_activations = activations[idx]
        offset_mapping = offset_mappings[idx + offset_start_idx]
        max_activation = get_valid_max_activation(example_activations, offset_mapping)
        logit_str = ""
        prob_str = ""
        if logits is not None and probs is not None:
            logit_vals = logits[idx + logits_offset].tolist()
            prob_vals = probs[idx + logits_offset].tolist()
            prob_str = f"<div class='probs'>Probabilities: {[f'{v:.3f}' for v in prob_vals]}</div>"
        html_parts.append(f"""
        <div class="example">
            <div class="label">Example {j + 1} (max activation: {max_activation:.4f})</div>
            {logit_str}
            {prob_str}
            <div class="text">{visualize_activations(text, example_activations, offset_mapping, max_activation)}</div>
        </div>
        """)
    # Examples without activation
    if INCLUDE_ZERO_ACTIVATION and show_no_activation:
        html_parts.append(f"<h3>{category_name} Examples (no activation)</h3>")
        for j, (example, idx) in enumerate(examples_without_activation):
            text = example["prompt"]
            example_activations = activations[idx]
            offset_mapping = offset_mappings[idx + offset_start_idx]
            max_activation = get_valid_max_activation(
                example_activations, offset_mapping
            )
            no_activation_class = "no-activation" if show_no_activation else ""
            logit_str = ""
            prob_str = ""
            if logits is not None and probs is not None:
                logit_vals = logits[idx + logits_offset].tolist()
                prob_vals = probs[idx + logits_offset].tolist()
                logit_str = f"<div class='logits'>Logits: {[f'{v:.3f}' for v in logit_vals]}</div>"
                prob_str = f"<div class='probs'>Probabilities: {[f'{v:.3f}' for v in prob_vals]}</div>"
            html_parts.append(f"""
            <div class="example {no_activation_class}">
                <div class="label {no_activation_class}">Example {j + 1} (max activation: {max_activation:.4f})</div>
                {logit_str}
                {prob_str}
                <div class="text">{visualize_activations(text, example_activations, offset_mapping, max_activation)}</div>
            </div>
            """)
    return "".join(html_parts)


# Helper to load all layer info
def load_all_layer_info(top_n_features=10):
    layer_info = {}
    for layer in available_layers:
        print(f"Loading info for layer {layer}...")
        # Load important features
        features_json = load_layer_important_features(layer)
        important_features = features_json["important_features"][:top_n_features]
        # Load SAE
        ae = load_layer_sae(layer, device)
        # Extract feature indices
        feature_indices = [f["index"] for f in important_features]
        # Compute activations for all texts
        print(f"Computing activations for layer {layer}...")
        feature_activations, offset_mappings = get_feature_activations_for_texts(
            model,
            tokenizer,
            ae,
            all_texts,
            layer,
            device,
            feature_indices=feature_indices,
            return_offsets_mapping=True,
        )
        layer_info[layer] = {
            "important_features": important_features,
            "feature_activations": feature_activations,
            "offset_mappings": offset_mappings,
        }
    return layer_info


# Main visualization logic
def analyze_feature_activations_all_layers():
    layer_info = load_all_layer_info(top_n_features=10)
    logits, probs = get_logits_probs()
    print("Generating HTML...")
    start_time = time.time()
    # HTML header
    html_output = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .example {{ margin-bottom: 30px; padding: 10px; border: 1px solid #ccc; }}
            .label {{ font-weight: bold; margin-bottom: 10px; }}
            .text {{ line-height: 1.6; }}
            .no-activation {{ display: none; }}
            .toggle-container {{ margin-bottom: 20px; }}
            .stats {{ margin-bottom: 20px; color: #666; }}
            .feature-selector {{ margin-bottom: 20px; }}
            .feature-info {{ margin-bottom: 20px; padding: 10px; background-color: #f0f0f0; }}
            .feature-section {{ display: none; }}
            .feature-section.active {{ display: block; }}
            .layer-section {{ display: none; }}
            .layer-section.active {{ display: block; }}
        </style>
        <script>
            function toggleNoActivation() {{
                const checkbox = document.getElementById('showNoActivation');
                const elements = document.querySelectorAll('.no-activation');
                elements.forEach(el => {{
                    el.style.display = checkbox.checked ? 'block' : 'none';
                }});
            }}
            function showLayer(layerIndex) {{
                // Hide all layer sections
                const layerSections = document.querySelectorAll('.layer-section');
                layerSections.forEach(section => {{
                    section.classList.remove('active');
                }});
                // Show selected layer section
                const selectedLayer = document.getElementById('layer-' + layerIndex);
                if (selectedLayer) {{
                    selectedLayer.classList.add('active');
                }}
                // Update feature selector for this layer
                const featureSelectors = document.querySelectorAll('.featureSelector');
                featureSelectors.forEach(sel => {{ sel.style.display = 'none'; }});
                const thisSelector = document.getElementById('featureSelector-' + layerIndex);
                if (thisSelector) {{ thisSelector.style.display = 'inline'; }}
                // Show first feature by default
                showFeature(layerIndex, 0);
            }}
            function showFeature(layerIndex, featureIndex) {{
                // Hide all feature sections in this layer
                const sections = document.querySelectorAll('.feature-section.layer-' + layerIndex);
                sections.forEach(section => {{
                    section.classList.remove('active');
                }});
                // Show selected feature section
                const selectedSection = document.getElementById('feature-' + layerIndex + '-' + featureIndex);
                if (selectedSection) {{
                    selectedSection.classList.add('active');
                }}
                // Update selector value
                const selector = document.getElementById('featureSelector-' + layerIndex);
                if (selector) {{ selector.value = featureIndex; }}
            }}
        </script>
    </head>
    <body>
        <h1>SAE Feature Activations Visualization (All Layers)</h1>
        <p>Red background indicates activation.</p>
        <div class="layer-selector">
            <label for="layerSelector">Select Layer: </label>
            <select id="layerSelector" onchange="showLayer(this.value)">
                {"".join([f'<option value="{layer}">Layer {layer}</option>' for layer in available_layers])}
            </select>
        </div>
        <div class="toggle-container">
            <label>
                <input type="checkbox" id="showNoActivation" onchange="toggleNoActivation()">
                Show examples where feature doesn't fire
            </label>
        </div>
    """
    # For each layer, add a section
    for layer in available_layers:
        info = layer_info[layer]
        important_features = info["important_features"]
        feature_activations = info["feature_activations"]
        offset_mappings = info["offset_mappings"]
        html_output += f'<div id="layer-{layer}" class="layer-section {"active" if layer == available_layers[0] else ""}">'  # Only first layer active by default
        # Feature selector for this layer
        html_output += f'<div class="feature-selector"><label for="featureSelector-{layer}">Select Feature: </label>'
        html_output += f'<select class="featureSelector" id="featureSelector-{layer}" onchange="showFeature({layer}, this.value)" style="display: {"inline" if layer == available_layers[0] else "none"}">'  # Only first layer's selector visible by default
        for i, f in enumerate(important_features):
            html_output += f'<option value="{i}">Feature {f["index"]} (p={f["p_value"]:.6f}, coef={f["coefficient"]:.6f})</option>'
        html_output += "</select></div>"
        # For each feature, add a section
        for i, feature in enumerate(important_features):
            feature_idx = feature["index"]
            feature_name = feature["name"]
            p_value = feature["p_value"]
            coefficient = feature["coefficient"]
            feature_acts = feature_activations[feature_idx]  # [batch_size, seq_len]
            # Vectorized activation mask: shape [num_examples]
            # Compute max(abs(activation)) over valid tokens for each example
            acts_tensor = (
                torch.tensor(feature_acts)
                if not isinstance(feature_acts, torch.Tensor)
                else feature_acts
            )
            # Build mask for valid tokens (not padding)
            valid_token_mask = []
            for offsets in offset_mappings:
                valid = [(start != 0 or end != 0) for (start, end) in offsets]
                valid_token_mask.append(valid)
            valid_token_mask = torch.tensor(valid_token_mask, device=acts_tensor.device)
            abs_acts = acts_tensor.abs()
            # Set activations for padding tokens to 0
            abs_acts_masked = abs_acts * valid_token_mask
            # Max activation per example
            max_acts = abs_acts_masked.max(dim=1).values
            # Threshold for activation
            threshold = 0.01
            is_activated = max_acts > threshold
            # Split by category
            n_jb = len(jailbreak_examples)
            n_cust = len(custom_examples)
            n_ben = len(benign_examples)
            # Indices for each category
            jb_idx = torch.arange(0, n_jb)
            cust_idx = torch.arange(n_jb, n_jb + n_cust)
            ben_idx = torch.arange(n_jb + n_cust, n_jb + n_cust + n_ben)
            # Masks for each category
            jb_activated = is_activated[jb_idx]
            cust_activated = is_activated[cust_idx]
            ben_activated = is_activated[ben_idx]
            # Category-specific activations
            jailbreak_acts = acts_tensor[:n_jb]
            custom_acts = acts_tensor[n_jb : n_jb + n_cust]
            benign_acts = acts_tensor[n_jb + n_cust :]
            # Prepare (example, idx) tuples for each group (fix index math)
            jb_idx = jb_idx.cpu() if hasattr(jb_idx, "cpu") else torch.tensor(jb_idx)
            cust_idx = (
                cust_idx.cpu() if hasattr(cust_idx, "cpu") else torch.tensor(cust_idx)
            )
            ben_idx = (
                ben_idx.cpu() if hasattr(ben_idx, "cpu") else torch.tensor(ben_idx)
            )
            jailbreak_with_activation = [
                (jailbreak_examples[j], j) for j in jb_idx[jb_activated.cpu()].tolist()
            ]
            jailbreak_without_activation = [
                (jailbreak_examples[j], j)
                for j in jb_idx[(~jb_activated).cpu()].tolist()
            ]
            custom_with_activation = [
                (custom_examples[j - n_jb], j - n_jb)
                for j in cust_idx[cust_activated.cpu()].tolist()
            ]
            custom_without_activation = [
                (custom_examples[j - n_jb], j - n_jb)
                for j in cust_idx[(~cust_activated).cpu()].tolist()
            ]
            benign_with_activation = [
                (benign_examples[j - n_jb - n_cust], j - n_jb - n_cust)
                for j in ben_idx[ben_activated.cpu()].tolist()
            ]
            benign_without_activation = [
                (benign_examples[j - n_jb - n_cust], j - n_jb - n_cust)
                for j in ben_idx[(~ben_activated).cpu()].tolist()
            ]
            html_output += f'<div id="feature-{layer}-{i}" class="feature-section layer-{layer} {"active" if i == 0 else ""}">'  # Only first feature active by default
            html_output += f"""
            <div class="feature-info">
                <h2>Layer {layer} - Feature {feature_idx} ({feature_name})</h2>
                <p><strong>P-value:</strong> {p_value:.6f} | <strong>Coefficient:</strong> {coefficient:.6f}</p>
                <p><strong>Custom examples:</strong> {len(custom_with_activation)} with activation, {len(custom_without_activation)} without activation</p>
                <p><strong>Jailbreak examples:</strong> {len(jailbreak_with_activation)} with activation, {len(jailbreak_without_activation)} without activation</p>
                <p><strong>Benign examples:</strong> {len(benign_with_activation)} with activation, {len(benign_without_activation)} without activation</p>
            </div>
            """
            # Generate HTML for each category using the helper function
            html_output += generate_example_html(
                custom_with_activation,
                custom_without_activation,
                custom_acts,
                offset_mappings,
                len(jailbreak_examples),
                "Custom",
                logits=logits,
                probs=probs,
                logits_offset=len(jailbreak_examples),
            )
            html_output += generate_example_html(
                jailbreak_with_activation,
                jailbreak_without_activation,
                jailbreak_acts,
                offset_mappings,
                0,
                "Jailbreak",
                logits=logits,
                probs=probs,
                logits_offset=0,
            )
            html_output += generate_example_html(
                benign_with_activation,
                benign_without_activation,
                benign_acts,
                offset_mappings,
                len(jailbreak_examples) + len(custom_examples),
                "Benign",
                logits=logits,
                probs=probs,
                logits_offset=len(jailbreak_examples) + len(custom_examples),
            )
            html_output += "</div>"  # Close feature section
        html_output += "</div>"  # Close layer section
    html_output += "</body></html>"
    with open("feature_visualization.html", "w") as f:
        f.write(html_output)
    print(
        f"Visualization saved to feature_visualization.html. HTML generation took {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    analyze_feature_activations_all_layers()
