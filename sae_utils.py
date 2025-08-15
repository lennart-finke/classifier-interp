import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dictionary_learning import AutoEncoder
from datasets import load_dataset
import glob
import os


def get_device():
    """Return the best available torch device."""
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model_and_tokenizer(model_name, device):
    """Load HuggingFace model and tokenizer, move model to device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_sae(checkpoint_path, device):
    """Load the trained SAE from checkpoint."""
    ae = AutoEncoder.from_pretrained(checkpoint_path, device=device)
    return ae


def extract_layer_from_checkpoint(checkpoint_path):
    """Extract the layer number from the checkpoint path."""
    layer_match = re.search(r"layer_(\d+)_", checkpoint_path)
    if layer_match:
        return int(layer_match.group(1))
    else:
        raise ValueError(
            f"Could not extract layer number from checkpoint path: {checkpoint_path}"
        )


def get_activations(
    model,
    tokenizer,
    texts,
    target_layer,
    device,
    batch_size=8,
    mean_sequence=True,
    return_offsets_mapping=False,
):
    """
    Get hidden states from the model for a batch of texts.
    If mean_sequence is True, returns [batch_size, hidden_size].
    If False, returns [batch_size, seq_len, hidden_size].
    If return_offsets_mapping is True, also returns offset mappings (for single text input).
    """
    if isinstance(texts, str):
        texts = [texts]
    all_activations = []
    all_offsets = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=return_offsets_mapping,
        )
        offset_mapping = None
        if return_offsets_mapping:
            offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[target_layer]
        if mean_sequence:
            activations = activations.mean(dim=1)  # [batch_size, hidden_size]
        all_activations.append(activations)
        if return_offsets_mapping and offset_mapping is not None:
            all_offsets.extend(list(offset_mapping))
    activations_cat = torch.cat(all_activations, dim=0)
    if return_offsets_mapping:
        return activations_cat, all_offsets
    return activations_cat


def get_activations_with_sae(
    model,
    tokenizer,
    ae,
    texts,
    target_layer,
    device,
    batch_size=8,
    return_offsets_mapping=False,
):
    """
    Get hidden states and SAE features for a batch of texts efficiently.
    Returns both raw activations and SAE features.

    Returns:
        - activations: [batch_size, seq_len, hidden_size]
        - sae_features: [batch_size, seq_len, n_features]
        - offset_mappings: list of offset mappings (if return_offsets_mapping=True)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Process all texts in a single batch to ensure consistent sequence lengths
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        return_offsets_mapping=return_offsets_mapping,
    )
    offset_mapping = None
    if return_offsets_mapping:
        offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[
            target_layer
        ]  # [batch_size, seq_len, hidden_size]
        sae_features = ae.encode(activations)  # [batch_size, seq_len, n_features]

    if return_offsets_mapping and offset_mapping is not None:
        return activations, sae_features, offset_mapping
    else:
        return activations, sae_features


def get_feature_activations_for_texts(
    model,
    tokenizer,
    ae,
    texts,
    target_layer,
    device,
    feature_indices=None,
    batch_size=8,
    return_offsets_mapping=False,
):
    """
    Get SAE feature activations for specific features across multiple texts efficiently.

    Args:
        feature_indices: List of feature indices to extract. If None, returns all features.
        return_offsets_mapping: Whether to return offset mappings for tokenization.

    Returns:
        - feature_activations: dict mapping feature_idx to [batch_size, seq_len] tensor
        - offset_mappings: list of offset mappings (if return_offsets_mapping=True)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Get all activations and SAE features at once
    result = get_activations_with_sae(
        model,
        tokenizer,
        ae,
        texts,
        target_layer,
        device,
        batch_size,
        return_offsets_mapping=return_offsets_mapping,
    )

    if return_offsets_mapping:
        # result should be a tuple of length 3
        if len(result) == 3:
            activations, sae_features, offset_mappings = result
        else:
            raise ValueError(
                "Expected 3 outputs (activations, sae_features, offset_mappings) when return_offsets_mapping=True"
            )
    else:
        # result should be a tuple of length 2
        if len(result) == 2:
            activations, sae_features = result
            offset_mappings = None
        else:
            raise ValueError(
                "Expected 2 outputs (activations, sae_features) when return_offsets_mapping=False"
            )

    # Extract specific features
    feature_activations = {}
    if feature_indices is None:
        # Return all features
        for i in range(sae_features.shape[-1]):
            feature_activations[i] = sae_features[:, :, i]  # [batch_size, seq_len]
    else:
        # Return only specified features
        for feature_idx in feature_indices:
            if feature_idx < sae_features.shape[-1]:
                feature_activations[feature_idx] = sae_features[
                    :, :, feature_idx
                ]  # [batch_size, seq_len]

    if return_offsets_mapping:
        return feature_activations, offset_mappings
    return feature_activations


def get_single_feature_activation(
    activations,
    sae_features,
    feature_idx,
    text_idx=0,
):
    """
    Extract activation for a single feature from pre-computed activations.

    Args:
        activations: [batch_size, seq_len, hidden_size] tensor
        sae_features: [batch_size, seq_len, n_features] tensor
        feature_idx: Index of the feature to extract
        text_idx: Index of the text in the batch (default 0)

    Returns:
        feature_activation: [seq_len] tensor for the specified feature and text
    """
    if feature_idx >= sae_features.shape[-1]:
        raise ValueError(
            f"Feature index {feature_idx} out of range. Max: {sae_features.shape[-1] - 1}"
        )

    if text_idx >= sae_features.shape[0]:
        raise ValueError(
            f"Text index {text_idx} out of range. Max: {sae_features.shape[0] - 1}"
        )

    return sae_features[text_idx, :, feature_idx]  # [seq_len]


def load_jailbreak_dataset(split="train"):
    """Load the jailbreak dataset from HuggingFace Datasets."""
    return load_dataset("jackhhao/jailbreak-classification", split=split)


def list_available_layers(cache_dir="cache"):
    """Return sorted list of available layer numbers (as ints) in the cache directory."""
    layer_dirs = glob.glob(f"{cache_dir}/layer_*")
    layers = []
    for d in layer_dirs:
        match = re.search(r"layer_(\d+)", d)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)


def load_layer_important_features(layer, cache_dir="cache"):
    """Load important features for a given layer from cache/layer_{layer}/important_features.json."""
    import json

    path = f"{cache_dir}/layer_{layer}/important_features.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Important features file not found for layer {layer}: {path}"
        )
    with open(path, "r") as f:
        return json.load(f)


def get_sae_checkpoint_path(layer, base_dir="sae-test"):
    """Return the SAE checkpoint path for a given layer (assumes standard naming)."""
    return f"{base_dir}/layer_{layer}/trainer_0/checkpoints/ae_100000.pt"


def load_layer_sae(layer, device, base_dir="sae-test"):
    """Load the SAE for a given layer using the standard checkpoint path."""
    checkpoint_path = get_sae_checkpoint_path(layer, base_dir)
    return load_sae(checkpoint_path, device)
