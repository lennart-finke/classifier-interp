import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import plotly.graph_objects as go
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import os

# Dataset configs (copied from prompt_dataset.py)
dataset_configs = [
    {
        "name": "allenai/wildjailbreak",
        "config": "train",
        "split": "train",
        "text_column": "adversarial",
        "subsample_ratio": 1.0,
        "delimiter": "\t",
        "keep_default_na": False,
    },
    {
        "name": "Open-Orca/OpenOrca",
        "split": "train",
        "text_column": "question",
        "subsample_ratio": 0.05,
    },
    {
        "name": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_12_25",
        "split": "train",
        "text_column": "prompt",
        "subsample_ratio": 1.0,
    },
    {
        "name": "EddyLuo/JailBreakV_28K",
        "config": "JailBreakV_28K",
        "split": "JailBreakV_28K",
        "text_column": "jailbreak_query",
        "subsample_ratio": 1.0,
    },
]

# Ground truth mapping
ground_truth = {
    "Open-Orca/OpenOrca": 0,  # not harmful
    "TrustAIRLab/in-the-wild-jailbreak-prompts": 1,  # harmful
    "allenai/wildjailbreak": 1,  # harmful
    "EddyLuo/JailBreakV_28K": 1,  # harmful
}

# Model setup
model_name = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    output_hidden_states=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 128
MAX_SAMPLES = 10000

CACHE_CSV = "cache/classified_prompts_with_probs.csv"

# Check if cache exists and is complete
use_cache = False
if os.path.exists(CACHE_CSV):
    cache_df = pd.read_csv(CACHE_CSV)
    # Check if all datasets are present and have at least one row
    cache_datasets = set(cache_df["dataset"].unique())
    config_datasets = set([c["name"] for c in dataset_configs])
    if config_datasets.issubset(cache_datasets):
        use_cache = True

results = []
probs_cache = []

if use_cache:
    print("Using cached probabilities from CSV...")
    for config in dataset_configs:
        name = config["name"]
        df = cache_df[cache_df["dataset"] == name]
        n_total = len(df)
        n_benign = (
            df["prob_harmless"] > df[["prob_injection", "prob_jailbreak"]].max(axis=1)
        ).sum()
        n_harmful = n_total - n_benign
        pct_benign = n_benign / n_total if n_total > 0 else 0
        pct_harmful = n_harmful / n_total if n_total > 0 else 0
        ci_benign = proportion_confint(n_benign, n_total, alpha=0.05, method="wilson")
        ci_harmful = proportion_confint(n_harmful, n_total, alpha=0.05, method="wilson")
        results.append(
            {
                "name": name,
                "benign": n_benign,
                "harmful": n_harmful,
                "n": n_total,
                "pct_benign": pct_benign,
                "pct_harmful": pct_harmful,
                "ci_benign_low": ci_benign[0],
                "ci_benign_high": ci_benign[1],
                "ci_harmful_low": ci_harmful[0],
                "ci_harmful_high": ci_harmful[1],
            }
        )
else:
    for config in dataset_configs:
        name = config["name"]
        text_column = config.get("text_column", "text")
        print(f"\nProcessing {name} [{text_column}] ...")
        # Load dataset
        dataset_kwargs = {
            "split": config["split"],
            "streaming": False,
        }
        if "delimiter" in config:
            dataset_kwargs["delimiter"] = config["delimiter"]
        if "keep_default_na" in config:
            dataset_kwargs["keep_default_na"] = config["keep_default_na"]
        if "config" in config:
            dataset = load_dataset(config["name"], config["config"], **dataset_kwargs)
        else:
            dataset = load_dataset(config["name"], **dataset_kwargs)
        # Special filter for wildjailbreak
        if name == "allenai/wildjailbreak":
            dataset = [
                x for x in dataset if x.get("data_type") == "adversarial_harmful"
            ]
        # Get texts
        texts = [x[text_column] for x in dataset if x.get(text_column)]
        # Subsample
        if len(texts) > MAX_SAMPLES:
            random.seed(42)
            texts = random.sample(texts, MAX_SAMPLES)
        else:
            texts = texts[:MAX_SAMPLES]
        # Run classifier
        harmful_preds = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Harmful if prob(class 1) + prob(class 2) > 0.5
            harmful = (probs[:, 1] + probs[:, 2]) > 0.5
            harmful_preds.extend(harmful.astype(int))
            # Cache probabilities for each prompt in the batch
            for j, prompt in enumerate(batch):
                probs_cache.append(
                    {
                        "dataset": name,
                        "prompt": prompt,
                        "prob_harmless": probs[j, 0],
                        "prob_injection": probs[j, 1],
                        "prob_jailbreak": probs[j, 2],
                    }
                )
    # Count benign/harmful
    n_harmful = int(np.sum(harmful_preds))
    n_benign = len(harmful_preds) - n_harmful
    n_total = len(harmful_preds)
    # Calculate percentages
    pct_benign = n_benign / n_total if n_total > 0 else 0
    pct_harmful = n_harmful / n_total if n_total > 0 else 0
    # Binomial confidence intervals (Wilson)
    ci_benign = proportion_confint(n_benign, n_total, alpha=0.05, method="wilson")
    ci_harmful = proportion_confint(n_harmful, n_total, alpha=0.05, method="wilson")
    results.append(
        {
            "name": name,
            "benign": n_benign,
            "harmful": n_harmful,
            "n": n_total,
            "pct_benign": pct_benign,
            "pct_harmful": pct_harmful,
            "ci_benign_low": ci_benign[0],
            "ci_benign_high": ci_benign[1],
            "ci_harmful_low": ci_harmful[0],
            "ci_harmful_high": ci_harmful[1],
        }
    )
    print(f"{name}: benign={n_benign}, harmful={n_harmful}, n={n_total}")

if not use_cache:
    # Save probabilities to CSV only if new data was computed
    probs_df = pd.DataFrame(probs_cache)
    probs_df.to_csv(CACHE_CSV, index=False)

# Plot with plotly (percentages + error bars)
labels = [r["name"] for r in results]
benign_pcts = [r["pct_benign"] * 100 for r in results]
harmful_pcts = [r["pct_harmful"] * 100 for r in results]
benign_err = (
    [(r["pct_benign"] - r["ci_benign_low"]) * 100 for r in results],
    [(r["ci_benign_high"] - r["pct_benign"]) * 100 for r in results],
)
harmful_err = (
    [(r["pct_harmful"] - r["ci_harmful_low"]) * 100 for r in results],
    [(r["ci_harmful_high"] - r["pct_harmful"]) * 100 for r in results],
)

fig = go.Figure()
for i, r in enumerate(results):
    name = r["name"]
    benign_pct = r["pct_benign"] * 100
    harmful_pct = r["pct_harmful"] * 100
    benign_err = [
        (r["pct_benign"] - r["ci_benign_low"]) * 100,
        (r["ci_benign_high"] - r["pct_benign"]) * 100,
    ]
    # Plotly-like colors
    benign_color = "#7fba7f"  # Softer green
    harmful_color = "#d62728"  # Red
    # Reverse for Open-Orca
    if name == "Open-Orca/OpenOrca":
        benign_color = "#d62728"  # Red
        harmful_color = "#7fba7f"  # Softer green
    accuracy = max(r["pct_benign"], r["pct_harmful"]) * 100
    name_with_acc = f"{name}\n(Acc: {accuracy:.1f}%)"
    # Benign bar
    fig.add_trace(
        go.Bar(
            y=[name_with_acc],
            x=[benign_pct],
            name="Benign" if i == 0 else None,
            orientation="h",
            marker_color=benign_color,
            showlegend=(i == 0),
            width=0.6,
        )
    )
    # Harmful bar
    fig.add_trace(
        go.Bar(
            y=[name_with_acc],
            x=[harmful_pct],
            name="Harmful" if i == 0 else None,
            orientation="h",
            marker_color=harmful_color,
            showlegend=(i == 0),
            width=0.6,
        )
    )
    # Add error bar as a scatter marker at the right edge of the benign bar
    fig.add_trace(
        go.Scatter(
            y=[name_with_acc],
            x=[benign_pct],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=1),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[benign_err[1]],
                arrayminus=[benign_err[0]],
                thickness=2,
                color="black",
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
fig.update_layout(
    barmode="stack",
    xaxis_title="Percentage of Classifications (%)",
    yaxis_title="Dataset (Accuracy)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400 + 40 * len(results),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
    ),
)
fig.write_image("figures/classifier_stacked_bar.png", scale=3)
