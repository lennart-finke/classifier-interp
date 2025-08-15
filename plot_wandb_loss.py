import pandas as pd
import wandb
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api = wandb.Api()

# Get entity and project from environment variables
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")

if not entity or not project:
    raise ValueError("WANDB_ENTITY and WANDB_PROJECT must be set in .env file")

# Project is specified by <entity/project-name>
runs = list(api.runs(f"{entity}/{project}"))

# Only keep the latest 13 runs
runs = sorted(runs, key=lambda r: r.created_at, reverse=True)[:13]

# Assert there is one run for each layer in range(13)
layers = [run.config.get("layer", run.name) for run in runs]
try:
    layers_int = [int(l) for l in layers]
except Exception:
    raise ValueError(f"Layer values could not be converted to int: {layers}")
assert sorted(layers_int) == list(range(13)), (
    f"Expected one run for each layer 0-12, got: {layers_int}"
)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("data/wandb_loss.csv")

fig_mse = go.Figure()
fig_sparsity = go.Figure()

for run in runs:
    layer = run.config.get("layer", run.name)
    # Download the full history for this run
    history = run.history(keys=["_step", "mse_loss", "sparsity_loss"], pandas=True)
    if "mse_loss" in history and "sparsity_loss" in history:
        fig_mse.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["mse_loss"],
                mode="lines",
                name=f"Layer {layer}",
            )
        )
        fig_sparsity.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["sparsity_loss"],
                mode="lines",
                name=f"Layer {layer}",
            )
        )

fig_mse.update_layout(
    title="MSE Loss Curve for SAE Training (All Layers)",
    xaxis_title="Step",
    yaxis_title="MSE Loss",
    legend_title="Layer",
    xaxis_type="log",
)
fig_sparsity.update_layout(
    title="Sparsity Loss Curve for SAE Training (All Layers)",
    xaxis_title="Step",
    yaxis_title="Sparsity Loss",
    legend_title="Layer",
    xaxis_type="log",
)

fig_mse.write_image("figures/mse_loss_curve.png")
fig_sparsity.write_image("figures/sparsity_loss_curve.png")

# Prepare colormap for 13 layers
layer_order = sorted(set(int(run.config.get("layer", run.name)) for run in runs))
colormap = plt.colormaps["viridis"]
layer_to_color = {
    layer: f"rgba{tuple(int(255 * c) for c in colormap(i / 12)[:3]) + (1,)}"
    for i, layer in enumerate(layer_order)
}

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "MSE Loss",
        "$L_1$ Loss ",
        "$L_0$ Loss",
        "Fraction Variance Explained",
    ),
    horizontal_spacing=0.08,  # default is 0.2
    vertical_spacing=0.08,  # default is 0.3
)

for run in runs:
    layer = int(run.config.get("layer", run.name))
    color = layer_to_color[layer]
    # Download the full history for this run
    history = run.history(
        keys=["_step", "mse_loss", "sparsity_loss", "l0", "frac_variance_explained"],
        pandas=True,
    )
    if "mse_loss" in history and "sparsity_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["mse_loss"],
                mode="lines",
                name=f"Layer {layer}",
                line=dict(color=color),
                legendgroup=f"Layer {layer}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["sparsity_loss"],
                mode="lines",
                name=f"Layer {layer}",
                line=dict(color=color),
                legendgroup=f"Layer {layer}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    if "l0" in history:
        fig.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["l0"],
                mode="lines",
                name=f"Layer {layer}",
                line=dict(color=color),
                legendgroup=f"Layer {layer}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if "frac_variance_explained" in history:
        fig.add_trace(
            go.Scatter(
                x=history["_step"],
                y=history["frac_variance_explained"],
                mode="lines",
                name=f"Layer {layer}",
                line=dict(color=color),
                legendgroup=f"Layer {layer}",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

# Update axes for log scale
fig.update_xaxes(type="log", row=1, col=1)
fig.update_xaxes(type="log", row=1, col=2)
fig.update_xaxes(type="log", row=2, col=1)
fig.update_xaxes(type="log", row=2, col=2)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(type="log", row=1, col=2)
fig.update_yaxes(type="log", row=2, col=1)
# frac_variance_explained y axis remains linear

fig.update_layout(
    height=1200,
    width=1600,
    legend=dict(title="Layer", traceorder="reversed", itemsizing="constant"),
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.write_image("figures/sae_training_metrics.png", scale=3)
