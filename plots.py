from pathlib import Path

from plotly import graph_objects as go

def save_metric_plots(history) -> None:
    """
    Saves one Plotly HTML plot per metric with train vs val curves.
    Produces: plots/{metric}.html
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    hist = history.history
    base_metrics = [k for k in hist.keys() if not k.startswith("val_")]
    if not base_metrics:
        return

    epochs = list(range(1, len(next(iter(hist.values()))) + 1))

    for m in base_metrics:
        train_key = m
        val_key = f"val_{m}"

        # If there's no validation series (e.g., no validation_data), skip gracefully.
        if train_key not in hist or val_key not in hist:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=hist[train_key], mode="lines+markers", name=f"train_{m}"))
        fig.add_trace(go.Scatter(x=epochs, y=hist[val_key], mode="lines+markers", name=f"val_{m}"))

        fig.update_layout(
            title=f"{m}: Training vs Validation",
            xaxis_title="Epoch",
            yaxis_title=m,
            template="plotly_white",
            legend_title_text="Split",
        )

        fig.write_html(plots_dir / f"{m}.html", include_plotlyjs="cdn")