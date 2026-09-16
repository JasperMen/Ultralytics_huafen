"""Compare mAP before and after adding DySampleFusion and grouped-normalized MLLA."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = ROOT / "runs/detect/compare"
ENHANCED_ROOT = ROOT / "runs/model2025_compare"
OUTPUT_STEM = ENHANCED_ROOT / "map_before_after_comparison"
FOCUS_OUTPUT_STEM = ENHANCED_ROOT / "yolo26_map_before_after_comparison"
COMBINED_OUTPUT_STEM = ENHANCED_ROOT / "map_before_after_combined"
DASHBOARD_OUTPUT_STEM = ENHANCED_ROOT / "map_before_after_dashboard"
BROKEN_AXIS_OUTPUT_STEM = ENHANCED_ROOT / "map_before_after_reference_style"
MULTI_METRIC_OUTPUT_STEM = ENHANCED_ROOT / "multi_metric_before_after_lines"
COMBINED_MULTI_METRIC_OUTPUT_STEM = ENHANCED_ROOT / "multi_metric_all_models_combined"
MULTI_METRIC_DATA_PATH = ENHANCED_ROOT / "multi_metric_before_after_data.csv"

# Each line goes from the original model (before) to Model2025 (after).
EXPERIMENTS = {
    "YOLO11n": (
        BASELINE_ROOT / "yolo11/results.csv",
        ENHANCED_ROOT / "yolo11n-DySampleFusion-MLLAGNorm/results.csv",
    ),
    "YOLO12n": (
        BASELINE_ROOT / "yolo12/results.csv",
        ENHANCED_ROOT / "yolo12n-DySampleFusion-MLLAGNorm/results.csv",
    ),
    "YOLOv8n": (
        BASELINE_ROOT / "yolov8/results.csv",
        ENHANCED_ROOT / "yolov8n-DySampleFusion-MLLAGNorm/results.csv",
    ),
    "YOLOv10n": (
        BASELINE_ROOT / "yolov10n/results.csv",
        ENHANCED_ROOT / "yolov10n-DySampleFusion-MLLAGNorm/results.csv",
    ),
    "YOLO26n": (
        BASELINE_ROOT / "yolo26/results.csv",
        ENHANCED_ROOT / "yolo26n-DySampleFusion-MLLAGNorm/results.csv",
    ),
}

COLORS = {
    "YOLO11n": "#F2A766",
    "YOLO12n": "#C79ACF",
    "YOLOv8n": "#76BF87",
    "YOLOv10n": "#6DAED3",
    "YOLO26n": "#1233A4",
}

METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50–95": "metrics/mAP50-95(B)",
}

PROFILE_METRICS = (
    ("Precision", "metrics/precision(B)"),
    ("Recall", "metrics/recall(B)"),
    ("F1", None),
    ("mAP50", "metrics/mAP50(B)"),
    ("mAP50–95", "metrics/mAP50-95(B)"),
)


def read_best_row(csv_path: Path) -> dict[str, float]:
    """Read the row selected by the standard Ultralytics detection fitness."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        for raw_row in csv.DictReader(file):
            row = {str(key).strip(): float(value) for key, value in raw_row.items() if key is not None and value}
            row["fitness"] = 0.1 * row["metrics/mAP50(B)"] + 0.9 * row["metrics/mAP50-95(B)"]
            rows.append(row)
    if not rows:
        raise ValueError(f"No metric rows found in: {csv_path}")
    return max(rows, key=lambda row: row["fitness"])


def load_comparison() -> dict[str, dict[str, tuple[float, float]]]:
    """Load before/after percentages for both mAP metrics."""
    comparison = {}
    for model, (baseline_csv, enhanced_csv) in EXPERIMENTS.items():
        before = read_best_row(baseline_csv)
        after = read_best_row(enhanced_csv)
        comparison[model] = {
            metric_name: (before[column] * 100, after[column] * 100) for metric_name, column in METRICS.items()
        }
    return comparison


def metric_profile(row: dict[str, float]) -> list[float]:
    """Return the five comparable percentage metrics used by the line chart."""
    precision = row["metrics/precision(B)"]
    recall = row["metrics/recall(B)"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    values = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": row["metrics/mAP50(B)"],
        "mAP50–95": row["metrics/mAP50-95(B)"],
    }
    return [values[name] * 100 for name, _ in PROFILE_METRICS]


def load_metric_profiles() -> dict[str, dict[str, list[float]]]:
    """Load before/after metric profiles from each experiment's best fitness epoch."""
    profiles = {}
    for model, (baseline_csv, enhanced_csv) in EXPERIMENTS.items():
        profiles[model] = {
            "Before": metric_profile(read_best_row(baseline_csv)),
            "After": metric_profile(read_best_row(enhanced_csv)),
        }
    return profiles


def write_metric_profile_data(profiles: dict[str, dict[str, list[float]]]) -> None:
    """Write the exact plotted values in a reusable long-form CSV."""
    with MULTI_METRIC_DATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "metric", "before_pct", "after_pct", "change_pp"])
        writer.writeheader()
        metric_names = [name for name, _ in PROFILE_METRICS]
        for model, values in profiles.items():
            for metric_name, before, after in zip(metric_names, values["Before"], values["After"]):
                writer.writerow(
                    {
                        "model": model,
                        "metric": metric_name,
                        "before_pct": f"{before:.4f}",
                        "after_pct": f"{after:.4f}",
                        "change_pp": f"{after - before:+.4f}",
                    }
                )


def create_plot() -> dict[str, dict[str, tuple[float, float]]]:
    """Create a two-panel before/after slope chart."""
    comparison = load_comparison()
    x = [0, 1]
    x_labels = ["Before modules\n(Baseline)", "After modules\n(DySampleFusion + MLLA-GNorm)"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.15,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")
    legend_handles = {}

    for ax, metric_name in zip(axes, METRICS):
        ax.set_facecolor("#FCFCFD")
        before26, after26 = comparison["YOLO26n"][metric_name]
        ax.scatter(x, [before26, after26], s=500, color="#FFD166", alpha=0.30, edgecolors="none", zorder=7)

        for model, metric_values in comparison.items():
            before, after = metric_values[metric_name]
            is_yolo26 = model == "YOLO26n"
            (line,) = ax.plot(
                x,
                [before, after],
                color=COLORS[model],
                linewidth=4.2 if is_yolo26 else 2.0,
                marker="o",
                markersize=10 if is_yolo26 else 7,
                markeredgecolor="white" if is_yolo26 else COLORS[model],
                markeredgewidth=1.5 if is_yolo26 else 0.8,
                alpha=1.0 if is_yolo26 else 0.58,
                zorder=9 if is_yolo26 else 4,
                label=model,
            )
            if is_yolo26:
                line.set_path_effects([path_effects.Stroke(linewidth=7, foreground="white"), path_effects.Normal()])
            legend_handles[model] = line

        # Exact values and gain are shown only for YOLO26 to keep the figure uncluttered.
        gain = after26 - before26
        for point_x, value in zip(x, [before26, after26]):
            ax.annotate(
                f"{value:.2f}",
                (point_x, value),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=COLORS["YOLO26n"],
                fontsize=13,
                fontweight="bold",
                zorder=10,
            )
        ax.annotate(
            f"YOLO26n  {gain:+.2f} pp",
            xy=(0.5, (before26 + after26) / 2),
            xytext=(0, 24),
            textcoords="offset points",
            ha="center",
            color=COLORS["YOLO26n"],
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.34", "facecolor": "#EEF2FF", "edgecolor": "#C9D2FF"},
            zorder=11,
        )

        ax.set_title(metric_name, fontsize=18, pad=14)
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(50, 100)
        ax.set_xticks(x, x_labels)
        ax.set_yticks(range(50, 101, 5))
        ax.set_xlabel("Module configuration", fontsize=13)
        ax.grid(axis="y", color="#B8BEC8", linewidth=0.9, alpha=0.34)
        ax.grid(axis="x", color="#B8BEC8", linewidth=0.8, alpha=0.20)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Best validation performance (%)", fontsize=14)
    fig.suptitle("mAP Before vs. After Adding Model2025 Modules", fontsize=21, fontweight="bold", y=0.985)

    legend_order = ["YOLO26n", "YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n"]
    legend = fig.legend(
        [legend_handles[name] for name in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        framealpha=0.97,
        fontsize=11,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[0].set_color(COLORS["YOLO26n"])

    fig.text(
        0.5,
        0.018,
        "Note: y-axis starts at 50%; best epoch is selected using Ultralytics fitness. "
        "Baseline: original model. After: DySampleFusion + MLLA grouped normalization.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.89), w_pad=3.0)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return comparison


def create_yolo26_focus_plot(comparison: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Create a single-axis YOLO26 chart with mAP50 and mAP50-95 as the two lines."""
    x = [0, 1]
    x_labels = ["Before modules\n(Baseline)", "After modules\n(DySampleFusion + MLLA-GNorm)"]
    styles = {
        "mAP50": ("#1233A4", "o"),
        "mAP50–95": ("#E88C30", "s"),
    }

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    for metric_name, values in comparison["YOLO26n"].items():
        color, marker = styles[metric_name]
        before, after = values
        ax.scatter(x, values, s=500, color="#FFD166", alpha=0.25, edgecolors="none", zorder=7)
        (line,) = ax.plot(
            x,
            values,
            color=color,
            linewidth=4,
            marker=marker,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=metric_name,
            zorder=9,
        )
        line.set_path_effects([path_effects.Stroke(linewidth=7, foreground="white"), path_effects.Normal()])
        for point_x, value in zip(x, values):
            ax.annotate(
                f"{value:.2f}",
                (point_x, value),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                fontsize=13,
                fontweight="bold",
                zorder=10,
            )
        ax.annotate(
            f"{after - before:+.2f} pp",
            xy=(0.5, (before + after) / 2),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
        )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(50, 100)
    ax.set_xticks(x, x_labels)
    ax.set_yticks(range(50, 101, 5))
    ax.set_xlabel("Module configuration", fontsize=14)
    ax.set_ylabel("Best validation performance (%)", fontsize=14)
    ax.set_title("YOLO26n mAP Before vs. After Adding Model2025 Modules", fontsize=18, pad=18)
    ax.grid(axis="y", color="#B8BEC8", linewidth=0.9, alpha=0.34)
    ax.grid(axis="x", color="#B8BEC8", linewidth=0.8, alpha=0.20)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C8CCD3", fontsize=12)
    ax.text(
        0.01,
        0.018,
        "Note: y-axis starts at 50%; best epoch selected by Ultralytics fitness.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.tight_layout()
    fig.savefig(FOCUS_OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(FOCUS_OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_combined_plot(comparison: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Put all five models and both mAP metrics into one before/after axes."""
    x = [0, 1]
    x_labels = ["Before modules\n(Baseline)", "After modules\n(DySampleFusion + MLLA-GNorm)"]
    metric_styles = {
        "mAP50": {"linestyle": "-", "marker": "o"},
        "mAP50–95": {"linestyle": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(12.3, 7.7), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    for model, metric_values in comparison.items():
        is_yolo26 = model == "YOLO26n"
        for metric_name, values in metric_values.items():
            style = metric_styles[metric_name]
            if is_yolo26:
                ax.scatter(x, values, s=460, color="#FFD166", alpha=0.25, edgecolors="none", zorder=7)
            (line,) = ax.plot(
                x,
                values,
                color=COLORS[model],
                linestyle=style["linestyle"],
                linewidth=4.0 if is_yolo26 else 1.9,
                marker=style["marker"],
                markersize=10 if is_yolo26 else 6.5,
                markeredgecolor="white" if is_yolo26 else COLORS[model],
                markeredgewidth=1.4 if is_yolo26 else 0.7,
                alpha=1.0 if is_yolo26 else 0.48,
                zorder=9 if is_yolo26 else 4,
            )
            if is_yolo26:
                line.set_path_effects([path_effects.Stroke(linewidth=7, foreground="white"), path_effects.Normal()])
                before, after = values
                for point_x, value in zip(x, values):
                    ax.annotate(
                        f"{value:.2f}",
                        (point_x, value),
                        xytext=(0, 13),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        color=COLORS[model],
                        fontsize=12.5,
                        fontweight="bold",
                        zorder=10,
                    )
                ax.annotate(
                    f"{metric_name}  {after - before:+.2f} pp",
                    xy=(0.5, (before + after) / 2),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS[model],
                    fontsize=11.5,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.31",
                        "facecolor": "#EEF2FF",
                        "edgecolor": "#C9D2FF",
                    },
                    zorder=11,
                )

    model_order = ["YOLO26n", "YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n"]
    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linewidth=4.0 if model == "YOLO26n" else 2.2,
            marker="o",
            markersize=8,
            label=model,
            alpha=1.0 if model == "YOLO26n" else 0.65,
        )
        for model in model_order
    ]
    metric_handles = [
        Line2D([0], [0], color="#30343B", linewidth=2.5, linestyle="-", marker="o", label="mAP50"),
        Line2D([0], [0], color="#30343B", linewidth=2.5, linestyle="--", marker="s", label="mAP50–95"),
    ]
    metric_legend = ax.legend(
        handles=metric_handles,
        loc="upper left",
        title="Metric",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        fontsize=11,
    )
    ax.add_artist(metric_legend)
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper right",
        title="Model",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        fontsize=10.5,
    )
    model_legend.get_texts()[0].set_fontweight("bold")
    model_legend.get_texts()[0].set_color(COLORS["YOLO26n"])

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(50, 100)
    ax.set_xticks(x, x_labels)
    ax.set_yticks(range(50, 101, 5))
    ax.set_xlabel("Module configuration", fontsize=14)
    ax.set_ylabel("Best validation performance (%)", fontsize=14)
    ax.set_title("mAP Before vs. After Adding Model2025 Modules", fontsize=19, pad=18)
    ax.grid(axis="y", color="#B8BEC8", linewidth=0.9, alpha=0.34)
    ax.grid(axis="x", color="#B8BEC8", linewidth=0.8, alpha=0.20)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.018,
        "Note: y-axis starts at 50%; best epoch selected by Ultralytics fitness.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.tight_layout()
    fig.savefig(COMBINED_OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(COMBINED_OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_dashboard_plot(comparison: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Combine absolute performance slopes with a magnified delta bar chart."""
    x = [0, 1]
    x_labels = ["Before modules\n(Baseline)", "After modules\n(DySampleFusion + MLLA-GNorm)"]
    metric_styles = {
        "mAP50": {"linestyle": "-", "marker": "o"},
        "mAP50–95": {"linestyle": "--", "marker": "s"},
    }
    model_order = ["YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n", "YOLO26n"]

    fig = plt.figure(figsize=(13.2, 10.0), dpi=180, facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.0], hspace=0.30)
    ax = fig.add_subplot(grid[0])
    delta_ax = fig.add_subplot(grid[1])
    ax.set_facecolor("#FCFCFD")
    delta_ax.set_facecolor("#FCFCFD")

    for model, metric_values in comparison.items():
        is_yolo26 = model == "YOLO26n"
        for metric_name, values in metric_values.items():
            style = metric_styles[metric_name]
            if is_yolo26:
                ax.scatter(x, values, s=430, color="#FFD166", alpha=0.27, edgecolors="none", zorder=7)
            (line,) = ax.plot(
                x,
                values,
                color=COLORS[model],
                linestyle=style["linestyle"],
                linewidth=4.0 if is_yolo26 else 1.9,
                marker=style["marker"],
                markersize=10 if is_yolo26 else 6.5,
                markeredgecolor="white" if is_yolo26 else COLORS[model],
                markeredgewidth=1.4 if is_yolo26 else 0.7,
                alpha=1.0 if is_yolo26 else 0.48,
                zorder=9 if is_yolo26 else 4,
            )
            if is_yolo26:
                line.set_path_effects([path_effects.Stroke(linewidth=7, foreground="white"), path_effects.Normal()])
                _before, _after = values
                for point_x, value in zip(x, values):
                    ax.annotate(
                        f"{value:.2f}",
                        (point_x, value),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        color=COLORS[model],
                        fontsize=12.5,
                        fontweight="bold",
                        zorder=10,
                    )

    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linewidth=4 if model == "YOLO26n" else 2.2,
            marker="o",
            markersize=8,
            label=model,
            alpha=1 if model == "YOLO26n" else 0.65,
        )
        for model in ["YOLO26n", "YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n"]
    ]
    metric_handles = [
        Line2D([0], [0], color="#30343B", linewidth=2.5, linestyle="-", marker="o", label="mAP50"),
        Line2D([0], [0], color="#30343B", linewidth=2.5, linestyle="--", marker="s", label="mAP50–95"),
    ]
    first_legend = ax.legend(
        handles=metric_handles,
        loc="upper left",
        title="Metric",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        fontsize=10.5,
    )
    ax.add_artist(first_legend)
    second_legend = ax.legend(
        handles=model_handles,
        loc="upper right",
        title="Model",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        fontsize=10,
    )
    second_legend.get_texts()[0].set_fontweight("bold")
    second_legend.get_texts()[0].set_color(COLORS["YOLO26n"])

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(50, 100)
    ax.set_xticks(x, x_labels)
    ax.set_yticks(range(50, 101, 5))
    ax.set_ylabel("Best validation performance (%)", fontsize=14)
    ax.set_title("Absolute Performance", fontsize=16, pad=12)
    ax.grid(axis="y", color="#B8BEC8", linewidth=0.9, alpha=0.34)
    ax.grid(axis="x", color="#B8BEC8", linewidth=0.8, alpha=0.20)
    ax.set_axisbelow(True)

    # Magnified improvement panel: this makes small, real differences visible without distorting the main y-axis.
    positions = list(range(len(model_order)))
    width = 0.34
    map50_deltas = [comparison[model]["mAP50"][1] - comparison[model]["mAP50"][0] for model in model_order]
    map95_deltas = [comparison[model]["mAP50–95"][1] - comparison[model]["mAP50–95"][0] for model in model_order]
    delta_ax.axvspan(3.55, 4.45, color="#EEF2FF", alpha=0.85, zorder=0)
    bars50 = delta_ax.bar(
        [position - width / 2 for position in positions],
        map50_deltas,
        width,
        color="#4361EE",
        alpha=0.86,
        label="Δ mAP50",
        zorder=3,
    )
    bars95 = delta_ax.bar(
        [position + width / 2 for position in positions],
        map95_deltas,
        width,
        color="#F28E2B",
        alpha=0.86,
        label="Δ mAP50–95",
        zorder=3,
    )
    for index, (bar50, bar95) in enumerate(zip(bars50, bars95)):
        if model_order[index] == "YOLO26n":
            for bar in (bar50, bar95):
                bar.set_edgecolor(COLORS["YOLO26n"])
                bar.set_linewidth(2.2)
        for bar in (bar50, bar95):
            height = bar.get_height()
            delta_ax.annotate(
                f"{height:+.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9.5,
                fontweight="bold" if model_order[index] == "YOLO26n" else "normal",
                color="#263044",
            )

    delta_ax.axhline(0, color="#444B55", linewidth=1.1)
    delta_ax.set_ylim(-0.35, 1.40)
    delta_ax.set_yticks([-0.25, 0, 0.25, 0.50, 0.75, 1.00, 1.25])
    delta_ax.set_xticks(positions, model_order)
    for label in delta_ax.get_xticklabels():
        if label.get_text() == "YOLO26n":
            label.set_fontweight("bold")
            label.set_color(COLORS["YOLO26n"])
    delta_ax.set_ylabel("Improvement (pp)", fontsize=13)
    delta_ax.set_title("Magnified Improvement After Adding Modules", fontsize=15, pad=10)
    delta_ax.grid(axis="y", color="#B8BEC8", linewidth=0.8, alpha=0.32)
    delta_ax.set_axisbelow(True)
    delta_ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#C8CCD3")

    fig.suptitle("Model2025 mAP: Before vs. After Module Integration", fontsize=21, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.016,
        "Top: absolute best-epoch performance (50–100%). Bottom: magnified change in percentage points. "
        "YOLO26n ranks first after integration on both mAP metrics.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.08, hspace=0.36)
    fig.savefig(DASHBOARD_OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(DASHBOARD_OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_reference_style_plot(comparison: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Create a reference-style, single-figure broken-axis slope chart."""
    x = [0, 1]
    x_labels = ["Before modules\n(Baseline)", "After modules\n(DySampleFusion + MLLA-GNorm)"]
    model_order = ["YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n", "YOLO26n"]

    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        figsize=(12.2, 8.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.055},
    )
    fig.patch.set_facecolor("white")
    for axis in (top_ax, bottom_ax):
        axis.set_facecolor("#FCFCFD")
        axis.grid(axis="y", color="#B8BEC8", linewidth=0.9, alpha=0.34)
        axis.grid(axis="x", color="#B8BEC8", linewidth=0.8, alpha=0.20)
        axis.set_axisbelow(True)
        axis.set_xlim(-0.12, 1.23)

    for model in model_order:
        is_yolo26 = model == "YOLO26n"
        for axis, metric_name, linestyle, marker in (
            (top_ax, "mAP50", "-", "o"),
            (bottom_ax, "mAP50–95", "--", "s"),
        ):
            values = comparison[model][metric_name]
            if is_yolo26:
                axis.scatter(x, values, s=460, color="#FFD166", alpha=0.27, edgecolors="none", zorder=7)
            (line,) = axis.plot(
                x,
                values,
                color=COLORS[model],
                linestyle=linestyle,
                linewidth=4.2 if is_yolo26 else 2.0,
                marker=marker,
                markersize=10 if is_yolo26 else 7,
                markeredgecolor="white" if is_yolo26 else COLORS[model],
                markeredgewidth=1.5 if is_yolo26 else 0.8,
                alpha=1.0 if is_yolo26 else 0.62,
                zorder=9 if is_yolo26 else 4,
            )
            if is_yolo26:
                line.set_path_effects([path_effects.Stroke(linewidth=7, foreground="white"), path_effects.Normal()])
                before, after = values
                for point_x, value in zip(x, values):
                    axis.annotate(
                        f"{value:.2f}",
                        (point_x, value),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        color=COLORS[model],
                        fontsize=12.5,
                        fontweight="bold",
                        zorder=10,
                    )
                axis.annotate(
                    f"YOLO26n  {after - before:+.2f} pp",
                    xy=(0.5, (before + after) / 2),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha="center",
                    color=COLORS[model],
                    fontsize=11.5,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.32", "facecolor": "#EEF2FF", "edgecolor": "#C9D2FF"},
                    zorder=11,
                )

            # Place the enhanced value at the right end, similar to labels in the reference chart.
            if not is_yolo26:
                label_offsets = {
                    ("YOLO11n", "mAP50"): (8, -13),
                    ("YOLOv8n", "mAP50"): (8, 10),
                    ("YOLO11n", "mAP50–95"): (8, -12),
                    ("YOLOv8n", "mAP50–95"): (8, 9),
                }
                axis.annotate(
                    f"{model}  {values[1]:.2f}",
                    (1, values[1]),
                    xytext=label_offsets.get((model, metric_name), (8, 0)),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=COLORS[model],
                    fontsize=9.5,
                    alpha=0.92,
                )

    # Tight, truthful ranges reproduce the visual logic of the supplied reference.
    top_ax.set_ylim(82.5, 89.2)
    top_ax.set_yticks([83, 84, 85, 86, 87, 88, 89])
    bottom_ax.set_ylim(52.5, 62.2)
    bottom_ax.set_yticks([53, 54, 55, 56, 57, 58, 59, 60, 61, 62])

    # Broken-axis marks show explicitly that the middle range is omitted.
    top_ax.spines["bottom"].set_visible(False)
    bottom_ax.spines["top"].set_visible(False)
    top_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    bottom_ax.tick_params(axis="x", which="both", top=False)
    diagonal = 0.008
    top_kwargs = {"transform": top_ax.transAxes, "color": "#252A32", "clip_on": False, "linewidth": 1.4}
    top_ax.plot((-diagonal, +diagonal), (-diagonal, +diagonal), **top_kwargs)
    top_ax.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), **top_kwargs)
    bottom_kwargs = {"transform": bottom_ax.transAxes, "color": "#252A32", "clip_on": False, "linewidth": 1.4}
    bottom_ax.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), **bottom_kwargs)
    bottom_ax.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), **bottom_kwargs)

    top_ax.text(0.015, 0.89, "mAP50", transform=top_ax.transAxes, fontsize=14, fontweight="bold", color="#30343B")
    bottom_ax.text(
        0.015,
        0.88,
        "mAP50–95",
        transform=bottom_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#30343B",
    )
    bottom_ax.set_xticks(x, x_labels)
    bottom_ax.set_xlabel("Module configuration", fontsize=14, labelpad=8)
    fig.text(0.025, 0.5, "Best validation performance (%)", rotation=90, va="center", fontsize=14)
    fig.suptitle("mAP Before vs. After Adding Model2025 Modules", fontsize=20, fontweight="bold", y=0.98)

    legend_order = ["YOLO26n", "YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n"]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linewidth=4 if model == "YOLO26n" else 2.2,
            marker="o",
            markersize=8,
            label=model,
            alpha=1 if model == "YOLO26n" else 0.7,
        )
        for model in legend_order
    ]
    legend = top_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.02),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CCD3",
        fontsize=10,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[0].set_color(COLORS["YOLO26n"])
    fig.text(
        0.5,
        0.018,
        "Broken y-axis zooms into the two metric ranges; all labels use the original measured percentages.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.17, hspace=0.055)
    fig.savefig(
        BROKEN_AXIS_OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(BROKEN_AXIS_OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_multi_metric_line_plot() -> dict[str, dict[str, list[float]]]:
    """Create small-multiple line charts with metrics on the x-axis."""
    profiles = load_metric_profiles()
    write_metric_profile_data(profiles)
    metric_names = [name for name, _ in PROFILE_METRICS]
    x = list(range(len(metric_names)))
    model_order = ["YOLO26n", "YOLOv8n", "YOLO11n", "YOLOv10n", "YOLO12n"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#2B313B",
            "axes.linewidth": 1.05,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()

    for panel_index, (ax, model) in enumerate(zip(axes_flat, model_order)):
        before = profiles[model]["Before"]
        after = profiles[model]["After"]
        model_color = COLORS[model]
        is_yolo26 = model == "YOLO26n"

        ax.set_facecolor("#F4F7FF" if is_yolo26 else "#FCFCFD")
        ax.fill_between(x, before, after, color=model_color, alpha=0.10 if is_yolo26 else 0.07, zorder=1)
        ax.plot(
            x,
            before,
            color="#7D8796",
            linewidth=2.2,
            linestyle=(0, (5, 3)),
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="Before modules",
            zorder=3,
        )
        (after_line,) = ax.plot(
            x,
            after,
            color=model_color,
            linewidth=4.0 if is_yolo26 else 2.8,
            marker="o",
            markersize=9 if is_yolo26 else 7.5,
            markerfacecolor=model_color,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label="After modules",
            zorder=5,
        )
        if is_yolo26:
            after_line.set_path_effects([path_effects.Stroke(linewidth=6.8, foreground="white"), path_effects.Normal()])
            ax.scatter(
                x[3:],
                after[3:],
                s=330,
                color="#FFD166",
                alpha=0.28,
                edgecolors="none",
                zorder=4,
            )

        for metric_x, before_value, after_value in zip(x, before, after):
            delta = after_value - before_value
            ax.annotate(
                f"{delta:+.2f}",
                (metric_x, after_value),
                xytext=(0, 10 if delta >= 0 else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                color="#16823B" if delta >= 0 else "#B33A3A",
                fontsize=8.7 if is_yolo26 else 8.1,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
                zorder=8,
            )

        title = f"{model}  ★" if is_yolo26 else model
        ax.set_title(title, fontsize=15 if is_yolo26 else 13.5, color=model_color, pad=12)
        ax.set_xticks(x, metric_names, rotation=10)
        ax.set_xlim(-0.25, len(metric_names) - 0.75)
        ax.set_ylim(50, 92)
        ax.set_yticks(range(50, 91, 5))
        ax.grid(axis="y", color="#AEB6C2", linewidth=0.8, alpha=0.30)
        ax.grid(axis="x", color="#AEB6C2", linewidth=0.65, alpha=0.16)
        ax.set_axisbelow(True)
        if panel_index % 3 == 0:
            ax.set_ylabel("Performance (%)", fontsize=11.5)
        if is_yolo26:
            for spine in ax.spines.values():
                spine.set_color(model_color)
                spine.set_linewidth(1.65)

    summary_ax = axes_flat[-1]
    summary_ax.set_facecolor("#F7F9FC")
    summary_ax.set_xticks([])
    summary_ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    for spine in summary_ax.spines.values():
        spine.set_visible(False)
    summary_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#7D8796",
                linewidth=2.2,
                linestyle=(0, (5, 3)),
                marker="o",
                markerfacecolor="white",
                label="Before modules (baseline)",
            ),
            Line2D([0], [0], color=COLORS["YOLO26n"], linewidth=3.2, marker="o", label="After modules (model color)"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=11.5,
    )
    summary_ax.text(
        0.04,
        0.68,
        "Numbers next to points are changes\n(after − before, percentage points).",
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#4B5563",
        linespacing=1.5,
    )
    yolo26_before = profiles["YOLO26n"]["Before"]
    yolo26_after = profiles["YOLO26n"]["After"]
    summary_ax.text(
        0.04,
        0.48,
        "YOLO26n after modules",
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["YOLO26n"],
    )
    summary_ax.text(
        0.04,
        0.39,
        f"mAP50       {yolo26_after[3]:.2f}%   ({yolo26_after[3] - yolo26_before[3]:+.2f} pp)\n"
        f"mAP50–95  {yolo26_after[4]:.2f}%   ({yolo26_after[4] - yolo26_before[4]:+.2f} pp)",
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#202938",
        linespacing=1.65,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#EEF2FF", "edgecolor": "#C5D0FF"},
    )
    summary_ax.text(
        0.04,
        0.12,
        "YOLO26n ranks first among the five models\non both mAP metrics after enhancement.",
        transform=summary_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        color="#4B5563",
        linespacing=1.45,
    )

    fig.suptitle(
        "Multi-Metric Performance Before vs. After Adding Model2025 Modules",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.943,
        "Precision, Recall, F1, mAP50 and mAP50–95 at the best Ultralytics fitness epoch",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#5D6470",
    )
    fig.text(
        0.5,
        0.018,
        "F1 is calculated from the measured precision and recall. All panels share the same y-axis; no values are normalized.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#5D6470",
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.975, 0.915), h_pad=3.0, w_pad=2.0)
    fig.savefig(
        MULTI_METRIC_OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(MULTI_METRIC_OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return profiles


def create_combined_multi_metric_plot() -> None:
    """Overlay all five model profiles in one axes and emphasize YOLO26n."""
    profiles = load_metric_profiles()
    metric_names = [name for name, _ in PROFILE_METRICS]
    x = list(range(len(metric_names)))
    model_order = ["YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n", "YOLO26n"]

    fig, ax = plt.subplots(figsize=(14.8, 8.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")

    # Draw the other models first so the emphasized YOLO26n profile remains on top.
    for model in model_order:
        before = profiles[model]["Before"]
        after = profiles[model]["After"]
        color = COLORS[model]
        is_yolo26 = model == "YOLO26n"

        ax.plot(
            x,
            before,
            color=color,
            linewidth=3.0 if is_yolo26 else 1.45,
            linestyle=(0, (5, 3)),
            marker="o",
            markersize=8 if is_yolo26 else 4.5,
            markerfacecolor="white",
            markeredgewidth=1.5 if is_yolo26 else 1.0,
            alpha=0.65 if is_yolo26 else 0.26,
            zorder=6 if is_yolo26 else 2,
        )
        if is_yolo26:
            ax.scatter(
                x[3:],
                after[3:],
                s=390,
                color="#FFD166",
                alpha=0.30,
                edgecolors="none",
                zorder=7,
            )
        (after_line,) = ax.plot(
            x,
            after,
            color=color,
            linewidth=4.8 if is_yolo26 else 2.5,
            marker="o",
            markersize=11 if is_yolo26 else 7.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5 if is_yolo26 else 1.0,
            alpha=1.0 if is_yolo26 else 0.76,
            zorder=10 if is_yolo26 else 4,
        )
        if is_yolo26:
            after_line.set_path_effects([path_effects.Stroke(linewidth=8.0, foreground="white"), path_effects.Normal()])

        ax.text(
            4.10,
            after[-1],
            f"{model}  {after[-1]:.2f}",
            ha="left",
            va="center",
            fontsize=11.5 if is_yolo26 else 10.2,
            fontweight="bold" if is_yolo26 else "normal",
            color=color,
            alpha=1.0 if is_yolo26 else 0.86,
            zorder=12,
        )

    yolo26_before = profiles["YOLO26n"]["Before"]
    yolo26_after = profiles["YOLO26n"]["After"]
    for metric_x, after_value in enumerate(yolo26_after):
        if metric_x >= 3:
            delta = after_value - yolo26_before[metric_x]
            label = f"{after_value:.2f}%  ({delta:+.2f} pp)"
        else:
            label = f"{after_value:.2f}%"
        ax.annotate(
            label,
            (metric_x, after_value),
            xytext=(0, 15 if metric_x != 4 else 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["YOLO26n"],
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D5DCFF", "alpha": 0.94},
            zorder=13,
        )

    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linewidth=4.5 if model == "YOLO26n" else 2.5,
            marker="o",
            markersize=8 if model == "YOLO26n" else 6,
            label=model,
            alpha=1.0 if model == "YOLO26n" else 0.82,
        )
        for model in ["YOLO26n", "YOLO11n", "YOLO12n", "YOLOv8n", "YOLOv10n"]
    ]
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#CDD2DA",
        fontsize=10.5,
    )
    model_legend.get_texts()[0].set_fontweight("bold")
    model_legend.get_texts()[0].set_color(COLORS["YOLO26n"])
    ax.add_artist(model_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#505866",
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                marker="o",
                markerfacecolor="white",
                label="Before modules",
            ),
            Line2D([0], [0], color="#505866", linewidth=2.8, marker="o", label="After modules"),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#CDD2DA",
        fontsize=10.5,
    )

    ax.set_xlim(-0.22, 4.78)
    ax.set_ylim(50, 92)
    ax.set_xticks(x, metric_names)
    ax.set_yticks(range(50, 91, 5))
    ax.set_xlabel("Evaluation metric", fontsize=13, labelpad=10)
    ax.set_ylabel("Best validation performance (%)", fontsize=13)
    ax.grid(axis="y", color="#AEB6C2", linewidth=0.85, alpha=0.34)
    ax.grid(axis="x", color="#AEB6C2", linewidth=0.7, alpha=0.18)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Five-Model Multi-Metric Comparison Before and After Model2025 Modules",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "YOLO26n is emphasized; dashed lines are baselines and solid lines are enhanced models",
        ha="center",
        va="center",
        fontsize=11.2,
        color="#5D6470",
    )
    fig.text(
        0.5,
        0.018,
        "F1 is calculated from measured precision and recall. All values use the best Ultralytics fitness epoch and are not normalized.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#5D6470",
    )
    fig.subplots_adjust(left=0.075, right=0.84, top=0.80, bottom=0.105)
    fig.savefig(
        COMBINED_MULTI_METRIC_OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        COMBINED_MULTI_METRIC_OUTPUT_STEM.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


if __name__ == "__main__":
    used_values = create_plot()
    create_yolo26_focus_plot(used_values)
    create_combined_plot(used_values)
    create_dashboard_plot(used_values)
    create_reference_style_plot(used_values)
    create_multi_metric_line_plot()
    create_combined_multi_metric_plot()
    for model, metrics in used_values.items():
        values = ", ".join(
            f"{metric}: {before:.3f}% -> {after:.3f}% ({after - before:+.3f} pp)"
            for metric, (before, after) in metrics.items()
        )
        print(f"{model}: {values}")
    print(f"Saved: {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {FOCUS_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {FOCUS_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {COMBINED_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {COMBINED_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {DASHBOARD_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {DASHBOARD_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {BROKEN_AXIS_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {BROKEN_AXIS_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {MULTI_METRIC_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {MULTI_METRIC_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {COMBINED_MULTI_METRIC_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {COMBINED_MULTI_METRIC_OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved: {MULTI_METRIC_DATA_PATH}")
