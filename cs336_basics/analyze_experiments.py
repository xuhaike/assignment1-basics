"""
Tools for analyzing and visualizing experiment results.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_experiment_metrics(experiment_dir: str) -> List[Dict[str, Any]]:
    """
    Load metrics from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of metric dictionaries
    """
    metrics_path = Path(experiment_dir) / "metrics.json"
    if not metrics_path.exists():
        # Try CSV if JSON doesn't exist
        csv_path = Path(experiment_dir) / "metrics.csv"
        if csv_path.exists():
            metrics = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    converted_row = {}
                    for k, v in row.items():
                        try:
                            converted_row[k] = float(v)
                        except:
                            converted_row[k] = v
                    metrics.append(converted_row)
            return metrics
        else:
            raise FileNotFoundError(f"No metrics found in {experiment_dir}")

    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(
    experiment_dirs: List[str],
    experiment_names: Optional[List[str]] = None,
    output_path: str = "loss_curves.png",
    x_axis: str = "step",  # "step" or "wallclock_time"
    metrics_to_plot: List[str] = None,
):
    """
    Plot loss curves for multiple experiments.

    Args:
        experiment_dirs: List of experiment directories
        experiment_names: Names for each experiment (defaults to directory names)
        output_path: Path to save the plot
        x_axis: X-axis type ("step" or "wallclock_time")
        metrics_to_plot: List of metric names to plot (defaults to all loss metrics)
    """
    if experiment_names is None:
        experiment_names = [Path(d).name for d in experiment_dirs]

    # Load data for all experiments
    all_data = []
    for exp_dir in experiment_dirs:
        try:
            metrics = load_experiment_metrics(exp_dir)
            all_data.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
            all_data.append([])

    # Determine which metrics to plot
    if metrics_to_plot is None:
        # Find all loss-related metrics
        all_metric_names = set()
        for data in all_data:
            if data:
                all_metric_names.update(data[0].keys())

        metrics_to_plot = [m for m in all_metric_names if 'loss' in m.lower()]
        if not metrics_to_plot:
            # Fallback to all numeric metrics except step and time
            metrics_to_plot = [m for m in all_metric_names
                             if m not in ['step', 'wallclock_time', 'iteration']]

    # Create subplots
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]

    # Plot each metric
    for ax, metric_name in zip(axes, metrics_to_plot):
        for exp_name, data in zip(experiment_names, all_data):
            if not data:
                continue

            # Extract x and y values, filtering out rows with missing data
            x_values = []
            y_values = []
            for d in data:
                if metric_name in d and x_axis in d:
                    x_val = d[x_axis]
                    y_val = d[metric_name]
                    # Only include if both values are present and not None
                    if x_val is not None and y_val is not None:
                        x_values.append(x_val)
                        y_values.append(y_val)

            if x_values and y_values:
                ax.plot(x_values, y_values, label=exp_name, alpha=0.8)

        ax.set_xlabel("Steps" if x_axis == "step" else "Wallclock Time (seconds)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} over {x_axis}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def compare_experiments(
    experiment_dirs: List[str],
    experiment_names: Optional[List[str]] = None,
    output_path: str = "experiment_comparison.png",
):
    """
    Create a comprehensive comparison of multiple experiments.

    Args:
        experiment_dirs: List of experiment directories
        experiment_names: Names for each experiment
        output_path: Path to save the plot
    """
    if experiment_names is None:
        experiment_names = [Path(d).name for d in experiment_dirs]

    # Load data
    all_data = []
    all_configs = []
    for exp_dir in experiment_dirs:
        try:
            metrics = load_experiment_metrics(exp_dir)
            all_data.append(metrics)

            # Load config
            config_path = Path(exp_dir) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    all_configs.append(json.load(f))
            else:
                all_configs.append({})
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
            all_data.append([])
            all_configs.append({})

    # Create comparison plots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Train loss vs steps
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, data in zip(experiment_names, all_data):
        if data and any('train_loss' in d or 'train/loss' in d or 'loss' in d for d in data):
            # Try different possible loss key names
            loss_key = None
            for key in ['train_loss', 'train/loss', 'loss']:
                if any(key in d for d in data):
                    loss_key = key
                    break
            if loss_key:
                steps = []
                losses = []
                for d in data:
                    if loss_key in d and 'step' in d:
                        step_val = d['step']
                        loss_val = d[loss_key]
                        # Only include if both values are present and not None
                        if step_val is not None and loss_val is not None:
                            steps.append(step_val)
                            losses.append(loss_val)
                if steps and losses:
                    ax1.plot(steps, losses, label=exp_name, alpha=0.8)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Train loss vs wallclock time
    ax2 = fig.add_subplot(gs[0, 1])
    for exp_name, data in zip(experiment_names, all_data):
        if data:
            loss_key = None
            for key in ['train_loss', 'train/loss', 'loss']:
                if any(key in d for d in data):
                    loss_key = key
                    break
            if loss_key:
                times = []
                losses = []
                for d in data:
                    if loss_key in d and 'wallclock_time' in d:
                        time_val = d['wallclock_time']
                        loss_val = d[loss_key]
                        # Only include if both values are present and not None
                        if time_val is not None and loss_val is not None:
                            times.append(time_val / 3600)
                            losses.append(loss_val)
                if times and losses:
                    ax2.plot(times, losses, label=exp_name, alpha=0.8)
    ax2.set_xlabel("Wallclock Time (hours)")
    ax2.set_ylabel("Train Loss")
    ax2.set_title("Training Loss (Time)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation loss vs steps
    ax3 = fig.add_subplot(gs[1, 0])
    for exp_name, data in zip(experiment_names, all_data):
        if data:
            loss_key = None
            for key in ['val_loss', 'val/loss']:
                if any(key in d for d in data):
                    loss_key = key
                    break
            if loss_key:
                steps = []
                losses = []
                for d in data:
                    if loss_key in d and 'step' in d:
                        step_val = d['step']
                        loss_val = d[loss_key]
                        # Only include if both values are present and not None
                        if step_val is not None and loss_val is not None:
                            steps.append(step_val)
                            losses.append(loss_val)
                if steps and losses:
                    ax3.plot(steps, losses, label=exp_name, marker='o', alpha=0.8)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Validation Loss")
    ax3.set_title("Validation Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate schedule
    ax4 = fig.add_subplot(gs[1, 1])
    for exp_name, data in zip(experiment_names, all_data):
        if data and any('learning_rate' in d or 'train/learning_rate' in d or 'lr' in d for d in data):
            lr_key = None
            for key in ['learning_rate', 'train/learning_rate', 'lr']:
                if any(key in d for d in data):
                    lr_key = key
                    break
            if lr_key:
                steps = []
                lrs = []
                for d in data:
                    if lr_key in d and 'step' in d:
                        step_val = d['step']
                        lr_val = d[lr_key]
                        # Only include if both values are present and not None
                        if step_val is not None and lr_val is not None:
                            steps.append(step_val)
                            lrs.append(lr_val)
                if steps and lrs:
                    ax4.plot(steps, lrs, label=exp_name, alpha=0.8)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Throughput
    ax5 = fig.add_subplot(gs[2, 0])
    for exp_name, data in zip(experiment_names, all_data):
        if data and any('tokens_per_sec' in d or 'train/tokens_per_sec' in d for d in data):
            tok_key = None
            for key in ['tokens_per_sec', 'train/tokens_per_sec']:
                if any(key in d for d in data):
                    tok_key = key
                    break
            if tok_key:
                steps = []
                throughput = []
                for d in data:
                    if tok_key in d and 'step' in d:
                        step_val = d['step']
                        tok_val = d[tok_key]
                        # Only include if both values are present and not None
                        if step_val is not None and tok_val is not None:
                            steps.append(step_val)
                            throughput.append(tok_val)
                if steps and throughput:
                    ax5.plot(steps, throughput, label=exp_name, alpha=0.8)
    ax5.set_xlabel("Steps")
    ax5.set_ylabel("Tokens/sec")
    ax5.set_title("Training Throughput")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Final metrics comparison
    ax6 = fig.add_subplot(gs[2, 1])
    final_train_losses = []
    final_val_losses = []
    for data in all_data:
        if data:
            # Get final train loss
            loss_key = None
            for key in ['train_loss', 'train/loss', 'loss']:
                if key in data[-1]:
                    loss_key = key
                    break
            if loss_key:
                final_train_losses.append(data[-1][loss_key])
            else:
                final_train_losses.append(None)

            # Get final val loss
            val_data = [d for d in data if 'val_loss' in d or 'val/loss' in d]
            if val_data:
                val_key = 'val_loss' if 'val_loss' in val_data[-1] else 'val/loss'
                final_val_losses.append(val_data[-1][val_key])
            else:
                final_val_losses.append(None)
        else:
            final_train_losses.append(None)
            final_val_losses.append(None)

    x = range(len(experiment_names))
    width = 0.35
    if any(l is not None for l in final_train_losses):
        train_vals = [l if l is not None else 0 for l in final_train_losses]
        ax6.bar([i - width/2 for i in x], train_vals, width, label='Train', alpha=0.8)
    if any(l is not None for l in final_val_losses):
        val_vals = [l if l is not None else 0 for l in final_val_losses]
        ax6.bar([i + width/2 for i in x], val_vals, width, label='Validation', alpha=0.8)

    ax6.set_ylabel("Loss")
    ax6.set_title("Final Loss Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def print_experiment_summary(experiment_dir: str):
    """
    Print a summary of an experiment.

    Args:
        experiment_dir: Path to experiment directory
    """
    exp_path = Path(experiment_dir)

    print("=" * 80)
    print(f"Experiment: {exp_path.name}")
    print("=" * 80)

    # Load and print config
    config_path = exp_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Load metrics
    try:
        metrics = load_experiment_metrics(experiment_dir)

        print(f"\nMetrics logged: {len(metrics)} steps")

        if metrics:
            print(f"First step: {metrics[0].get('step', 'N/A')}")
            print(f"Last step: {metrics[-1].get('step', 'N/A')}")

            # Print final metrics
            print("\nFinal metrics:")
            for key, value in metrics[-1].items():
                if key not in ['step', 'wallclock_time']:
                    print(f"  {key}: {value:.4f}")

            # Print best validation loss if available
            val_metrics = [m for m in metrics if 'val_loss' in m or 'val/loss' in m]
            if val_metrics:
                val_key = 'val_loss' if 'val_loss' in val_metrics[0] else 'val/loss'
                best_val = min(val_metrics, key=lambda m: m[val_key])
                print(f"\nBest validation loss: {best_val[val_key]:.4f} at step {best_val['step']}")

    except Exception as e:
        print(f"\nError loading metrics: {e}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")

    parser.add_argument("--experiment_dirs", type=str, nargs='+', required=True,
                        help="Directories containing experiment logs")
    parser.add_argument("--names", type=str, nargs='*', default=None,
                        help="Names for experiments (optional)")
    parser.add_argument("--output", type=str, default="experiment_analysis.png",
                        help="Output path for plots")
    parser.add_argument("--plot_type", type=str, default="comparison",
                        choices=["comparison", "loss_curves"],
                        help="Type of plot to generate")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary instead of plotting")

    args = parser.parse_args()

    if args.summary:
        for exp_dir in args.experiment_dirs:
            print_experiment_summary(exp_dir)
    elif args.plot_type == "comparison":
        compare_experiments(args.experiment_dirs, args.names, args.output)
    elif args.plot_type == "loss_curves":
        plot_loss_curves(args.experiment_dirs, args.names, args.output)


if __name__ == "__main__":
    main()
