"""
Experiment logging infrastructure for tracking training runs.

Supports:
- Logging metrics with gradient steps and wallclock time
- Saving experiment metadata and hyperparameters
- CSV logging for easy analysis
- JSON logging for complete records
- Integration with Weights & Biases
"""

import json
import csv
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    # Experiment metadata
    experiment_name: str
    description: str
    tags: list[str] = field(default_factory=list)

    # Model architecture
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0

    # Training hyperparameters
    batch_size: int = 8
    max_iters: int = 100000
    seed: int = 42

    # Optimizer
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 100000

    # Data
    train_data_path: str = ""
    val_data_path: str = ""

    # Device
    device: str = "cuda"

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentLogger:
    """
    Logger for tracking experiments with metrics, hyperparameters, and metadata.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[ExperimentConfig] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """
        Initialize experiment logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            config: Experiment configuration
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config
        self.start_time = time.time()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV logger
        self.metrics_csv_path = self.experiment_dir / "metrics.csv"
        self.csv_file = open(self.metrics_csv_path, 'w', newline='')
        self.csv_writer = None
        self.csv_fieldnames = None

        # Initialize JSON logger for all metrics
        self.metrics_json_path = self.experiment_dir / "metrics.json"
        self.metrics_data = []

        # Save config
        if config:
            config_path = self.experiment_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

        # Initialize W&B if requested
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config.to_dict() if config else {},
                )
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False

        print(f"Experiment logs will be saved to: {self.experiment_dir}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        wallclock_time: Optional[float] = None,
    ):
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step/iteration
            wallclock_time: Wallclock time since training start (computed if not provided)
        """
        if wallclock_time is None:
            wallclock_time = time.time() - self.start_time

        # Add step and time to metrics
        log_entry = {
            'step': step,
            'wallclock_time': wallclock_time,
            **metrics
        }

        # Log to CSV
        if self.csv_writer is None:
            # Initialize CSV writer with fieldnames from first log
            self.csv_fieldnames = list(log_entry.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
            self.csv_writer.writeheader()
        else:
            new_fields = [key for key in log_entry.keys() if key not in self.csv_fieldnames]
            if new_fields:
                self.csv_fieldnames.extend(new_fields)
                # Rewrite CSV with the expanded header to include new fields.
                self.csv_file.close()
                self.csv_file = open(self.metrics_csv_path, 'w', newline='')
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
                for entry in self.metrics_data:
                    self.csv_writer.writerow(entry)

        self.csv_writer.writerow(log_entry)
        self.csv_file.flush()

        # Log to JSON
        self.metrics_data.append(log_entry)

        # Log to W&B
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({**metrics, 'step': step, 'wallclock_time': wallclock_time})

    def log_text(self, text: str, filename: str = "log.txt"):
        """
        Log text to a file.

        Args:
            text: Text to log
            filename: Name of the log file
        """
        log_path = self.experiment_dir / filename
        with open(log_path, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")

    def save_checkpoint_info(self, checkpoint_path: str, step: int, metrics: Dict[str, float]):
        """
        Log information about a saved checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint
            step: Training step when checkpoint was saved
            metrics: Metrics at checkpoint time
        """
        checkpoint_info = {
            'path': checkpoint_path,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }

        checkpoints_path = self.experiment_dir / "checkpoints.json"

        # Load existing checkpoints
        if checkpoints_path.exists():
            with open(checkpoints_path, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []

        checkpoints.append(checkpoint_info)

        # Save updated list
        with open(checkpoints_path, 'w') as f:
            json.dump(checkpoints, f, indent=2)

    def finalize(self):
        """Finalize logging (save all data, close files)."""
        # Save JSON metrics
        with open(self.metrics_json_path, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)

        # Close CSV file
        self.csv_file.close()

        # Finalize W&B
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        print(f"Experiment logs saved to: {self.experiment_dir}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class ExperimentRegistry:
    """
    Registry to keep track of all experiments.
    """

    def __init__(self, registry_path: str = "experiments/registry.json"):
        """
        Initialize experiment registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []

    def add_experiment(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        log_dir: str,
        status: str = "running",
    ):
        """
        Add an experiment to the registry.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            log_dir: Directory where logs are saved
            status: Status of experiment (running, completed, failed)
        """
        experiment_entry = {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'log_dir': str(log_dir),
            'status': status,
            'config': config.to_dict(),
        }

        self.experiments.append(experiment_entry)
        self._save()

    def update_status(self, experiment_name: str, status: str):
        """
        Update experiment status.

        Args:
            experiment_name: Name of the experiment
            status: New status
        """
        for exp in self.experiments:
            if exp['name'] == experiment_name:
                exp['status'] = status
                exp['updated_at'] = datetime.now().isoformat()
        self._save()

    def list_experiments(self, status: Optional[str] = None) -> list[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.

        Args:
            status: Filter by status (None for all)

        Returns:
            List of experiment entries
        """
        if status is None:
            return self.experiments
        return [exp for exp in self.experiments if exp['status'] == status]

    def _save(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)


def format_experiment_summary(config: ExperimentConfig) -> str:
    """
    Format experiment configuration as a readable summary.

    Args:
        config: Experiment configuration

    Returns:
        Formatted string summary
    """
    summary = f"""
Experiment: {config.experiment_name}
Description: {config.description}
Tags: {', '.join(config.tags)}

Model Architecture:
  - Vocabulary Size: {config.vocab_size:,}
  - Context Length: {config.context_length}
  - Model Dimension: {config.d_model}
  - Layers: {config.num_layers}
  - Heads: {config.num_heads}
  - FFN Dimension: {config.d_ff}
  - RoPE Theta: {config.rope_theta}

Training Configuration:
  - Batch Size: {config.batch_size}
  - Max Iterations: {config.max_iters:,}
  - Seed: {config.seed}

Optimizer (AdamW):
  - Max Learning Rate: {config.max_lr}
  - Min Learning Rate: {config.min_lr}
  - Beta1: {config.beta1}, Beta2: {config.beta2}
  - Epsilon: {config.eps}
  - Weight Decay: {config.weight_decay}
  - Gradient Clipping: {config.grad_clip}

Learning Rate Schedule:
  - Warmup Iterations: {config.warmup_iters}
  - Cosine Cycle Iterations: {config.cosine_cycle_iters}

Data:
  - Train: {config.train_data_path}
  - Validation: {config.val_data_path}

Notes: {config.notes}
"""
    return summary
