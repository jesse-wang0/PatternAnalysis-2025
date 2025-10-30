"""
Utility module for visualizing training and validation loss curves.

Used to generate and save loss plots for individual or multiple BioLay-Summ model runs.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_training_history(training_history_file: Path, results_folder: Path):
    """
    Plots and saves training and validation loss curves from a single run.

    Args:
        training_history_file (Path): Path to the CSV log file containing loss values.
        results_folder (Path): Directory where the plot will be saved.
    """

    results_folder = Path(results_folder)
    
    # Load training history
    df = pd.read_csv(training_history_file)
    
    # Extract data
    train_data = df[df['train_loss'].notna() & df['eval_loss'].isna()].copy()  # Only interval logs
    eval_data = df[df['eval_loss'].notna()].copy()  # Epoch-end summaries
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['step'], train_data['train_loss'], 
            label='Train Loss', alpha=0.7, linewidth=1)
    plt.plot(eval_data['step'], eval_data['eval_loss'], 
            label='Eval Loss', marker='o', markersize=10, 
            linewidth=2, color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add epoch boundaries
    for idx, step in enumerate(eval_data['step'].values):
        plt.axvline(x=step, color='gray', linestyle='--', 
                alpha=0.3, linewidth=1)
        plt.text(step, plt.ylim()[1]*0.95, f'Epoch {idx+1}', 
                rotation=0, ha='center', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(results_folder / 'loss_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve saved to {results_folder / 'loss_curve.png'}")
    print(f"  Training points logged: {len(train_data)}")
    print(f"  Evaluation points: {len(eval_data)}")

def plot_multiple_training_histories(model_histories: dict, results_folder: Path):
    """
    Plots and saves comparative loss curves across multiple model runs.

    Args:
        model_histories (dict): Mapping of model names to their training history CSV files.
        results_folder (Path): Directory where the comparison plot will be saved.
    """
    
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 8))

    train_colors = {
        "Flan-T5 Small": "#ff7f0e",  # orange
        "Flan-T5 Base": "#1f77b4",  # blue
        "Flan-T5 Base + LoRA": "#2ca02c",  # green
    }

    eval_colors = {
        "Flan-T5 Small": "#d62728",  # red
        "Flan-T5 Base": "#9467bd",  # purple
        "Flan-T5 Base + LoRA": "#a0522d"  # brown
    }

    for model_name, history_file in model_histories.items():
        df = pd.read_csv(history_file)

        # Keep only interval logs for training
        train_data = df[df['train_loss'].notna() & df['eval_loss'].isna()].copy()
        eval_data = df[df['eval_loss'].notna()].copy()

        # Solid line for training loss
        plt.plot(train_data['epoch'], train_data['train_loss'],
                 label=f'{model_name} (Train)',
                 linewidth=1.5, alpha=0.7, color=train_colors.get(model_name, "#7f7f7f"),
                 linestyle='-', zorder=1)

        # Dash-dot line for evaluation loss
        plt.plot(eval_data['epoch'], eval_data['eval_loss'],
                 label=f'{model_name} (Eval)',
                 linewidth=2.2, alpha=0.9, color=eval_colors.get(model_name, "#000000"),
                 linestyle='dotted',
                 marker='D', markersize=6, markerfacecolor=eval_colors.get(model_name, "#000000"),
                 markeredgecolor='black', zorder=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss by Epoch', fontsize=14)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = results_folder / 'loss_curve_comparison_by_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve (by epoch) saved to {output_path}")
