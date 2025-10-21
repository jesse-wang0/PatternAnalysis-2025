import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_training_history(training_history_file: Path, results_folder: Path):
    """Plot and save training/validation loss curves"""
    results_folder = Path(results_folder)
    
    # Load training history
    df = pd.read_csv(training_history_file)
    
    # Extract data
    train_data = df[df['train_loss'].notna()].copy()
    eval_data = df[df['eval_loss'].notna()].copy()
    
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
