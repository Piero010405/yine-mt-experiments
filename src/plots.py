"""
Plotting utilities for training logs. This module provides a function to save training plots for    
evaluation metrics such as loss, BLEU, ChrF, and TER. The plots are saved in the specified output 
directory.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_training_plots(log_history, out_dir: str):
    """
    Saves training plots for evaluation metrics such as loss, BLEU, ChrF, and TER. The log_history is
    expected to be a list of dictionaries containing the logged metrics for each evaluation step. The 
    plots are saved in the specified output directory.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(log_history)

    # keep eval logs
    eval_df = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()
    if eval_df.empty:
        return

    # helper
    def plot_metric(col, filename):
        """
        Plots a metric over epochs and saves the plot to a file. If the metric column is not present 
        in the evaluation DataFrame, the function returns without plotting.
        """
        if col not in eval_df.columns:
            return
        plt.figure()
        plt.plot(eval_df["epoch"], eval_df[col])
        plt.xlabel("epoch")
        plt.ylabel(col)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(str(Path(out_dir) / filename), dpi=200)
        plt.close()

    plot_metric("eval_loss", "eval_loss.png")
    plot_metric("eval_bleu", "eval_bleu.png")
    plot_metric("eval_chrf", "eval_chrf.png")
    plot_metric("eval_chrfpp", "eval_chrfpp.png")
    plot_metric("eval_ter", "eval_ter.png")
