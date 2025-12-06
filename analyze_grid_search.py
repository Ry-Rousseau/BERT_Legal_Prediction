"""
Analysis and Visualization Script for Vanilla KD Grid Search Results

This script:
1. Loads evaluation results from CSV
2. Identifies best models and optimal hyperparameters
3. Extracts training loss curves from trainer_state.json files
4. Generates visualizations (heatmaps, learning curves, convergence analysis)
5. Creates summary report

User-Approved: Includes training loss analysis and saves to
results/training_runs/vanilla_kd_grid_search/training_loss_analysis.csv
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyzes evaluation results and generates reports"""

    def __init__(self, results_csv="results/eval_results/vanilla_kd_grid_search_eval.csv"):
        self.results_csv = Path(results_csv)
        self.df = None
        self.output_dir = Path("results/eval_results")
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self):
        """Load evaluation results from CSV"""
        print(f"Loading results from {self.results_csv}...")
        self.df = pd.read_csv(self.results_csv)
        print(f"Loaded {len(self.df)} evaluation results")
        print(f"Unique runs: {self.df['run_name'].nunique()}")
        print(f"Checkpoints per run: {self.df.groupby('run_name').size().describe()}")
        return self.df

    def find_best_models(self):
        """
        Identify best models across different criteria

        Returns:
            Dict with keys: best_overall, best_per_combo, best_per_epoch
        """
        if self.df is None:
            self.load_results()

        # Best overall model
        best_overall_idx = self.df['dev_accuracy'].idxmax()
        best_overall = self.df.loc[best_overall_idx].to_dict()

        # Best model per hyperparameter combination (across all epochs)
        best_per_combo = self.df.loc[self.df.groupby('run_name')['dev_accuracy'].idxmax()]

        # Best model per epoch (across all combinations)
        best_per_epoch = self.df.loc[self.df.groupby('epoch')['dev_accuracy'].idxmax()]

        results = {
            'best_overall': best_overall,
            'best_per_combo': best_per_combo,
            'best_per_epoch': best_per_epoch
        }

        return results

    def generate_summary_report(self, best_models):
        """
        Generate text summary report

        Args:
            best_models: Dict from find_best_models()
        """
        output_path = self.output_dir / "best_models_summary.txt"

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VANILLA KNOWLEDGE DISTILLATION GRID SEARCH - RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")

            # Best Overall Model
            best = best_models['best_overall']
            f.write("BEST OVERALL MODEL\n")
            f.write("-"*70 + "\n")
            f.write(f"Run Name:        {best['run_name']}\n")
            f.write(f"Alpha:           {best['alpha']}\n")
            f.write(f"Temperature:     {best['temperature']}\n")
            f.write(f"Epoch:           {best['epoch']}\n")
            f.write(f"Step:            {best['checkpoint_step']}\n")
            f.write(f"Dev Accuracy:    {best['dev_accuracy']:.4f}\n")
            f.write(f"Dev Loss:        {best['dev_loss']:.4f}\n")
            f.write(f"Checkpoint Path: {best['run_name']}/checkpoint-{best['checkpoint_step']}\n")
            f.write("\n")

            # Best per hyperparameter combination
            f.write("BEST MODEL PER HYPERPARAMETER COMBINATION\n")
            f.write("-"*70 + "\n")
            best_per_combo = best_models['best_per_combo'].sort_values('dev_accuracy', ascending=False)
            for idx, row in best_per_combo.iterrows():
                f.write(f"{row['run_name']:30s} | "
                        f"Epoch {row['epoch']} | "
                        f"Acc: {row['dev_accuracy']:.4f} | "
                        f"Loss: {row['dev_loss']:.4f}\n")
            f.write("\n")

            # Best per epoch
            f.write("BEST MODEL PER EPOCH (Across All Combinations)\n")
            f.write("-"*70 + "\n")
            best_per_epoch = best_models['best_per_epoch'].sort_values('epoch')
            for idx, row in best_per_epoch.iterrows():
                f.write(f"Epoch {row['epoch']} | "
                        f"{row['run_name']:30s} | "
                        f"Acc: {row['dev_accuracy']:.4f} | "
                        f"Alpha: {row['alpha']:.1f}, Temp: {row['temperature']:.0f}\n")
            f.write("\n")

            # Hyperparameter Analysis
            f.write("HYPERPARAMETER ANALYSIS\n")
            f.write("-"*70 + "\n")

            # Average accuracy by alpha (using final epoch only)
            final_epoch_df = self.df[self.df['epoch'] == self.df['epoch'].max()]
            alpha_avg = final_epoch_df.groupby('alpha')['dev_accuracy'].mean().sort_values(ascending=False)
            f.write("\nAverage Dev Accuracy by Alpha (Final Epoch):\n")
            for alpha, acc in alpha_avg.items():
                f.write(f"  Alpha {alpha:.1f}: {acc:.4f}\n")

            # Average accuracy by temperature
            temp_avg = final_epoch_df.groupby('temperature')['dev_accuracy'].mean().sort_values(ascending=False)
            f.write("\nAverage Dev Accuracy by Temperature (Final Epoch):\n")
            for temp, acc in temp_avg.items():
                f.write(f"  Temp {temp:.0f}: {acc:.4f}\n")

            f.write("\n")

            # Convergence Analysis
            f.write("CONVERGENCE ANALYSIS\n")
            f.write("-"*70 + "\n")

            # For runs with multiple epochs, check if accuracy is still improving
            multi_epoch_runs = self.df.groupby('run_name').filter(lambda x: len(x) > 1)
            if not multi_epoch_runs.empty:
                for run_name in multi_epoch_runs['run_name'].unique():
                    run_df = multi_epoch_runs[multi_epoch_runs['run_name'] == run_name].sort_values('epoch')
                    if len(run_df) >= 2:
                        last_epoch_acc = run_df.iloc[-1]['dev_accuracy']
                        second_last_acc = run_df.iloc[-2]['dev_accuracy']
                        improvement = last_epoch_acc - second_last_acc

                        status = "Still improving" if improvement > 0.001 else \
                                 "Converged" if abs(improvement) < 0.001 else \
                                 "Overfitting"

                        f.write(f"{run_name}: {status} "
                                f"(Epoch {run_df.iloc[-2]['epoch']}: {second_last_acc:.4f} -> "
                                f"Epoch {run_df.iloc[-1]['epoch']}: {last_epoch_acc:.4f}, "
                                f"Delta={improvement:+.4f})\n")

            f.write("\n")
            f.write("="*70 + "\n")
            f.write(f"Report generated: {pd.Timestamp.now()}\n")
            f.write("="*70 + "\n")

        print(f"Summary report saved to: {output_path}")

    def plot_alpha_temp_heatmap(self):
        """
        Generate heatmap of dev accuracy vs alpha and temperature (best epoch per combo)
        """
        if self.df is None:
            self.load_results()

        # Get best epoch for each combination
        best_per_combo = self.df.loc[self.df.groupby('run_name')['dev_accuracy'].idxmax()]

        # Create pivot table
        pivot = best_per_combo.pivot_table(
            values='dev_accuracy',
            index='temperature',
            columns='alpha',
            aggfunc='mean'
        )

        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                    cbar_kws={'label': 'Dev Accuracy'})
        plt.title('Dev Accuracy Heatmap: Alpha vs Temperature\n(Best Epoch per Combination)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Alpha (Distillation Weight)', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.tight_layout()

        output_path = self.plots_dir / 'alpha_temp_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Heatmap saved to: {output_path}")

    def plot_learning_curves(self):
        """
        Plot dev accuracy vs epoch for each hyperparameter combination
        """
        if self.df is None:
            self.load_results()

        # Filter to runs with multiple epochs
        multi_epoch_runs = self.df.groupby('run_name').filter(lambda x: len(x) > 1)

        if multi_epoch_runs.empty:
            print("No multi-epoch runs found. Skipping learning curves plot.")
            return

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot each run
        for run_name in multi_epoch_runs['run_name'].unique():
            run_df = multi_epoch_runs[multi_epoch_runs['run_name'] == run_name].sort_values('epoch')
            alpha = run_df.iloc[0]['alpha']
            temp = run_df.iloc[0]['temperature']

            label = f"α={alpha:.1f}, T={temp:.0f}"
            plt.plot(run_df['epoch'], run_df['dev_accuracy'],
                     marker='o', label=label, linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Dev Accuracy', fontsize=12)
        plt.title('Learning Curves: Dev Accuracy vs Epoch\n(By Hyperparameter Combination)',
                  fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.plots_dir / 'learning_curves_by_combo.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Learning curves saved to: {output_path}")


class TrainingLossAnalyzer:
    """Extracts and analyzes training loss curves from trainer_state.json files"""

    def __init__(self, grid_search_dir="results/training_runs/vanilla_kd_grid_search"):
        self.grid_search_dir = Path(grid_search_dir)

    def extract_training_curves(self):
        """
        Parse all trainer_state.json files and extract training loss curves

        Returns:
            pd.DataFrame with columns: run_name, alpha, temperature, step, epoch, loss, learning_rate, grad_norm
        """
        all_curves = []

        # Iterate through run directories
        for run_dir in sorted(self.grid_search_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name

            # Parse hyperparameters from run name
            try:
                parts = run_name.split('_')
                alpha = float(parts[2][1:].replace('p', '.'))
                temperature = float(parts[3][1:])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse {run_name}")
                continue

            # Find the final checkpoint's trainer_state.json (most complete)
            trainer_state_files = list(run_dir.glob('checkpoint-*/trainer_state.json'))
            if not trainer_state_files:
                print(f"Warning: No trainer_state.json found in {run_name}")
                continue

            # Use the last checkpoint (highest step number)
            trainer_state_path = sorted(trainer_state_files, key=lambda x: int(x.parent.name.split('-')[1]))[-1]

            # Load trainer_state.json
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)

            # Extract log history
            log_history = trainer_state.get('log_history', [])

            for entry in log_history:
                # Only include entries with loss (training steps)
                if 'loss' in entry:
                    all_curves.append({
                        'run_name': run_name,
                        'alpha': alpha,
                        'temperature': temperature,
                        'step': entry.get('step', 0),
                        'epoch': entry.get('epoch', 0),
                        'loss': entry.get('loss', 0),
                        'learning_rate': entry.get('learning_rate', 0),
                        'grad_norm': entry.get('grad_norm', 0)
                    })

        df = pd.DataFrame(all_curves)
        return df

    def save_training_loss_analysis(self):
        """
        Extract training curves and save to CSV alongside training runs
        """
        print("Extracting training loss curves from trainer_state.json files...")

        df = self.extract_training_curves()

        if df.empty:
            print("Warning: No training curves extracted")
            return

        output_path = self.grid_search_dir / 'training_loss_analysis.csv'
        df.to_csv(output_path, index=False)

        print(f"Training loss analysis saved to: {output_path}")
        print(f"Extracted {len(df)} training steps across {df['run_name'].nunique()} runs")

        return df

    def plot_training_curves(self, df):
        """
        Plot training loss curves for all runs

        Args:
            df: DataFrame from extract_training_curves()
        """
        if df is None or df.empty:
            print("No training data to plot")
            return

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot each run
        for run_name in df['run_name'].unique():
            run_df = df[df['run_name'] == run_name].sort_values('step')
            alpha = run_df.iloc[0]['alpha']
            temp = run_df.iloc[0]['temperature']

            label = f"α={alpha:.1f}, T={temp:.0f}"
            plt.plot(run_df['epoch'], run_df['loss'],
                     label=label, linewidth=2, alpha=0.7)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Curves\n(By Hyperparameter Combination)',
                  fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path("results/plots") / 'training_loss_curves.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training loss curves saved to: {output_path}")

    def plot_training_vs_dev(self, training_df, eval_df):
        """
        Compare training loss with dev accuracy progression

        Args:
            training_df: DataFrame from extract_training_curves()
            eval_df: DataFrame from ResultsAnalyzer.load_results()
        """
        if training_df is None or eval_df is None:
            print("Missing data for training vs dev comparison")
            return

        # Get unique runs that appear in both datasets
        common_runs = set(training_df['run_name'].unique()) & set(eval_df['run_name'].unique())

        if not common_runs:
            print("No common runs found for training vs dev comparison")
            return

        # Create subplot for first few runs (to keep plot readable)
        n_plots = min(4, len(common_runs))
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))

        if n_plots == 1:
            axes = [axes]

        for i, run_name in enumerate(sorted(common_runs)[:n_plots]):
            ax = axes[i]

            # Get training data
            train_run = training_df[training_df['run_name'] == run_name].sort_values('step')

            # Get eval data
            eval_run = eval_df[eval_df['run_name'] == run_name].sort_values('epoch')

            # Plot training loss on left axis
            ax1 = ax
            color = 'tab:blue'
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Training Loss', color=color, fontsize=10)
            ax1.plot(train_run['epoch'], train_run['loss'], color=color, linewidth=2, label='Train Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)

            # Plot dev accuracy on right axis
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Dev Accuracy', color=color, fontsize=10)
            ax2.plot(eval_run['epoch'], eval_run['dev_accuracy'],
                     color=color, marker='o', linewidth=2, label='Dev Acc')
            ax2.tick_params(axis='y', labelcolor=color)

            # Title
            alpha = eval_run.iloc[0]['alpha']
            temp = eval_run.iloc[0]['temperature']
            ax1.set_title(f"{run_name} (α={alpha:.1f}, T={temp:.0f})", fontsize=11, fontweight='bold')

        plt.tight_layout()

        output_path = Path("results/plots") / 'training_vs_dev_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training vs dev comparison saved to: {output_path}")


def main():
    """Main execution function"""

    print("="*70)
    print("VANILLA KD GRID SEARCH - ANALYSIS AND VISUALIZATION")
    print("="*70)

    # 1. Analyze evaluation results
    print("\n1. ANALYZING EVALUATION RESULTS")
    print("-"*70)

    analyzer = ResultsAnalyzer()
    df = analyzer.load_results()

    best_models = analyzer.find_best_models()

    # Print quick summary to console
    print(f"\nBest Overall Model:")
    best = best_models['best_overall']
    print(f"  {best['run_name']} (Epoch {best['epoch']})")
    print(f"  Alpha: {best['alpha']}, Temp: {best['temperature']}")
    print(f"  Dev Accuracy: {best['dev_accuracy']:.4f}")

    # Generate full summary report
    analyzer.generate_summary_report(best_models)

    # 2. Generate visualizations
    print(f"\n2. GENERATING VISUALIZATIONS")
    print("-"*70)

    analyzer.plot_alpha_temp_heatmap()
    analyzer.plot_learning_curves()

    # 3. Extract and analyze training loss curves
    print(f"\n3. EXTRACTING TRAINING LOSS CURVES")
    print("-"*70)

    loss_analyzer = TrainingLossAnalyzer()
    training_df = loss_analyzer.save_training_loss_analysis()

    if training_df is not None:
        loss_analyzer.plot_training_curves(training_df)
        loss_analyzer.plot_training_vs_dev(training_df, df)

    # 4. Final summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  - results/eval_results/best_models_summary.txt")
    print(f"  - results/training_runs/vanilla_kd_grid_search/training_loss_analysis.csv")
    print(f"  - results/plots/alpha_temp_heatmap.png")
    print(f"  - results/plots/learning_curves_by_combo.png")
    print(f"  - results/plots/training_loss_curves.png")
    print(f"  - results/plots/training_vs_dev_comparison.png")
    print(f"\nNext steps:")
    print(f"  1. Review summary report and visualizations")
    print(f"  2. Identify best model for your use case")
    print(f"  3. If performance is satisfactory, proceed with deployment or further experiments")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
