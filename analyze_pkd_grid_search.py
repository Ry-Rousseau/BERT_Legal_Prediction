"""
Analysis and Visualization Script for PKD-Skip Grid Search Results

This script analyzes the evaluation results from the PKD-Skip grid search,
generates summary reports, and creates visualizations.

Analyzes:
- Best model per student size
- Beta value impact per student size
- Learning curves across epochs
- Comparison with vanilla KD baseline
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ResultsAnalyzer:
    """Analyzes PKD grid search evaluation results"""

    def __init__(self, csv_path="results/eval_results/pkd_skip_grid_search_eval.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self.load_results()

    def load_results(self):
        """Load evaluation results from CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} evaluation results from {self.csv_path}")

    def find_best_overall(self):
        """Find the best performing model overall"""
        best_row = self.df.loc[self.df['dev_accuracy'].idxmax()]
        return {
            'run_name': best_row['run_name'],
            'student_layers': best_row['student_layers'],
            'beta': best_row['beta'],
            'epoch': best_row['epoch'],
            'dev_accuracy': best_row['dev_accuracy'],
            'dev_loss': best_row['dev_loss']
        }

    def find_best_per_student_size(self):
        """Find best model for each student size"""
        results = {}
        for layers in sorted(self.df['student_layers'].unique()):
            subset = self.df[self.df['student_layers'] == layers]
            best_row = subset.loc[subset['dev_accuracy'].idxmax()]
            results[layers] = {
                'run_name': best_row['run_name'],
                'beta': best_row['beta'],
                'epoch': best_row['epoch'],
                'dev_accuracy': best_row['dev_accuracy'],
                'dev_loss': best_row['dev_loss']
            }
        return results

    def find_best_per_beta(self):
        """Find best model for each beta value"""
        results = {}
        for beta in sorted(self.df['beta'].unique()):
            subset = self.df[self.df['beta'] == beta]
            best_row = subset.loc[subset['dev_accuracy'].idxmax()]
            results[beta] = {
                'run_name': best_row['run_name'],
                'student_layers': best_row['student_layers'],
                'epoch': best_row['epoch'],
                'dev_accuracy': best_row['dev_accuracy'],
                'dev_loss': best_row['dev_loss']
            }
        return results

    def generate_summary_report(self, output_path="results/eval_results/pkd_best_models_summary.txt"):
        """Generate comprehensive text summary report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PKD-SKIP GRID SEARCH RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")

            # Overall best model
            f.write("OVERALL BEST MODEL\n")
            f.write("-"*70 + "\n")
            best = self.find_best_overall()
            f.write(f"Run: {best['run_name']}\n")
            f.write(f"Student Layers: {best['student_layers']}\n")
            f.write(f"Beta: {best['beta']}\n")
            f.write(f"Best Epoch: {best['epoch']}\n")
            f.write(f"Dev Accuracy: {best['dev_accuracy']:.4f}\n")
            f.write(f"Dev Loss: {best['dev_loss']:.4f}\n\n")

            # Best per student size
            f.write("BEST MODEL PER STUDENT SIZE\n")
            f.write("-"*70 + "\n")
            best_per_size = self.find_best_per_student_size()
            for layers in sorted(best_per_size.keys(), reverse=True):
                result = best_per_size[layers]
                f.write(f"\n{layers}-layer student:\n")
                f.write(f"  Run: {result['run_name']}\n")
                f.write(f"  Best Beta: {result['beta']}\n")
                f.write(f"  Best Epoch: {result['epoch']}\n")
                f.write(f"  Dev Accuracy: {result['dev_accuracy']:.4f}\n")
                f.write(f"  Dev Loss: {result['dev_loss']:.4f}\n")

            # Best per beta
            f.write("\n" + "="*70 + "\n")
            f.write("BEST MODEL PER BETA VALUE\n")
            f.write("-"*70 + "\n")
            best_per_beta = self.find_best_per_beta()
            for beta in sorted(best_per_beta.keys()):
                result = best_per_beta[beta]
                f.write(f"\nBeta = {beta}:\n")
                f.write(f"  Run: {result['run_name']}\n")
                f.write(f"  Student Layers: {result['student_layers']}\n")
                f.write(f"  Best Epoch: {result['epoch']}\n")
                f.write(f"  Dev Accuracy: {result['dev_accuracy']:.4f}\n")
                f.write(f"  Dev Loss: {result['dev_loss']:.4f}\n")

            # Convergence analysis
            f.write("\n" + "="*70 + "\n")
            f.write("CONVERGENCE ANALYSIS (Epoch 3 -> Epoch 4)\n")
            f.write("-"*70 + "\n")

            for run_name in sorted(self.df['run_name'].unique()):
                run_df = self.df[self.df['run_name'] == run_name].sort_values('epoch')
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


def plot_student_size_vs_beta_heatmap(df, output_path="results/plots/pkd_student_beta_heatmap.png"):
    """Create heatmap showing best accuracy for each student_size/beta combination"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get best accuracy for each student_size/beta combination (across all epochs)
    pivot_data = df.groupby(['student_layers', 'beta'])['dev_accuracy'].max().unstack()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Dev Accuracy'})
    plt.title('PKD-Skip: Best Dev Accuracy by Student Size and Beta\n(Best across all epochs)', fontsize=14, fontweight='bold')
    plt.xlabel('Beta (Patient Loss Weight)', fontsize=12)
    plt.ylabel('Student Layers', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def plot_learning_curves_by_student_size(df, output_path="results/plots/pkd_learning_curves_by_size.png"):
    """Plot learning curves grouped by student size"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    student_sizes = sorted(df['student_layers'].unique(), reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, layers in enumerate(student_sizes):
        ax = axes[idx]
        subset = df[df['student_layers'] == layers]

        for beta in sorted(subset['beta'].unique()):
            beta_data = subset[subset['beta'] == beta].sort_values('epoch')
            ax.plot(beta_data['epoch'], beta_data['dev_accuracy'],
                   marker='o', label=f'Beta={beta}', linewidth=2)

        ax.set_title(f'{layers}-Layer Student', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Dev Accuracy', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df['epoch'].unique()))

    plt.suptitle('PKD-Skip Learning Curves by Student Size', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Learning curves saved to: {output_path}")


def plot_beta_impact_across_sizes(df, output_path="results/plots/pkd_beta_impact.png"):
    """Plot how beta affects final accuracy across different student sizes"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get final epoch (epoch 4) results
    final_epoch_df = df[df['epoch'] == df['epoch'].max()]

    plt.figure(figsize=(12, 6))

    for layers in sorted(final_epoch_df['student_layers'].unique(), reverse=True):
        subset = final_epoch_df[final_epoch_df['student_layers'] == layers].sort_values('beta')
        plt.plot(subset['beta'], subset['dev_accuracy'],
                marker='o', label=f'{layers} layers', linewidth=2, markersize=8)

    plt.title('PKD-Skip: Beta Impact on Final Performance (Epoch 4)', fontsize=14, fontweight='bold')
    plt.xlabel('Beta (Patient Loss Weight)', fontsize=12)
    plt.ylabel('Dev Accuracy', fontsize=12)
    plt.legend(title='Student Size', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Log scale for beta
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Beta impact plot saved to: {output_path}")


def compare_with_vanilla_kd(pkd_csv="results/eval_results/pkd_skip_grid_search_eval.csv",
                            vanilla_csv="results/eval_results/vanilla_kd_grid_search_eval.csv",
                            output_path="results/eval_results/pkd_vs_vanilla_comparison.txt"):
    """Compare PKD results with vanilla KD baseline"""
    output_path = Path(output_path)

    # Load both datasets
    try:
        pkd_df = pd.read_csv(pkd_csv)
        vanilla_df = pd.read_csv(vanilla_csv)
    except FileNotFoundError as e:
        print(f"Warning: Could not load comparison data: {e}")
        return

    # Get best vanilla KD result (6-layer student)
    vanilla_6layer = vanilla_df[vanilla_df['run_name'].str.contains('L6')]
    best_vanilla = vanilla_6layer['dev_accuracy'].max()
    best_vanilla_run = vanilla_6layer.loc[vanilla_6layer['dev_accuracy'].idxmax(), 'run_name']

    # Get best PKD results per student size
    pkd_best_per_size = {}
    for layers in sorted(pkd_df['student_layers'].unique()):
        subset = pkd_df[pkd_df['student_layers'] == layers]
        best_acc = subset['dev_accuracy'].max()
        best_run = subset.loc[subset['dev_accuracy'].idxmax(), 'run_name']
        pkd_best_per_size[layers] = (best_acc, best_run)

    # Write comparison report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PKD-SKIP vs VANILLA KD COMPARISON\n")
        f.write("="*70 + "\n\n")

        f.write("VANILLA KD BASELINE (6-layer student)\n")
        f.write("-"*70 + "\n")
        f.write(f"Best Run: {best_vanilla_run}\n")
        f.write(f"Dev Accuracy: {best_vanilla:.4f}\n\n")

        f.write("PKD-SKIP RESULTS\n")
        f.write("-"*70 + "\n")
        for layers in sorted(pkd_best_per_size.keys(), reverse=True):
            acc, run = pkd_best_per_size[layers]
            improvement = acc - best_vanilla
            f.write(f"\n{layers}-layer student:\n")
            f.write(f"  Best Run: {run}\n")
            f.write(f"  Dev Accuracy: {acc:.4f}\n")
            if layers == 6:
                f.write(f"  vs Vanilla KD: {improvement:+.4f} ({improvement/best_vanilla*100:+.2f}%)\n")

        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n")

        # PKD 6-layer vs Vanilla 6-layer
        if 6 in pkd_best_per_size:
            pkd_6layer_acc = pkd_best_per_size[6][0]
            improvement = pkd_6layer_acc - best_vanilla
            f.write(f"1. PKD 6-layer vs Vanilla KD 6-layer: {improvement:+.4f}\n")
            if improvement > 0:
                f.write(f"   -> PKD provides {improvement*100:.2f}% absolute improvement\n")
            else:
                f.write(f"   -> Vanilla KD performs better by {abs(improvement)*100:.2f}%\n")

        # Smallest effective model
        smallest_better_than_vanilla = None
        for layers in sorted(pkd_best_per_size.keys()):
            if pkd_best_per_size[layers][0] >= best_vanilla:
                smallest_better_than_vanilla = layers
                break

        if smallest_better_than_vanilla:
            f.write(f"\n2. Smallest PKD model matching vanilla KD: {smallest_better_than_vanilla} layers\n")
            compression = (12 - smallest_better_than_vanilla) / 12 * 100
            f.write(f"   -> {compression:.1f}% model compression while maintaining performance\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Comparison report saved to: {output_path}")


def main():
    """Main execution function"""

    print("="*70)
    print("PKD-SKIP GRID SEARCH ANALYSIS")
    print("="*70)

    # 1. Load and analyze results
    analyzer = ResultsAnalyzer()

    # 2. Generate text summary
    print("\nGenerating summary report...")
    analyzer.generate_summary_report()

    # 3. Generate visualizations
    print("\nGenerating visualizations...")
    plot_student_size_vs_beta_heatmap(analyzer.df)
    plot_learning_curves_by_student_size(analyzer.df)
    plot_beta_impact_across_sizes(analyzer.df)

    # 4. Compare with vanilla KD
    print("\nGenerating comparison with vanilla KD...")
    compare_with_vanilla_kd()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/eval_results/pkd_best_models_summary.txt")
    print("  - results/eval_results/pkd_vs_vanilla_comparison.txt")
    print("  - results/plots/pkd_student_beta_heatmap.png")
    print("  - results/plots/pkd_learning_curves_by_size.png")
    print("  - results/plots/pkd_beta_impact.png")
    print("="*70)


if __name__ == "__main__":
    main()
