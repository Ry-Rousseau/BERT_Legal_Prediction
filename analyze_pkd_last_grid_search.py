"""
Analysis and Visualization Script for PKD-Last Grid Search Results

This script analyzes the evaluation results from the PKD-Last grid search,
generates summary reports, and creates visualizations.

Since PKD-Last uses a fixed beta=500, analysis focuses on:
- Best model per student size
- Learning curves across epochs
- Comparison with PKD-Skip and vanilla KD baselines
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
    """Analyzes PKD-Last grid search evaluation results"""

    def __init__(self, csv_path="results/eval_results/pkd_last_grid_search_eval.csv"):
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
                'epoch': best_row['epoch'],
                'dev_accuracy': best_row['dev_accuracy'],
                'dev_loss': best_row['dev_loss']
            }
        return results

    def generate_summary_report(self, output_path="results/eval_results/pkd_last_best_models_summary.txt"):
        """Generate comprehensive text summary report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PKD-LAST GRID SEARCH RESULTS SUMMARY\n")
            f.write("="*70 + "\n")
            f.write("Strategy: Last N layers matching\n")
            f.write("Fixed Beta: 500\n")
            f.write("Fixed Alpha: 0.7, Temperature: 20\n\n")

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


def plot_learning_curves(df, output_path="results/plots/pkd_last_learning_curves.png"):
    """Plot learning curves for all student sizes"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df['student_layers'].unique())))

    for idx, layers in enumerate(sorted(df['student_layers'].unique(), reverse=True)):
        subset = df[df['student_layers'] == layers].sort_values('epoch')
        plt.plot(subset['epoch'], subset['dev_accuracy'],
                marker='o', label=f'{layers} layers', linewidth=2.5,
                markersize=8, color=colors[idx])

    plt.title('PKD-Last: Learning Curves Across Student Sizes\n(Beta=500, Last-N Layer Matching)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dev Accuracy', fontsize=12)
    plt.legend(title='Student Size', fontsize=11, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df['epoch'].unique()))

    # Add horizontal line for teacher baseline (if known)
    # plt.axhline(y=0.7548, color='red', linestyle='--', linewidth=2, label='Teacher (12-layer)', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Learning curves saved to: {output_path}")


def plot_final_performance_comparison(df, output_path="results/plots/pkd_last_final_performance.png"):
    """Plot final epoch performance across student sizes"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get final epoch results
    final_epoch = df['epoch'].max()
    final_df = df[df['epoch'] == final_epoch].sort_values('student_layers', ascending=False)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(final_df['student_layers'].astype(str),
                   final_df['dev_accuracy'],
                   color=plt.cm.viridis(np.linspace(0, 0.9, len(final_df))),
                   edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title(f'PKD-Last: Final Performance by Student Size (Epoch {final_epoch})\n(Beta=500)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Student Layers', fontsize=12)
    plt.ylabel('Dev Accuracy', fontsize=12)
    plt.ylim(min(final_df['dev_accuracy']) * 0.98, max(final_df['dev_accuracy']) * 1.01)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Final performance plot saved to: {output_path}")


def compare_strategies(pkd_last_csv="results/eval_results/pkd_last_grid_search_eval.csv",
                       pkd_skip_csv="results/eval_results/pkd_skip_grid_search_eval.csv",
                       vanilla_csv="results/eval_results/vanilla_kd_grid_search_eval.csv",
                       output_path="results/eval_results/pkd_last_vs_skip_comparison.txt"):
    """Compare PKD-Last with PKD-Skip and vanilla KD"""
    output_path = Path(output_path)

    # Load PKD-Last results
    try:
        last_df = pd.read_csv(pkd_last_csv)
    except FileNotFoundError:
        print(f"Warning: PKD-Last results not found at {pkd_last_csv}")
        return

    # Load PKD-Skip results (optional)
    try:
        skip_df = pd.read_csv(pkd_skip_csv)
        has_skip = True
    except FileNotFoundError:
        print(f"Warning: PKD-Skip results not found, comparison will be limited")
        has_skip = False
        skip_df = None

    # Load Vanilla KD results (optional)
    try:
        vanilla_df = pd.read_csv(vanilla_csv)
        has_vanilla = True
    except FileNotFoundError:
        print(f"Warning: Vanilla KD results not found, comparison will be limited")
        has_vanilla = False
        vanilla_df = None

    # Get best PKD-Last results per student size
    last_best = {}
    for layers in sorted(last_df['student_layers'].unique()):
        subset = last_df[last_df['student_layers'] == layers]
        best_acc = subset['dev_accuracy'].max()
        best_run = subset.loc[subset['dev_accuracy'].idxmax()]
        last_best[layers] = {
            'accuracy': best_acc,
            'run': best_run['run_name'],
            'epoch': best_run['epoch']
        }

    # Get best PKD-Skip results per student size (fixed beta=500)
    skip_best = {}
    if has_skip:
        for layers in sorted(skip_df['student_layers'].unique()):
            subset = skip_df[(skip_df['student_layers'] == layers) & (skip_df['beta'] == 500)]
            if len(subset) > 0:
                best_acc = subset['dev_accuracy'].max()
                best_run = subset.loc[subset['dev_accuracy'].idxmax()]
                skip_best[layers] = {
                    'accuracy': best_acc,
                    'run': best_run['run_name'],
                    'epoch': best_run['epoch']
                }

    # Get vanilla KD baseline (6-layer)
    vanilla_baseline = None
    if has_vanilla:
        vanilla_6layer = vanilla_df[vanilla_df['run_name'].str.contains('L6')]
        if len(vanilla_6layer) > 0:
            vanilla_baseline = vanilla_6layer['dev_accuracy'].max()

    # Write comparison report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PKD-LAST vs PKD-SKIP vs VANILLA KD COMPARISON\n")
        f.write("="*70 + "\n\n")

        if has_vanilla and vanilla_baseline:
            f.write("VANILLA KD BASELINE (6-layer student)\n")
            f.write("-"*70 + "\n")
            f.write(f"Dev Accuracy: {vanilla_baseline:.4f}\n\n")

        f.write("PKD-LAST STRATEGY (Last-N Layer Matching, Beta=500)\n")
        f.write("-"*70 + "\n")
        for layers in sorted(last_best.keys(), reverse=True):
            result = last_best[layers]
            f.write(f"\n{layers}-layer student:\n")
            f.write(f"  Best Run: {result['run']}\n")
            f.write(f"  Best Epoch: {result['epoch']}\n")
            f.write(f"  Dev Accuracy: {result['accuracy']:.4f}\n")

            if has_vanilla and vanilla_baseline:
                improvement = result['accuracy'] - vanilla_baseline
                f.write(f"  vs Vanilla KD: {improvement:+.4f} ({improvement/vanilla_baseline*100:+.2f}%)\n")

        if has_skip:
            f.write("\n" + "="*70 + "\n")
            f.write("PKD-SKIP STRATEGY (Evenly-Spaced Matching, Beta=500)\n")
            f.write("-"*70 + "\n")
            for layers in sorted(skip_best.keys(), reverse=True):
                result = skip_best[layers]
                f.write(f"\n{layers}-layer student:\n")
                f.write(f"  Best Run: {result['run']}\n")
                f.write(f"  Best Epoch: {result['epoch']}\n")
                f.write(f"  Dev Accuracy: {result['accuracy']:.4f}\n")

        if has_skip:
            f.write("\n" + "="*70 + "\n")
            f.write("PKD-LAST vs PKD-SKIP (Beta=500 Comparison)\n")
            f.write("-"*70 + "\n")
            for layers in sorted(set(last_best.keys()) & set(skip_best.keys()), reverse=True):
                last_acc = last_best[layers]['accuracy']
                skip_acc = skip_best[layers]['accuracy']
                diff = last_acc - skip_acc

                winner = "LAST" if diff > 0 else "SKIP" if diff < 0 else "TIE"

                f.write(f"\n{layers}-layer student:\n")
                f.write(f"  PKD-Last: {last_acc:.4f}\n")
                f.write(f"  PKD-Skip: {skip_acc:.4f}\n")
                f.write(f"  Difference: {diff:+.4f} ({diff/skip_acc*100:+.2f}%)\n")
                f.write(f"  Winner: {winner}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n")

        # Finding 1: Best overall model
        best_overall_layers = max(last_best.keys(), key=lambda k: last_best[k]['accuracy'])
        f.write(f"1. Best PKD-Last model: {best_overall_layers}-layer student ")
        f.write(f"({last_best[best_overall_layers]['accuracy']:.4f} accuracy)\n")

        # Finding 2: Strategy comparison
        if has_skip:
            last_wins = sum(1 for layers in set(last_best.keys()) & set(skip_best.keys())
                           if last_best[layers]['accuracy'] > skip_best[layers]['accuracy'])
            skip_wins = sum(1 for layers in set(last_best.keys()) & set(skip_best.keys())
                           if skip_best[layers]['accuracy'] > last_best[layers]['accuracy'])

            f.write(f"\n2. Strategy comparison (Beta=500):\n")
            f.write(f"   PKD-Last wins: {last_wins} student sizes\n")
            f.write(f"   PKD-Skip wins: {skip_wins} student sizes\n")

            if last_wins > skip_wins:
                f.write(f"   -> PKD-Last (matching final layers) performs better overall\n")
            elif skip_wins > last_wins:
                f.write(f"   -> PKD-Skip (evenly-spaced matching) performs better overall\n")
            else:
                f.write(f"   -> Both strategies perform similarly\n")

        # Finding 3: Compression capability
        if has_vanilla and vanilla_baseline:
            smallest_better = None
            for layers in sorted(last_best.keys()):
                if last_best[layers]['accuracy'] >= vanilla_baseline:
                    smallest_better = layers
                    break

            if smallest_better:
                compression = (12 - smallest_better) / 12 * 100
                f.write(f"\n3. Smallest model matching vanilla KD: {smallest_better} layers\n")
                f.write(f"   -> {compression:.1f}% model compression while maintaining performance\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Comparison report saved to: {output_path}")


def main():
    """Main execution function"""

    print("="*70)
    print("PKD-LAST GRID SEARCH ANALYSIS")
    print("="*70)

    # 1. Load and analyze results
    analyzer = ResultsAnalyzer()

    # 2. Generate text summary
    print("\nGenerating summary report...")
    analyzer.generate_summary_report()

    # 3. Generate visualizations
    print("\nGenerating visualizations...")
    plot_learning_curves(analyzer.df)
    plot_final_performance_comparison(analyzer.df)

    # 4. Compare with PKD-Skip and vanilla KD
    print("\nGenerating strategy comparison...")
    compare_strategies()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/eval_results/pkd_last_best_models_summary.txt")
    print("  - results/eval_results/pkd_last_vs_skip_comparison.txt")
    print("  - results/plots/pkd_last_learning_curves.png")
    print("  - results/plots/pkd_last_final_performance.png")
    print("="*70)


if __name__ == "__main__":
    main()
