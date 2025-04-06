import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import argparse
import json

def load_annotations(annotations_path):
    """Load annotations from CSV file."""
    return pd.read_csv(annotations_path)

def analyze_correlations(annotations_df):
    """
    Analyze correlations between automatic metrics and human judgment.
    
    Args:
        annotations_df (DataFrame): DataFrame with annotations
        
    Returns:
        dict: Correlation results
    """
    # Identify metrics columns (exclude non-numeric and identifier columns)
    exclude_cols = ['sample_id', 'file_path', 'human_comments']
    metric_cols = [col for col in annotations_df.columns 
                   if col not in exclude_cols and col != 'human_score'
                   and pd.api.types.is_numeric_dtype(annotations_df[col])]
    
    results = {}
    
    # Calculate correlations for each metric
    for metric in metric_cols:
        # Skip columns with all NaN values
        if annotations_df[metric].isna().all():
            continue
            
        # Calculate correlations with human scores
        valid_data = annotations_df[[metric, 'human_score']].dropna()
        
        if len(valid_data) < 2:
            continue
            
        # Pearson correlation (linear relationship)
        pearson, p_pearson = pearsonr(valid_data[metric], valid_data['human_score'])
        
        # Spearman correlation (monotonic relationship)
        spearman, p_spearman = spearmanr(valid_data[metric], valid_data['human_score'])
        
        results[metric] = {
            'pearson': {
                'correlation': pearson,
                'p_value': p_pearson
            },
            'spearman': {
                'correlation': spearman,
                'p_value': p_spearman
            }
        }
    
    return results

def print_correlation_results(results):
    """Print correlation results in a formatted table."""
    print("\n=== Correlation Analysis ===")
    
    # Create headers
    headers = ["Metric", "Pearson r", "p-value", "Spearman r", "p-value"]
    row_format = "{:<15} {:>10} {:>10} {:>10} {:>10}"
    
    # Print header
    print(row_format.format(*headers))
    print("-" * 60)
    
    # Print each row
    for metric, data in results.items():
        pearson_r = data['pearson']['correlation']
        pearson_p = data['pearson']['p_value']
        spearman_r = data['spearman']['correlation']
        spearman_p = data['spearman']['p_value']
        
        print(row_format.format(
            metric,
            f"{pearson_r:.4f}",
            f"{pearson_p:.4f}",
            f"{spearman_r:.4f}",
            f"{spearman_p:.4f}"
        ))

def create_correlation_bar_chart(results, output_path):
    """Create a bar chart of correlation coefficients."""
    metrics = list(results.keys())
    pearson_values = [data['pearson']['correlation'] for data in results.values()]
    spearman_values = [data['spearman']['correlation'] for data in results.values()]
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position of bars
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, pearson_values, width, label='Pearson r', color='#3498db')
    ax.bar(x + width/2, spearman_values, width, label='Spearman r', color='#e74c3c')
    
    # Add details
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation between Metrics and Human Judgment')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(-1, 1)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Bar chart saved to {output_path}")
    return fig

def create_scatter_plots(annotations_df, results, output_path):
    """Create scatter plots of metrics vs human scores."""
    metrics = list(results.keys())
    
    # Determine grid size
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten axes array if necessary
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create scatter plots
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get data
        data = annotations_df[[metric, 'human_score']].dropna()
        
        # Create scatter plot with jitter
        x = data[metric]
        y = data['human_score']
        
        # Add some jitter to y values to show overlapping points better
        y_jitter = y + np.random.normal(0, 0.05, size=len(y))
        
        # Plot
        ax.scatter(x, y_jitter, alpha=0.6, c='#3498db')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8)
        
        # Add correlation coefficient to title
        pearson_r = results[metric]['pearson']['correlation']
        ax.set_title(f"{metric} vs Human Score (r={pearson_r:.4f})")
        
        # Set labels
        ax.set_xlabel(metric)
        ax.set_ylabel('Human Score')
        
        # Set y-axis limits
        ax.set_ylim(-0.5, 5.5)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Scatter plots saved to {output_path}")
    return fig

def save_correlation_results(results, output_path):
    """Save correlation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Correlation results saved to {output_path}")

def analyze_human_scores(annotations_df):
    """Analyze the distribution of human scores."""
    scores = annotations_df['human_score']
    
    # Calculate statistics
    stats = {
        'count': len(scores),
        'mean': scores.mean(),
        'median': scores.median(),
        'min': scores.min(),
        'max': scores.max(),
        'std': scores.std()
    }
    
    # Count each score
    counts = scores.value_counts().sort_index()
    distribution = {f"score_{int(score)}": count for score, count in counts.items()}
    
    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=np.arange(-0.5, 6, 1), alpha=0.7, color='#3498db')
    plt.xticks(range(6))
    plt.xlabel('Human Score')
    plt.ylabel('Count')
    plt.title('Distribution of Human Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('human_score_distribution.png')
    print("Human score distribution saved to human_score_distribution.png")
    
    return {'stats': stats, 'distribution': distribution}

def main():
    parser = argparse.ArgumentParser(description="Analyze correlations between metrics and human judgment")
    parser.add_argument("--annotations", type=str, default="manual_annotations.csv",
                        help="Path to the annotations CSV file")
    parser.add_argument("--output_prefix", type=str, default="correlation_analysis",
                        help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Load annotations
    annotations_df = load_annotations(args.annotations)
    print(f"Loaded {len(annotations_df)} annotations from {args.annotations}")
    
    # Analyze human scores
    score_analysis = analyze_human_scores(annotations_df)
    print("\n=== Human Score Analysis ===")
    for key, value in score_analysis['stats'].items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Analyze correlations
    results = analyze_correlations(annotations_df)
    
    # Print results
    print_correlation_results(results)
    
    # Create visualizations
    create_correlation_bar_chart(results, f"{args.output_prefix}_bar_chart.png")
    create_scatter_plots(annotations_df, results, f"{args.output_prefix}_scatter_plots.png")
    
    # Save results
    save_correlation_results(results, f"{args.output_prefix}_results.json")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()


'''
python analyze_correlations.py --annotations manual_annotations.csv


python analyze_correlations.py --annotations "tiny_starcoder_annotations.csv" --output_prefix "tiny_starcoder_analysis"



python analyze_correlations.py --annotations codegen_annotations.csv --output_prefix codegen_analysis
python analyze_correlations.py --annotations starcoder_annotations.csv --output_prefix starcoder_analysis

python analyze_correlations.py --annotations claude_annotations.csv --output_prefix claude_analysis
'''