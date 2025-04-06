import json
import requests
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import difflib


def load_dataset(dataset_path):
    """Load the dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_claude_completion(prefix, api_key, max_tokens=250):
    """
    Generate code completion using Claude API.
    
    Args:
        prefix (str): Code prefix to complete
        api_key (str): Claude API key
        max_tokens (int): Maximum tokens to generate
        
    Returns:
        str: Generated completion
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",  # Add this line
        "content-type": "application/json"
    }
    
    # Format the prompt for code completion
    prompt = f"""Complete the following code snippet. Only return the code that would come next, without any explanations or comments about what you're doing:

```
{prefix}
```"""
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,  # Use deterministic outputs for better evaluation
    }
    
    # Make API request with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                completion = response.json()["content"][0]["text"]
                # Remove any markdown formatting if present
                if completion.startswith("```") and "```" in completion[3:]:
                    language_marker = completion[3:completion[3:].find("\n")+3]
                    code_body = completion[len(language_marker)+4:]
                    completion = code_body[:code_body.rfind("```")].strip()
                return completion
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit hit, waiting before retry ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"API error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Exception during API call: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    # If all retries failed or other error
    return "// Error generating completion"


def generate_completions(dataset, api_key, num_samples=None):
    """
    Generate completions for the dataset using Claude API.
    
    Args:
        dataset (list): The dataset of code samples
        api_key (str): Claude API key
        num_samples (int): Number of samples to use (None for all)
        
    Returns:
        list: Dataset with added completions
    """
    if num_samples:
        dataset = dataset[:num_samples]
    
    for sample in tqdm(dataset, desc="Generating completions"):
        # Get the prefix
        prefix = sample['prefix']
        
        # Truncate very long prefixes to avoid excessive token usage
        if len(prefix) > 4000:
            prefix = prefix[-4000:]
            
        # Generate completion
        completion = generate_claude_completion(prefix, api_key)
        
        # Store the completion
        sample['completion'] = completion
    
    return dataset


def compute_exact_match(reference, prediction):
    """Compute exact match accuracy."""
    return 1.0 if reference.strip() == prediction.strip() else 0.0


def compute_line_overlap(reference, prediction):
    """Compute line-by-line overlap percentage."""
    ref_lines = reference.strip().split('\n')
    pred_lines = prediction.strip().split('\n')
    
    # Count lines that appear in both
    common_lines = set(ref_lines).intersection(set(pred_lines))
    
    # Compute percentage based on reference length
    denominator = max(len(ref_lines), 1)
    return len(common_lines) / denominator


def compute_token_match(reference, prediction):
    """Compute token-wise match percentage."""
    import re
    
    # Tokenize by splitting on whitespace and punctuation
    ref_tokens = re.findall(r'[\w\']+|[.,!?;(){}[\]=<>:+\-*/]', reference.lower())
    pred_tokens = re.findall(r'[\w\']+|[.,!?;(){}[\]=<>:+\-*/]', prediction.lower())
    
    matches = 0
    for token in pred_tokens:
        if token in ref_tokens:
            matches += 1
            ref_tokens.remove(token)  # Remove to avoid double counting
    
    denominator = max(len(re.findall(r'[\w\']+|[.,!?;(){}[\]=<>:+\-*/]', reference.lower())), 1)
    return matches / denominator


def compute_diff_score(reference, prediction):
    """Compute similarity based on diff."""
    matcher = difflib.SequenceMatcher(None, reference, prediction)
    return matcher.ratio()


def evaluate_dataset(dataset):
    """
    Evaluate all completions in the dataset using multiple metrics.
    
    Args:
        dataset (list): Dataset with completions
        
    Returns:
        dict: Metrics for each sample and overall averages
    """
    metrics = {
        'exact_match': [],
        'line_overlap': [],
        'token_match': [],
        'diff_score': []
    }
    
    for sample in dataset:
        reference = sample['middle']
        prediction = sample['completion']
        
        # Compute metrics
        exact = compute_exact_match(reference, prediction)
        line_overlap = compute_line_overlap(reference, prediction)
        token = compute_token_match(reference, prediction)
        diff = compute_diff_score(reference, prediction)
        
        # Store per-sample metrics
        sample['metrics'] = {
            'exact_match': exact,
            'line_overlap': line_overlap,
            'token_match': token,
            'diff_score': diff
        }
        
        # Add to overall metrics
        metrics['exact_match'].append(exact)
        metrics['line_overlap'].append(line_overlap)
        metrics['token_match'].append(token)
        metrics['diff_score'].append(diff)
    
    # Calculate averages
    overall_metrics = {
        'exact_match': np.mean(metrics['exact_match']),
        'line_overlap': np.mean(metrics['line_overlap']),
        'token_match': np.mean(metrics['token_match']),
        'diff_score': np.mean(metrics['diff_score'])
    }
    
    return {
        'per_sample': metrics,
        'overall': overall_metrics
    }


def export_for_annotation(dataset, output_path):
    """
    Export the evaluation results to a CSV file for manual annotation.
    
    Args:
        dataset (list): Dataset with completions and metrics
        output_path (str): Path to save the CSV file
    """
    data = []
    for i, sample in enumerate(dataset):
        # Get truncated versions of long strings for better CSV readability
        prefix_snippet = sample['prefix'][-200:] if len(sample['prefix']) > 200 else sample['prefix']
        
        # Prepare sample data
        row = {
            'sample_id': i,
            'file_path': sample.get('file_path', 'Unknown'),
            'prefix_snippet': prefix_snippet,
            'actual_completion': sample['middle'],
            'model_completion': sample['completion'],
            'exact_match': sample['metrics']['exact_match'],
            'line_overlap': sample['metrics']['line_overlap'],
            'token_match': sample['metrics']['token_match'],
            'diff_score': sample['metrics']['diff_score'],
            'human_correct': None,  # To be filled manually
            'human_comments': None  # To be filled manually
        }
        data.append(row)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(data)} samples to {output_path} for manual annotation")


def analyze_annotated_results(annotated_csv):
    """
    Analyze correlation between automatic metrics and human judgment.
    
    Args:
        annotated_csv (str): Path to the annotated CSV file
        
    Returns:
        dict: Correlation analysis results
    """
    from scipy.stats import pearsonr, spearmanr
    import matplotlib.pyplot as plt
    
    # Load annotated data
    df = pd.read_csv(annotated_csv)
    
    # Skip if no human annotations
    if df['human_correct'].isna().all():
        print("No human annotations found. Please complete annotations first.")
        return None
    
    # Remove rows without annotations
    df = df.dropna(subset=['human_correct'])
    
    # Calculate correlations
    metrics = ['exact_match', 'line_overlap', 'token_match', 'diff_score']
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for metric in metrics:
        # Skip if metric has no variation
        if len(df[metric].unique()) <= 1:
            results[metric] = {
                'pearson': None,
                'spearman': None,
                'message': "Metric has no variation"
            }
            continue
        
        # Calculate Pearson correlation
        pearson, p_pearson = pearsonr(df[metric], df['human_correct'])
        
        # Calculate Spearman correlation
        spearman, p_spearman = spearmanr(df[metric], df['human_correct'])
        
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
    
    # Print results
    print("\nCorrelation Analysis:")
    for metric, data in results.items():
        if data.get('message'):
            print(f"  {metric}: {data['message']}")
        else:
            print(f"  {metric}:")
            print(f"    Pearson: r={data['pearson']['correlation']:.4f} (p={data['pearson']['p_value']:.4f})")
            print(f"    Spearman: r={data['spearman']['correlation']:.4f} (p={data['spearman']['p_value']:.4f})")
    
    # Create visualization of Pearson correlations
    plt.figure(figsize=(10, 6))
    
    pearson_values = []
    metric_names = []
    
    for metric, data in results.items():
        if data.get('pearson') and data['pearson']['correlation'] is not None:
            pearson_values.append(data['pearson']['correlation'])
            metric_names.append(metric)
    
    plt.bar(metric_names, pearson_values)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.ylim(-1, 1)
    plt.title('Correlation between Metrics and Human Judgment')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.savefig('correlation_analysis.png')
    
    return results


def main(args):
    if args.annotated:
        # Analyze annotated results
        analyze_annotated_results(args.annotated)
        return
    
    # Get API key
    api_key = args.api_key or os.environ.get('CLAUDE_API_KEY')
    if not api_key:
        print("Claude API key is required. Provide it with --api_key or set the CLAUDE_API_KEY environment variable.")
        return
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    
    # Generate completions
    dataset = generate_completions(dataset, api_key, args.num_samples)
    print(f"Generated completions for {len(dataset)} samples")
    
    # Compute metrics
    metrics = evaluate_dataset(dataset)
    print("\nOverall Metrics:")
    for metric, value in metrics['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'metrics': metrics['overall'],
        'samples': dataset
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")
    
    # Export for annotation
    export_for_annotation(dataset, args.csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude API Code Completion Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output", type=str, default="claude_evaluation_results.json", help="Path to save the results")
    parser.add_argument("--csv", type=str, default="for_annotation.csv", help="Path to save the CSV for annotation")
    parser.add_argument("--api_key", type=str, default=None, help="Claude API key")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--annotated", type=str, default=None, help="Path to annotated CSV for analysis")
    
    args = parser.parse_args()
    main(args)



'''
python claude-evaluation.py --dataset code_completion_dataset2.json --api_key "your_anthropic_api_key" 
'''