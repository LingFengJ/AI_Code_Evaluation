import json
import pandas as pd
import argparse
import os
from difflib import unified_diff
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init()

def load_results(results_path):
    """Load the evaluation results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_annotations(annotations, output_path):
    """Save annotations to CSV file."""
    df = pd.DataFrame(annotations)
    df.to_csv(output_path, index=False)
    print(f"Annotations saved to {output_path}")

def display_code(code, title, max_lines=20):
    """Display code with title, potentially truncating long code."""
    print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
    print("```")
    
    lines = code.split('\n')
    if len(lines) > max_lines:
        # Show first and last few lines
        first_lines = lines[:max_lines//2]
        last_lines = lines[-max_lines//2:]
        print('\n'.join(first_lines))
        print(f"{Fore.YELLOW}... ({len(lines) - max_lines} more lines) ...{Style.RESET_ALL}")
        print('\n'.join(last_lines))
    else:
        print(code)
    
    print("```")

def get_colored_diff(original, completion):
    """Get a colored diff between original and completion."""
    diff_lines = list(unified_diff(
        original.splitlines(),
        completion.splitlines(),
        lineterm='',
        n=2
    ))
    
    colored_lines = []
    for line in diff_lines[2:]:  # Skip the header lines
        if line.startswith('+'):
            colored_lines.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        elif line.startswith('-'):
            colored_lines.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
        elif line.startswith('@@'):
            colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        else:
            colored_lines.append(line)
    
    return '\n'.join(colored_lines)

def display_metrics(metrics):
    """Display metrics in a table format."""
    if not metrics:
        return
    
    print(f"\n{Fore.CYAN}METRICS{Style.RESET_ALL}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

def annotate_results(results_path, output_path, start_from=0):
    """Interactively annotate completion results."""
    # Load results
    results = load_results(results_path)
    samples = results.get('samples', [])
    
    if not samples:
        print("No samples found in the results file.")
        return []
    
    # Load existing annotations if file exists
    annotations = []
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        annotations = existing_df.to_dict('records')
        print(f"Loaded {len(annotations)} existing annotations.")
    
    # Calculate how many samples are left to annotate
    annotated_ids = {a.get('sample_id') for a in annotations if 'sample_id' in a}
    total_remaining = len(samples) - len(annotated_ids)
    
    print(f"\n{Fore.GREEN}=== Code Completion Annotation Tool ==={Style.RESET_ALL}")
    print(f"Found {len(samples)} samples, {total_remaining} remaining to annotate.")
    
    # Start from the specified index
    i = max(0, start_from)
    while i < len(samples):
        sample = samples[i]
        
        # Skip if already annotated
        if i in annotated_ids:
            i += 1
            continue
        
        print(f"\n\n{Fore.GREEN}=== Sample {i+1}/{len(samples)} ==={Style.RESET_ALL}")
        print(f"File: {sample.get('file_path', 'Unknown')}")
        
        # Display the code sections
        display_code(sample.get('prefix', '')[-500:], "PREFIX (last 500 chars)")
        display_code(sample.get('middle', ''), "EXPECTED (MIDDLE)")
        display_code(sample.get('completion', ''), "COMPLETION")
        
        # Display colored diff
        print(f"\n{Fore.CYAN}DIFF{Style.RESET_ALL}")
        print(get_colored_diff(sample.get('middle', ''), sample.get('completion', '')))
        
        # Display metrics
        display_metrics(sample.get('metrics', {}))
        
        # Get annotation
        print("\nAnnotation Guidelines:")
        print("0 - Completely incorrect, irrelevant, or harmful")
        print("1 - Major errors, but some relevant elements")
        print("2 - Partially correct but significant issues")
        print("3 - Mostly correct with minor issues")
        print("4 - Correct and good quality")
        print("5 - Excellent, matches or improves upon the expected solution")
        
        while True:
            try:
                score_input = input(f"\n{Fore.YELLOW}Score (0-5, n=next, p=prev, q=quit):{Style.RESET_ALL} ")
                
                if score_input.lower() == 'n':
                    i += 1
                    break
                elif score_input.lower() == 'p':
                    i = max(0, i-1)
                    break
                elif score_input.lower() == 'q':
                    print("Quitting annotation process.")
                    save_annotations(annotations, output_path)
                    return annotations
                
                score = float(score_input)
                if 0 <= score <= 5:
                    comments = input(f"{Fore.YELLOW}Comments (optional):{Style.RESET_ALL} ")
                    
                    annotation = {
                        'sample_id': i,
                        'file_path': sample.get('file_path', 'Unknown'),
                        'human_score': score,
                        'human_comments': comments
                    }
                    
                    # Add metrics if available
                    if 'metrics' in sample:
                        for metric, value in sample['metrics'].items():
                            annotation[metric] = value
                    
                    annotations.append(annotation)
                    i += 1
                    break
                else:
                    print("Please enter a score between 0 and 5.")
            except ValueError:
                print("Please enter a valid number or command.")
        
        # Save progress periodically
        if len(annotations) % 5 == 0:
            save_annotations(annotations, output_path)
            print(f"Progress saved. {len(samples) - i} samples remaining.")
    
    # Final save
    save_annotations(annotations, output_path)
    print(f"Annotation complete! All {len(annotations)} annotations saved.")
    return annotations

def main():
    parser = argparse.ArgumentParser(description="Manually annotate code completion results")
    parser.add_argument("--results", type=str, default="claude_evaluation_results.json",
                        help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, default="manual_annotations.csv",
                        help="Path to save the annotations CSV file")
    parser.add_argument("--start", type=int, default=0,
                        help="Sample index to start annotation from")
    
    args = parser.parse_args()
    
    annotate_results(args.results, args.output, args.start)

if __name__ == "__main__":
    main()


'''
python annotate_completions.py --results claude_evaluation_results.json --output manual_annotations.csv

python annotate_completions.py --results "tiny_starcoder_results.json" --output "tiny_starcoder_annotations.csv"


python annotate_completions.py --results "codegen_results.json" --output "codegen_annotations.csv"
'''