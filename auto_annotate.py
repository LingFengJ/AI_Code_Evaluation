import json
import pandas as pd
import requests
import time
import os
import argparse
from tqdm import tqdm

def call_claude(message, api_key):
    """Call Claude API to evaluate code completion."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    prompt = f"""You're a code completion evaluator. You'll be shown:
1. The expected code (ground truth)
2. The model-generated completion

Rate the completion on a scale from 0-5 where:
0 - Completely incorrect or irrelevant
1 - Major errors, but some relevant elements
2 - Partially correct with significant issues
3 - Mostly correct with minor issues
4 - Correct and good quality
5 - Excellent, matches or improves upon expected solution

Only respond with a rating number (0-5) and a brief explanation. For example:
"4 - Completion is correct and follows good practices. It matches the expected functionality."

Expected code:
```python
{message['expected']}
```

Model completion:
```python
{message['completion']}
```

Please rate this completion:"""
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    
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
                return response.json()["content"][0]["text"].strip()
            elif response.status_code == 429:
                print(f"Rate limit hit, waiting before retry ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"API error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Exception during API call: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    return "0 - Error evaluating completion"

def extract_rating(claude_response):
    """Extract numerical rating from Claude's response."""
    try:
        # Extract the first digit from the response
        for char in claude_response:
            if char.isdigit() and int(char) in range(6):
                return int(char)
        return 3  # Default to middle rating if parsing fails
    except:
        return 3

def parse_comment(claude_response):
    """Extract comment from Claude's response."""
    try:
        # Remove the rating digit and dash
        parts = claude_response.split("-", 1)
        if len(parts) > 1:
            return parts[1].strip()
        return claude_response
    except:
        return claude_response

def auto_annotate_results(results_file, output_csv, api_key):
    """Automatically annotate code completions using Claude."""
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    samples = results.get('samples', [])
    
    if not samples:
        print("No samples found in the results file.")
        return
    
    # Prepare data for annotations
    annotations = []
    
    for i, sample in enumerate(tqdm(samples, desc="Annotating samples")):
        # Skip if no completion or reference
        if 'completion' not in sample:
            continue
            
        reference = sample.get('reference', '')
        if not reference and 'middle' in sample:
            reference = sample['middle']
        
        if not reference:
            continue
        
        # Prepare message for Claude
        message = {
            "expected": reference,
            "completion": sample.get('completion', '')
        }
        
        # Get Claude's evaluation
        claude_response = call_claude(message, api_key)
        
        # Extract score and comment
        score = extract_rating(claude_response)
        comment = parse_comment(claude_response)
        
        # Create annotation entry
        annotation = {
            'sample_id': i,
            'file_path': sample.get('file_path', 'Unknown'),
            'human_score': score,
            'human_comments': comment
        }
        
        # Add metrics if available
        if 'metrics' in sample:
            for metric, value in sample['metrics'].items():
                annotation[metric] = value
        
        annotations.append(annotation)
        
        # Avoid rate limiting
        time.sleep(0.5)
    
    # Save to CSV
    df = pd.DataFrame(annotations)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(annotations)} annotations to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Automatically annotate code completions using Claude")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save the annotations CSV file")
    parser.add_argument("--api_key", type=str, default=None,
                       help="Claude API key (or set CLAUDE_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('CLAUDE_API_KEY')
    if not api_key:
        print("Claude API key is required. Provide it with --api_key or set the CLAUDE_API_KEY environment variable.")
        return
    
    auto_annotate_results(args.results, args.output, api_key)

if __name__ == "__main__":
    main()


'''
# For CodeGen results
python auto_annotate.py --results codegen_results.json --output codegen_annotations.csv --api_key "you_anthropic_api_key"

# For TinyStarCoder results
python auto_annotate.py --results starcoder_results.json --output starcoder_annotations.csv --api_key "you_anthropic_api_key"

# For Claude's own results (optional, for comparison)
python auto_annotate.py --results claude_evaluation_results.json --output claude_annotations.csv --api_key "you_anthropic_api_key"
'''