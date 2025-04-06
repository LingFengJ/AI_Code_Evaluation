import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from difflib import SequenceMatcher
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path):
    """Load the dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    def tokenize(text):
        return re.findall(r'[\w\']+|[.,!?;(){}[\]=<>:+\-*/]', text.lower())
    
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    
    matches = 0
    remaining_tokens = ref_tokens.copy()
    
    for token in pred_tokens:
        if token in remaining_tokens:
            matches += 1
            remaining_tokens.remove(token)  # Remove to avoid double counting
    
    denominator = max(len(ref_tokens), 1)
    return matches / denominator

def compute_diff_score(reference, prediction):
    """Compute similarity based on diff."""
    matcher = SequenceMatcher(None, reference, prediction)
    return matcher.ratio()

def generate_completion_with_model(model, tokenizer, prefix, max_new_tokens=100, device="cuda"):
    """Generate code completion using HuggingFace model."""
    try:
        # Prepare inputs
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True).to(device)
        
        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new tokens (the completion)
        completion = full_output[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return completion
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return ""

def evaluate_dataset(model, tokenizer, dataset, device="cuda", max_new_tokens=100):
    """Evaluate the model on the dataset."""
    model = model.to(device)
    model.eval()
    
    results = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
        prefix = sample['prefix']
        reference = sample['middle']
        
        # Generate completion
        try:
            completion = generate_completion_with_model(
                model, 
                tokenizer, 
                prefix, 
                max_new_tokens=max_new_tokens,
                device=device
            )
            
            # Compute metrics
            exact_match = compute_exact_match(reference, completion)
            line_overlap = compute_line_overlap(reference, completion)
            token_match = compute_token_match(reference, completion)
            diff_score = compute_diff_score(reference, completion)
            
            # Store results
            sample_result = {
                'sample_id': i,
                'file_path': sample.get('file_path', 'Unknown'),
                'prefix': prefix,
                'reference': reference,
                'completion': completion,
                'metrics': {
                    'exact_match': exact_match,
                    'line_overlap': line_overlap,
                    'token_match': token_match,
                    'diff_score': diff_score
                }
            }
            
            results.append(sample_result)
            
            # Log progress periodically
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            results.append({
                'sample_id': i,
                'file_path': sample.get('file_path', 'Unknown'),
                'error': str(e)
            })
    
    return results

def compute_overall_metrics(results):
    """Compute overall metrics across all samples."""
    metrics = {
        'exact_match': [],
        'line_overlap': [],
        'token_match': [],
        'diff_score': []
    }
    
    for result in results:
        if 'metrics' in result:
            for metric, value in result['metrics'].items():
                metrics[metric].append(value)
    
    # Calculate averages
    overall_metrics = {
        metric: np.mean(values) for metric, values in metrics.items() if values
    }
    
    return overall_metrics

def export_for_annotation(results, output_csv):
    """Export results to a CSV file for manual annotation."""
    data = []
    for result in results:
        if 'metrics' in result:
            row = {
                'sample_id': result['sample_id'],
                'file_path': result['file_path'],
                'exact_match': result['metrics']['exact_match'],
                'line_overlap': result['metrics']['line_overlap'],
                'token_match': result['metrics']['token_match'],
                'diff_score': result['metrics']['diff_score'],
                'human_score': None,  # To be filled manually
                'human_comments': None  # To be filled manually
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logger.info(f"Exported {len(data)} samples to {output_csv} for manual annotation")

def main():
    parser = argparse.ArgumentParser(description="Evaluate code completion models on a dataset")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID (e.g., 'bigcode/tiny_starcoder')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset JSON file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save the evaluation results")
    parser.add_argument("--csv", type=str, default="for_annotation.csv",
                        help="Path to save the CSV for annotation")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--half_precision", action="store_true", 
                        help="Use half precision (float16) for faster inference")
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model_kwargs = {}
        
        if args.half_precision and args.device == "cuda":
            logger.info("Using half precision (float16)")
            model_kwargs["torch_dtype"] = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        
        # Check if the model supports the device
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
        
        # Evaluate dataset
        logger.info(f"Starting evaluation on {args.device}")
        results = evaluate_dataset(
            model, 
            tokenizer, 
            dataset, 
            device=args.device,
            max_new_tokens=args.max_new_tokens
        )
        
        # Compute overall metrics
        overall_metrics = compute_overall_metrics(results)
        logger.info("Overall metrics:")
        for metric, value in overall_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save results
        output = {
            'model_id': args.model_id,
            'overall_metrics': overall_metrics,
            'samples': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output}")
        
        # Export for annotation
        export_for_annotation(results, args.csv)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if "401" in str(e):
            logger.error("Authentication error: You may need to login to Hugging Face")
            logger.error("Run 'huggingface-cli login' or use a public model")

if __name__ == "__main__":
    main()


'''

# Option 1: Using tiny_starcoder (small but specialized code model)
python huggingface_evaluation.py --model_id "bigcode/tiny_starcoder_py" --dataset "code_completion_dataset2.json" --output "starcoder_results.json"

# Option 2: Using CodeLlama (if you have enough GPU memory)
python huggingface_evaluation.py --model_id "codellama/CodeLlama-7b-hf" --dataset "code_completion_dataset2.json" --output "codellama_results.json" --half_precision

# Option 3: If you have limited GPU memory, try starcoder-1b
python huggingface_evaluation.py --model_id "bigcode/starcoder-1b" --dataset "code_completion_dataset2.json" --output "starcoder1b_results.json" --half_precision


# Try these models instead:
python huggingface_evaluation.py --model_id "bigcode/santacoder" --dataset "code_completion_dataset2.json" --output "santacoder_results.json"

# Or a smaller model:
python huggingface_evaluation.py --model_id "Salesforce/codegen-350M-mono" --dataset "code_completion_dataset2.json" --output "codegen_results.json"
'''