# Code Completion Model Evaluation

This repository contains tools to evaluate code completion models using a dataset created from your own code repositories. The pipeline allows you to:

1. Create a dataset of code completion samples
2. Evaluate different models on this dataset
3. Analyze results against human judgments
4. Determine which automatic metrics best correlate with human assessment


## Dataset and Report
The Dataset, annotations, and evaluation results with Report can be found  [here](https://drive.google.com/drive/folders/1eS_ftnlDT3IHfeC9YSwtP3ZdxM2BPu8j)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- CUDA-capable GPU (recommended for model evaluation)

```bash
pip install torch transformers pandas matplotlib scipy colorama tqdm difflib numpy
```

## Quick Start

```bash
# 1. Create a dataset from your own code
python code-splitter2.py --directory /path/to/your/code --output code_completion_dataset.json

# 2. Evaluate a model
python huggingface_evaluation.py --model_id "Salesforce/codegen-350M-mono" --dataset "code_completion_dataset.json" --output "codegen_results.json"

# 3. Manually annotate results 
python annotate_completions.py --results "codegen_results.json" --output "codegen_annotations.csv"

# 4. Analyze correlation with human judgment
python analyze_correlations.py --annotations "codegen_annotations.csv" --output_prefix "codegen_analysis"
```

## Components Overview

This pipeline consists of four main components that should be run in sequence:

### 1. Dataset Creation (`code-splitter2.py`)

Creates a dataset by intelligently splitting code files into prefix (context), middle (target to predict), and suffix sections.

**Usage:**
```bash
python code-splitter2.py --directory /path/to/your/code --output code_completion_dataset.json --num_samples 30 --extensions .py .java
```

**Parameters:**
- `--directory`: Path to your code repository
- `--output`: Where to save the dataset
- `--num_samples` (optional): Number of samples to generate (default: 30)
- `--extensions` (optional): File extensions to include (default: .py)

### 2. Model Evaluation (`huggingface_evaluation.py`)

Evaluates HuggingFace models on the dataset and computes automatic metrics.

**Usage:**
```bash
python huggingface_evaluation.py --model_id "Salesforce/codegen-350M-mono" --dataset "code_completion_dataset.json" --output "codegen_results.json" --device cuda
```

**Parameters:**
- `--model_id`: HuggingFace model ID
- `--dataset`: Path to the dataset JSON file
- `--output`: Where to save evaluation results
- `--device` (optional): Device to use (default: cuda if available, otherwise cpu)
- `--max_new_tokens` (optional): Maximum tokens to generate (default: 100)
- `--half_precision` (optional): Use half precision for faster inference

**Recommended Models:**
- `Salesforce/codegen-350M-mono`: 350M parameter code model (4GB+ GPU memory)
- `bigcode/tiny_starcoder_py`: Small Python model (2GB+ GPU memory)

**For hardware debugging:**
```bash
python model_debugger.py --model_id "bigcode/tiny_starcoder_py"
```

### 3. Manual Annotation (`annotate_completions.py`)

Provides an interface to manually review and score completions, or you can use the auto-annotation script.

**Manual Annotation:**
```bash
python annotate_completions.py --results "codegen_results.json" --output "codegen_annotations.csv"
```

**Parameters:**
- `--results`: Path to evaluation results JSON
- `--output`: Where to save the annotations
- `--start` (optional): Sample index to start annotation from

**Auto Annotation Using Claude API:**
```bash
python auto_annotate.py --results "codegen_results.json" --output "codegen_annotations.csv" --api_key "your_claude_api_key"
```

**Parameters:**
- `--results`: Path to evaluation results JSON
- `--output`: Where to save the annotations
- `--api_key`: Claude API key (or set the CLAUDE_API_KEY environment variable)

### 4. Correlation Analysis (`analyze_correlations.py`)

Analyzes which automatic metrics best correlate with human judgment.

**Usage:**
```bash
python analyze_correlations.py --annotations "codegen_annotations.csv" --output_prefix "codegen_analysis"
```

**Parameters:**
- `--annotations`: Path to the annotations CSV file
- `--output_prefix`: Prefix for output files

**Outputs:**
- `{output_prefix}_results.json`: Correlation results
- `{output_prefix}_bar_chart.png`: Bar chart visualization
- `{output_prefix}_scatter_plots.png`: Scatter plots
- `human_score_distribution.png`: Distribution of human scores

## Example Workflow

To evaluate multiple models and compare them:

```bash
# Create the dataset
python code-splitter2.py --directory /path/to/your/code --output code_completion_dataset.json

# Evaluate CodeGen model
python huggingface_evaluation.py --model_id "Salesforce/codegen-350M-mono" --dataset "code_completion_dataset.json" --output "codegen_results.json"

# Evaluate TinyStarCoder
python huggingface_evaluation.py --model_id "bigcode/tiny_starcoder_py" --dataset "code_completion_dataset.json" --output "starcoder_results.json"

# Auto-annotate results using Claude API
python auto_annotate.py --results "codegen_results.json" --output "codegen_annotations.csv" --api_key "your_claude_api_key"
python auto_annotate.py --results "starcoder_results.json" --output "starcoder_annotations.csv" --api_key "your_claude_api_key"

# Analyze correlations
python analyze_correlations.py --annotations "codegen_annotations.csv" --output_prefix "codegen_analysis"
python analyze_correlations.py --annotations "starcoder_annotations.csv" --output_prefix "starcoder_analysis"
```

## Claude API Integration

To use Claude for auto-annotation:

1. Get an API key from Anthropic
2. Either:
   - Pass it directly: `--api_key "your_claude_api_key"`
   - Set as environment variable: `export CLAUDE_API_KEY="your_claude_api_key"`

## Understanding Evaluation Metrics

The evaluation uses four key metrics:

1. **Exact Match**: Binary measure of whether prediction exactly matches ground truth
2. **Line Overlap**: Percentage of lines from ground truth that appear in completion
3. **Token Match**: Percentage of tokens from ground truth that appear in completion
4. **Diff Score**: Similarity score based on sequence matching algorithm

Our research found that **Diff Score** has the strongest correlation with human judgment across multiple models.

## Troubleshooting

### Model Loading Issues
- For "model not found" errors, verify the model ID on HuggingFace
- Some models require login: `huggingface-cli login`
- For memory errors, try `--half_precision` or use a smaller model

### CUDA Out of Memory
- Reduce `--max_new_tokens`
- Use `--half_precision`
- Try a smaller model like `tiny_starcoder_py`

### API Rate Limits
- Claude API has rate limits - if you encounter them, add delays between calls or reduce batch size

## Contributing

Contributions to improve the evaluation pipeline are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.