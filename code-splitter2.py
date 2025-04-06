import os
import random
import json
from pathlib import Path


def get_file_paths(directory, extensions=None):
    """
    Get all file paths in the given directory with specified extensions.
    
    Args:
        directory (str): Path to the directory
        extensions (list): List of file extensions to include (e.g., ['.py', '.java'])
        
    Returns:
        list: List of file paths
    """
    if extensions is None:
        extensions = ['.py']  # Default to Python files
        
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_paths.append(os.path.join(root, file))
                
    return file_paths


# def split_code(code, min_context_length=5, max_prefix_length=2000, max_middle_length=10):
#     """
#     Split code into prefix, middle, and suffix parts.
    
#     Args:
#         code (str): The code to split
#         min_context_length (int): Minimum length for prefix and suffix in lines
#         max_prefix_length (int): Maximum length for prefix in characters
#         max_middle_length (int): Maximum length for the middle part in lines
        
#     Returns:
#         tuple: (prefix, middle, suffix)
#     """
#     lines = code.split('\n')
    
#     # Skip files that are too short
#     if len(lines) < min_context_length + 10:  # Ensure there's enough for meaningful prefix/suffix
#         return None, None, None
    
#     # Choose a random line to be the middle, but not too close to the beginning
#     middle_start = random.randint(min_context_length, len(lines) - min_context_length)
    
#     # Try to find good split points at function/class boundaries
#     # Look for lines that start with 'def ', 'class ', or have pattern like '    def '
#     candidate_starts = []
#     for i in range(min_context_length, len(lines) - min_context_length):
#         line = lines[i].strip()
#         if (line.startswith('def ') or line.startswith('class ') or 
#             (i > 0 and lines[i-1].strip() == '' and line)):
#             candidate_starts.append(i)
    
#     # If we found good candidates, use one of them
#     if candidate_starts:
#         middle_start = random.choice(candidate_starts)
    
#     # Determine how many lines to include in the middle
#     # Try to include complete function/method or block
#     middle_length = 0
#     indent_level = None
#     for i in range(middle_start, min(len(lines), middle_start + max_middle_length)):
#         line = lines[i]
        
#         # Skip empty lines at the beginning
#         if middle_length == 0 and not line.strip():
#             middle_start += 1
#             continue
            
#         # Determine indent level of the first non-empty line
#         if middle_length == 0 and line.strip():
#             indent_level = len(line) - len(line.lstrip())
        
#         # Include this line
#         middle_length += 1
        
#         # If we're back to the original indent level after having content, it's a good place to stop
#         if (middle_length > 1 and line.strip() and 
#             indent_level is not None and len(line) - len(line.lstrip()) <= indent_level):
#             break
    
#     # If we didn't add any lines (e.g., middle_start was an empty line), default to a simpler approach
#     if middle_length == 0:
#         middle_length = min(
#             random.randint(1, 5),   # Random length between 1 and 5 lines
#             max_middle_length,      # Cap at max_middle_length
#             len(lines) - middle_start - min_context_length  # Ensure we leave enough for suffix
#         )
    
#     # Get the initial split
#     prefix_lines = lines[:middle_start]
#     middle_lines = lines[middle_start:middle_start + middle_length]
#     suffix_lines = lines[middle_start + middle_length:]
    
#     # Check if prefix is too long and truncate if needed
#     prefix = '\n'.join(prefix_lines)
#     if len(prefix) > max_prefix_length:
#         # Use the last portion of the prefix to stay within limits
#         # and keep the most relevant context
#         prefix_char_limit = max_prefix_length
#         prefix = prefix[-prefix_char_limit:]
#         # Make sure we don't cut in the middle of a line
#         if not prefix.startswith('\n'):
#             prefix = prefix[prefix.find('\n')+1:]
    
#     middle = '\n'.join(middle_lines)
#     if not any(line.strip() and not line.strip().startswith('#') for line in middle_lines):
#         return None, None, None  # Skip samples with empty middle or only comments
#     if len(middle.strip()) < 10:
#         return None, None, None # At least 10 non-whitespace characters in the middle
#     suffix = '\n'.join(suffix_lines[:min(50, len(suffix_lines))])  # Limit suffix length too
    
#     return prefix, middle, suffix

def split_code(code, min_context_length=5, max_prefix_length=2000, max_middle_length=10):
    lines = code.split('\n')
    
    # Skip files that are too short
    if len(lines) < min_context_length + 10:
        return None, None, None
    
    # 1. Skip header-only prefixes by identifying first real code
    first_code_line = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped and 
            not stripped.startswith('#') and 
            not stripped.startswith('"""') and 
            not stripped.startswith("'''") and
            'import ' not in stripped):
            first_code_line = max(0, i - 2)  # Keep a bit of context
            break
    
    # Ensure we're not starting too late
    first_code_line = min(first_code_line, len(lines) // 5)
    
    # 2. Look for good split points (function/method definitions or complete statements)
    candidate_starts = []
    for i in range(first_code_line + min_context_length, len(lines) - min_context_length):
        line = lines[i].strip()
        prev_line = lines[i-1].strip() if i > 0 else ""
        
        # Good candidates: function/method defs, class defs, or statement after blank line
        if ((line.startswith('def ') or line.startswith('class ')) or
            (prev_line == "" and line and not line.startswith('#'))):
            candidate_starts.append(i)
    
    # If no good candidates, use weighted random (prefer real code)
    if not candidate_starts:
        weighted_indices = []
        for i in range(first_code_line + min_context_length, len(lines) - min_context_length):
            line = lines[i].strip()
            if line and not line.startswith('#') and not line.startswith(('"""', "'''")): 
                weighted_indices.extend([i] * 3)  # Weight actual code higher
            else:
                weighted_indices.append(i)
        
        if weighted_indices:
            middle_start = random.choice(weighted_indices)
        else:
            middle_start = random.randint(first_code_line + min_context_length, 
                                         len(lines) - min_context_length)
    else:
        middle_start = random.choice(candidate_starts)
    
    # 3. Determine appropriate middle length (try to get complete logical units)
    middle_length = 0
    indent_level = None
    for i in range(middle_start, min(len(lines), middle_start + max_middle_length * 2)):
        line = lines[i]
        
        # Skip empty lines at the beginning
        if middle_length == 0 and not line.strip():
            middle_start += 1
            continue
        
        # Determine indent level of the first non-empty line
        if middle_length == 0 and line.strip():
            indent_level = len(line) - len(line.lstrip())
        
        # Include this line
        middle_length += 1
        
        # If we have sufficient content and find a good stopping point, stop
        if middle_length >= 3:  # Ensure we have at least 3 lines
            # Stop at end of block or at blank line followed by differently indented code
            if (not line.strip() and i+1 < len(lines) and 
                lines[i+1].strip() and len(lines[i+1]) - len(lines[i+1].lstrip()) != indent_level):
                break
    
    # 4. Ensure minimum content in middle (not just comments)
    middle_lines = lines[middle_start:middle_start + middle_length]
    code_lines = [l for l in middle_lines if l.strip() and not l.strip().startswith('#')]
    if not code_lines or len(''.join(code_lines)) < 20:
        return None, None, None  # Skip if middle doesn't have enough actual code
    
    # Get the final split
    prefix_lines = lines[:middle_start]
    suffix_lines = lines[middle_start + middle_length:]
    
    # Check if prefix is too long and truncate if needed
    prefix = '\n'.join(prefix_lines)
    if len(prefix) > max_prefix_length:
        # Keep the most relevant context
        prefix = prefix[-max_prefix_length:]
        # Ensure we don't cut in the middle of a line
        if not prefix.startswith('\n'):
            prefix = prefix[prefix.find('\n')+1:]
    
    middle = '\n'.join(middle_lines)
    suffix = '\n'.join(suffix_lines[:min(50, len(suffix_lines))])
    
    return prefix, middle, suffix


def create_dataset(directory, output_file, num_samples=30, extensions=None):
    """
    Create a code completion dataset from files in a directory.
    
    Args:
        directory (str): Path to the directory containing code files
        output_file (str): Path to save the dataset
        num_samples (int): Number of samples to generate
        extensions (list): List of file extensions to include
        
    Returns:
        list: The generated dataset
    """
    file_paths = get_file_paths(directory, extensions)
    
    if not file_paths:
        raise ValueError(f"No files with extensions {extensions} found in {directory}")
    
    dataset = []
    
    # Keep generating samples until we have enough
    while len(dataset) < num_samples:
        # Choose a random file
        file_path = random.choice(file_paths)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # Skip empty files
            if not code.strip():
                continue
                
            prefix, middle, suffix = split_code(code)
            
            # Skip if the split failed
            if prefix is None:
                continue
                
            # Add to dataset
            sample = {
                'file_path': file_path,
                'prefix': prefix,
                'middle': middle,
                'suffix': suffix
            }
            
            dataset.append(sample)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    # Replace with the path to your code repository
    repo_path = "/home/aki/Desktop/Competitive_Programming/leetcode"
    
    # Replace with your preferred file extensions
    extensions = ['.py', '.java', '.js']
    
    # Create the dataset
    dataset = create_dataset(
        directory=repo_path,
        output_file="code_completion_dataset2.json",
        num_samples=30,
        extensions=extensions
    )


''''
python code-splitter2.py
'''