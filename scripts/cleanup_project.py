import os
import re
import tokenize
from io import BytesIO

def remove_emojis(text):
    # Regex for emojis (simplified range for common emojis)
    # Using a safer, broader range approach to avoid "bad character range" errors
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2700-\u27BF"          # dingbats
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_python_comments(source):
    io_obj = BytesIO(source.encode('utf-8'))
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    try:
        tokens = tokenize.tokenize(io_obj.readline)
    except tokenize.TokenError:
        return source # Return original if tokenization fails

    for tok in tokens:
        token_type = tok.type
        token_string = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end
        
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        
        # Skip comments
        if token_type == tokenize.COMMENT:
            # Preserve shebang
            if start_line == 1 and token_string.startswith("#!"):
                out += token_string
            pass
        # Skip docstrings (optional, but user asked for comments, docstrings are often documentation)
        # Let's keep docstrings for now as they are functional documentation, but remove inline comments.
        # If user wants docstrings removed, we can add that logic.
        # Actually, "comments from the code files" usually implies # comments.
        else:
            out += token_string
            
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
        
    # Post-processing to remove empty lines created by full-line comments
    lines = out.split('\n')
    cleaned_lines = [line for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)

def remove_shell_comments(text):
    lines = text.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        # Preserve shebang
        if i == 0 and line.startswith("#!"):
            cleaned_lines.append(line)
            continue
            
        # Remove full line comments
        if line.strip().startswith("#"):
            continue
            
        # Remove inline comments (naive approach, might break strings with #)
        # For safety, let's just remove full line comments for shell scripts to avoid breaking commands
        # or use a safer regex if needed.
        # A safe bet for "comments" is usually the explanatory text blocks.
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def process_file(file_path):
    ext = os.path.splitext(file_path)[1]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Remove Emojis
        content = remove_emojis(content)
        
        # 2. Remove Comments based on type
        if ext == '.py':
            # For Python, use tokenizer for robustness
            content = remove_python_comments(content)
        elif ext in ['.sh', '.yaml', '.yml']:
            content = remove_shell_comments(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Walk through directory
    root_dir = "."
    skip_dirs = ['.git', 'venv', '__pycache__', 'data', 'experiments/results']
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(('.py', '.sh', '.md', '.yaml', '.json', '.txt')):
                file_path = os.path.join(root, file)
                # Skip the cleanup script itself
                if 'cleanup_project.py' in file_path:
                    continue
                process_file(file_path)

if __name__ == "__main__":
    main()
