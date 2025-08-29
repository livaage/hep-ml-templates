#!/usr/bin/env python3
"""
Quick script to clean up common linting issues in the codebase.
"""

import os
import re
from pathlib import Path


def clean_whitespace_issues(file_path: Path) -> bool:
    """Remove trailing whitespace and clean up blank lines with whitespace."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove trailing whitespace from all lines
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove any whitespace-only blank lines (W293)
        cleaned_lines = ['' if not line.strip() else line for line in cleaned_lines]
        
        # Ensure file ends with a single newline
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        content = '\n'.join(cleaned_lines) + '\n'
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def remove_unused_imports(file_path: Path) -> bool:
    """Remove some obviously unused imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        
        # Track which imports are used
        imports_to_check = {
            'import pandas as pd': 'pd.',
            'import numpy as np': 'np.',
            'from typing import Union': 'Union',
            'from typing import List': 'List',
            'from typing import Tuple': 'Tuple',
            'from typing import Optional': 'Optional',
            'import os': 'os.',
        }
        
        # Simple check - if import is not used anywhere in the file, remove it
        lines_to_keep = []
        for line in lines:
            should_keep = True
            for import_line, usage_pattern in imports_to_check.items():
                if line.strip() == import_line:
                    # Check if this import is actually used
                    rest_of_file = '\n'.join(lines)
                    if usage_pattern not in rest_of_file.replace(line, ''):
                        should_keep = False
                        break
            if should_keep:
                lines_to_keep.append(line)
        
        content = '\n'.join(lines_to_keep)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error removing imports from {file_path}: {e}")
        return False


def main():
    """Clean up linting issues in the source directory."""
    src_dir = Path('src')
    files_cleaned = 0
    
    print("🧹 Cleaning up linting issues...")
    
    # Find all Python files
    for py_file in src_dir.rglob('*.py'):
        print(f"Processing {py_file}...")
        
        whitespace_cleaned = clean_whitespace_issues(py_file)
        imports_cleaned = remove_unused_imports(py_file)
        
        if whitespace_cleaned or imports_cleaned:
            files_cleaned += 1
            changes = []
            if whitespace_cleaned:
                changes.append("whitespace")
            if imports_cleaned:
                changes.append("imports")
            print(f"  ✅ Cleaned: {', '.join(changes)}")
        else:
            print(f"  ✓ No changes needed")
    
    print(f"\n🎉 Cleanup complete! {files_cleaned} files modified.")


if __name__ == '__main__':
    main()
