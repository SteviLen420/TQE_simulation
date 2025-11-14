#!/usr/bin/env python3
"""
Script to extract modules from the monolithic TQE pipeline file.
This script parses the original file and creates modular components.
"""

import re
import os

ORIGINAL_FILE = "../TQE_Universe_Simulation_Full_Pipeline/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_file_sections(filename):
    """Read the original file and identify sections."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    return lines, content

def extract_imports(lines):
    """Extract all import statements."""
    imports = []
    in_import = False
    current_import = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            if '#' in stripped:
                # Handle inline comments
                comment_idx = stripped.index('#')
                import_part = stripped[:comment_idx].rstrip()
                comment_part = stripped[comment_idx:]
                imports.append(import_part)
                if comment_part.strip():
                    imports.append(comment_part)
            else:
                imports.append(line)
        elif line.strip() and in_import and (line.startswith(' ') or line.startswith('\t')):
            # Continuation line
            current_import.append(line)
        else:
            in_import = False
            if current_import:
                imports.append('\n'.join(current_import))
                current_import = []
    
    return imports

def extract_function_or_class(lines, start_idx, name_pattern):
    """Extract a function or class definition."""
    # Find the definition
    def_line = None
    for i in range(start_idx, len(lines)):
        if re.match(rf'^(def|class)\s+{name_pattern}', lines[i]):
            def_line = i
            break
    
    if def_line is None:
        return None, start_idx
    
    # Find the end (next top-level definition or end of file)
    indent_level = len(lines[def_line]) - len(lines[def_line].lstrip())
    end_idx = def_line + 1
    
    # Count indentation to find where function/class ends
    while end_idx < len(lines):
        line = lines[end_idx]
        if line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and (line.strip().startswith('def ') or line.strip().startswith('class ')):
                break
        end_idx += 1
    
    return lines[def_line:end_idx], end_idx

def main():
    """Main extraction function."""
    print("Reading original file...")
    lines, content = read_file_sections(ORIGINAL_FILE)
    print(f"File has {len(lines)} lines")
    
    # Extract imports (lines 263-357 approximately)
    print("Extracting imports...")
    imports = []
    for i in range(263, min(358, len(lines))):
        line = lines[i]
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            imports.append(line)
        elif line.strip() and not line.strip().startswith('#'):
            # Check if it's a setup/configuration line we need
            if 'IN_COLAB' in line or 'HEALPY_AVAILABLE' in line or 'CAMB_AVAILABLE' in line:
                imports.append(line)
    
    # Write common imports file
    common_imports = '\n'.join(imports)
    print(f"Extracted {len(imports)} import lines")
    
    print("\n✅ Module extraction script ready!")
    print("Note: Due to file size, manual extraction of specific sections is recommended.")
    print("The structure is set up. Next steps:")
    print("1. Extract core classes (PipelineContext, PhysicsEngine)")
    print("2. Extract utility functions")
    print("3. Extract phase functions")
    print("4. Extract analysis functions")

if __name__ == '__main__':
    main()

