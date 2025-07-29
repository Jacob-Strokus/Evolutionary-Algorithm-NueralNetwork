#!/usr/bin/env python3
"""
Fix import paths for test files moved to tests/ folder
"""

import os
import glob

def fix_test_imports():
    """Fix all test files to use correct import path"""
    test_files = glob.glob('tests/*.py')
    
    fixed_count = 0
    
    for file_path in test_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for the incorrect path pattern
            old_pattern = 'sys.path.append(os.path.dirname(os.path.abspath(__file__)))'
            new_pattern = 'sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'
            
            if old_pattern in content:
                # Replace the path
                new_content = content.replace(old_pattern, new_pattern)
                
                # Write back to file
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                print(f"✅ Fixed: {file_path}")
                fixed_count += 1
            else:
                print(f"⚪ Skipped: {file_path} (already correct or no sys.path)")
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"\n🏁 Fixed {fixed_count} test files")

if __name__ == "__main__":
    fix_test_imports()
