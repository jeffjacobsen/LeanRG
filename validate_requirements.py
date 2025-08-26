#!/usr/bin/env python3
"""
Validate that all required dependencies are installed and importable.
Run this script to check if your environment has all needed packages.
"""

import sys
import importlib

# Core dependencies that must be importable
REQUIRED_IMPORTS = [
    # Scientific computing
    'numpy',
    'pandas', 
    'scipy',
    'sklearn',  # scikit-learn
    'matplotlib',
    'seaborn',
    'networkx',
    'umap',  # umap-learn
    
    # AI/ML
    'openai',
    'tiktoken',
    'jieba',
    'fastembed',
    
    # Database
    'pymysql',
    'supabase',
    'qdrant_client',  # qdrant-client
    
    # Utilities
    'requests',
    'aiohttp',
    'tqdm',
    'yaml',  # PyYAML
    'streamlit',
]

# Optional imports (not required for core functionality)
OPTIONAL_IMPORTS = [
    'chromadb',  # Legacy, may not be needed
]

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("🔍 Validating LeanRAG Dependencies...")
    print("=" * 50)
    
    all_good = True
    missing_required = []
    missing_optional = []
    
    # Check required imports
    print("\n📦 Required Dependencies:")
    for module in REQUIRED_IMPORTS:
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} - {error}")
            missing_required.append(module)
            all_good = False
    
    # Check optional imports
    print("\n📦 Optional Dependencies:")
    for module in OPTIONAL_IMPORTS:
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ⚠️  {module} - {error} (optional)")
            missing_optional.append(module)
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All required dependencies are installed!")
        print("✅ Your environment is ready for LeanRAG")
        
        if missing_optional:
            print(f"\n⚠️  {len(missing_optional)} optional packages missing:")
            for pkg in missing_optional:
                print(f"     - {pkg}")
    else:
        print("❌ Missing required dependencies!")
        print(f"\n🔧 To install missing packages:")
        print("   pip install -r requirements.txt")
        print(f"\n📋 Missing required packages:")
        for pkg in missing_required:
            print(f"     - {pkg}")
        
        sys.exit(1)
    
    print("\n🧪 Run 'python tests/quick_test.py' to test functionality")

if __name__ == '__main__':
    main()