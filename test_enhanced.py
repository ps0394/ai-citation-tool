#!/usr/bin/env python3
"""
Quick test of enhanced benchmark functionality
"""

import sys
import json
from datetime import datetime

# Test the new functions we added
def test_enhanced_functions():
    print("Testing enhanced benchmark functions...")
    
    # Add the current directory to path so we can import
    sys.path.append('.')
    
    try:
        from run_benchmark import (
            categorize_citations, 
            extract_title_from_url, 
            create_enhanced_results_structure
        )
        
        print("✅ Successfully imported enhanced functions")
        
        # Test citation categorization
        test_citations = [
            "https://learn.microsoft.com/azure/functions/",
            "https://docs.microsoft.com/azure/templates/",
            "https://stackoverflow.com/questions/12345",
            "https://github.com/Azure/azure-quickstart-templates"
        ]
        
        categorized = categorize_citations(test_citations)
        print(f"✅ Citation categorization works:")
        print(f"   Learn citations: {len(categorized['learn'])}")
        print(f"   Other citations: {len(categorized['other'])}")
        
        # Test URL title extraction
        title = extract_title_from_url("https://learn.microsoft.com/azure/azure-functions/")
        print(f"✅ Title extraction: {title}")
        
        # Test enhanced results structure
        prompts = [{"prompt_id": "001", "prompt": "Test", "category": "test"}]
        structure = create_enhanced_results_structure(prompts, "2026-02-06")
        print(f"✅ Enhanced structure created:")
        print(f"   Version: {structure['metadata']['benchmark_version']}")
        print(f"   Prompts: {structure['metadata']['total_prompts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_functions()
    if success:
        print("\n🎉 All enhanced functions working correctly!")
    else:
        print("\n💥 Some issues found - check the code")