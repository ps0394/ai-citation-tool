#!/usr/bin/env python3
"""
AI Citation Benchmark Tool
Queries ChatGPT, Gemini, and Perplexity with Microsoft/Azure questions
and tracks citations to learn.microsoft.com
"""

import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configuration
PROMPTS_FILE = "prompts.csv"
DELAY_BETWEEN_CALLS = 1  # seconds to avoid rate limiting

# API Configuration - Set these as environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def load_prompts(file_path: str) -> List[Dict[str, str]]:
    """Load prompts from CSV file"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts

def normalize_domain(url: str) -> str:
    """Normalize learn.microsoft.com citations"""
    if not url:
        return ""
    
    # Parse the URL to get the domain
    try:
        domain = urlparse(url).netloc.lower()
        # Normalize docs.microsoft.com to learn.microsoft.com
        if domain == "docs.microsoft.com":
            domain = "learn.microsoft.com"
        return domain
    except:
        return ""

def categorize_citations(citations: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize citations into Learn vs Other with metadata"""
    learn_citations = []
    other_citations = []
    
    for url in citations:
        if not url or url in ['NONE', 'ERROR']:
            continue
            
        domain = normalize_domain(url)
        citation_data = {
            "url": url,
            "domain": domain,
            "title": extract_title_from_url(url)
        }
        
        if domain == "learn.microsoft.com":
            learn_citations.append(citation_data)
        else:
            other_citations.append(citation_data)
    
    return {
        "learn": learn_citations,
        "other": other_citations
    }

def extract_title_from_url(url: str) -> str:
    """Extract a readable title from URL (simple heuristic)"""
    try:
        # Remove protocol and extract the path
        path = urlparse(url).path
        # Get the last part of the path and clean it up
        title = path.split('/')[-1] or path.split('/')[-2]
        # Replace hyphens with spaces and capitalize
        title = title.replace('-', ' ').replace('_', ' ').title()
        return title if title else "Documentation Link"
    except:
        return "Documentation Link"

def call_stub_chatgpt(prompt: str) -> Tuple[str, List[str]]:
    """Stub implementation for ChatGPT"""
    citations = [
        "https://learn.microsoft.com/azure/azure-functions/",
        "https://docs.microsoft.com/azure/templates/"
    ] if "function" in prompt.lower() or "bicep" in prompt.lower() else [
        "https://learn.microsoft.com/azure/",
        "https://stackoverflow.com/questions/12345"
    ]
    
    response = f"Here's how to {prompt.split('?')[0].lower().replace('how do i ', '')}: [detailed explanation with citations]"
    return response, citations

def call_stub_gemini(prompt: str) -> Tuple[str, List[str]]:
    """Stub implementation for Gemini"""
    citations = [
        "https://learn.microsoft.com/en-us/azure/",
        "https://cloud.google.com/docs/"
    ] if "azure" in prompt.lower() else [
        "https://docs.microsoft.com/troubleshoot/",
        "https://learn.microsoft.com/training/"
    ]
    
    response = f"Based on official documentation: {prompt} can be addressed by following these steps..."
    return response, citations

def call_stub_perplexity(prompt: str) -> Tuple[str, List[str]]:
    """Stub implementation for Perplexity"""
    citations = [
        "https://learn.microsoft.com/azure/devops/",
        "https://techcommunity.microsoft.com/blog/",
        "https://github.com/Azure/azure-quickstart-templates"
    ] if "devops" in prompt.lower() or "pipeline" in prompt.lower() else [
        "https://docs.microsoft.com/azure/",
        "https://learn.microsoft.com/azure/well-architected/"
    ]
    
    response = f"According to multiple sources including Microsoft Learn: {prompt.replace('?', '')} involves several key considerations..."
    return response, citations

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def call_openai(prompt: str, model: str = "gpt-4o-mini", use_web_search: bool = True):
    """
    Returns: (response_text, citations_list)
    Uses the current OpenAI Chat API
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env)")

    print(f"[DEBUG] Making OpenAI API call with model: {model}")
    print(f"[DEBUG] API key present: {bool(api_key)} (ends with: {api_key[-6:] if api_key else 'None'})")

    client = OpenAI(api_key=api_key)

    # Use current chat completions API
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please provide accurate information and include citations with source URLs when possible. Format URLs clearly in your response."},
        {"role": "user", "content": f"{prompt} Please include relevant source URLs and documentation links in your response."}
    ]

    try:
        print(f"[DEBUG] Calling OpenAI API...")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        print(f"[DEBUG] OpenAI API call successful")

        # Get response text
        text = resp.choices[0].message.content or ""
        print(f"[DEBUG] Response length: {len(text)} characters")
        print(f"[DEBUG] Response preview: {text[:100]}...")

        # Extract URLs from the response text (since current API doesn't return structured citations)
        citations = extract_urls_from_text(text)
        print(f"[DEBUG] Extracted {len(citations)} citations: {citations}")

        return text, citations
        
    except Exception as e:
        print(f"[DEBUG] OpenAI API call failed: {e}")
        print(f"[DEBUG] Full error details: {repr(e)}")
        return f"Error calling OpenAI: {e}", []

def call_openai_api(prompt: str) -> Tuple[str, List[str]]:
    """Call OpenAI API (ChatGPT) with web search enabled"""
    if not OPENAI_API_KEY:
        return "API key not configured", []
    
    # Try new OpenAI client first, fall back to requests
    if OpenAI:
        try:
            return call_openai(prompt)
        except Exception as e:
            print(f"    New OpenAI client failed, falling back to requests: {e}")
    
    if not requests:
        return "Neither OpenAI client nor requests available", []
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use GPT-4 with web browsing capability
    data = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Please provide accurate information and include citations to authoritative sources when possible."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            # Extract URLs from the response text
            citations = extract_urls_from_text(response_text)
            return response_text, citations
        else:
            return f"API Error: {response.status_code}", []
            
    except Exception as e:
        return f"Error: {str(e)}", []

def call_gemini_api(prompt: str) -> Tuple[str, List[str]]:
    """Call Gemini API with grounding"""
    print(f"[DEBUG] Making Gemini API call with model: gemini-pro")
    print(f"[DEBUG] API key present: {bool(GEMINI_API_KEY)} (ends with: {GEMINI_API_KEY[-6:] if GEMINI_API_KEY else 'None'})")
    
    if not GEMINI_API_KEY or not requests:
        print(f"[DEBUG] Gemini API key or requests unavailable - API key: {bool(GEMINI_API_KEY)}, requests: {bool(requests)}")
        return "API key not configured or requests not available", []
    
    print(f"[DEBUG] Calling Gemini API...")
    
    # Note: This is a simplified implementation
    # Actual Gemini API with grounding may require different endpoints
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": f"Please answer this question with citations to official documentation: {prompt}"
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 1000
        }
    }
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        print(f"[DEBUG] Making request to: {url[:80]}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"[DEBUG] Gemini API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[DEBUG] Gemini API response structure: {list(result.keys())}")
            
            response_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            print(f"[DEBUG] Gemini API call successful")
            print(f"[DEBUG] Response length: {len(response_text)} characters")
            print(f"[DEBUG] Response preview: {response_text[:100]}...")
            
            # Extract URLs from the response text
            citations = extract_urls_from_text(response_text)
            print(f"[DEBUG] Extracted {len(citations)} citations: {citations[:3]}...")
            return response_text, citations
        else:
            error_text = response.text if hasattr(response, 'text') else 'No error details'
            print(f"[DEBUG] Gemini API error: {response.status_code} - {error_text}")
            return f"API Error: {response.status_code} - {error_text}", []
            
    except Exception as e:
        print(f"[DEBUG] Gemini API exception: {type(e).__name__}: {str(e)}")
        return f"Error: {str(e)}", []

def call_perplexity_api(prompt: str) -> Tuple[str, List[str]]:
    """Call Perplexity API"""
    print(f"[DEBUG] Making Perplexity API call with model: llama-3.1-sonar-small-128k-online")
    print(f"[DEBUG] API key present: {bool(PERPLEXITY_API_KEY)} (ends with: {PERPLEXITY_API_KEY[-6:] if PERPLEXITY_API_KEY else 'None'})")
    
    if not PERPLEXITY_API_KEY or not requests:
        print(f"[DEBUG] Perplexity API key or requests unavailable - API key: {bool(PERPLEXITY_API_KEY)}, requests: {bool(requests)}")
        return "API key not configured or requests not available", []
    
    print(f"[DEBUG] Calling Perplexity API...")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "user", "content": f"Please answer this question with citations to official documentation: {prompt}"}
        ],
        "max_tokens": 1000,
        "return_citations": True
    }
    
    try:
        print(f"[DEBUG] Making request to Perplexity API...")
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"[DEBUG] Perplexity API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[DEBUG] Perplexity API response structure: {list(result.keys())}")
            
            response_text = result['choices'][0]['message']['content']
            print(f"[DEBUG] Perplexity API call successful")
            print(f"[DEBUG] Response length: {len(response_text)} characters")
            print(f"[DEBUG] Response preview: {response_text[:100]}...")
            
            # Extract citations from Perplexity response
            citations = []
            if 'citations' in result:
                citations = result['citations']
                print(f"[DEBUG] Found {len(citations)} citations in response object")
            else:
                # Fall back to extracting URLs from text
                citations = extract_urls_from_text(response_text)
                print(f"[DEBUG] Extracted {len(citations)} citations from response text: {citations[:3]}...")
            
            return response_text, citations
        else:
            error_text = response.text if hasattr(response, 'text') else 'No error details'
            print(f"[DEBUG] Perplexity API error: {response.status_code} - {error_text}")
            return f"API Error: {response.status_code} - {error_text}", []
            
    except Exception as e:
        print(f"[DEBUG] Perplexity API exception: {type(e).__name__}: {str(e)}")
        return f"Error: {str(e)}", []

def write_results_to_csv(results: List[Dict], output_file: str):
    """Write results to CSV file (legacy format for existing dashboard)"""
    fieldnames = ['date', 'provider', 'model', 'prompt_id', 'prompt', 'response', 'citation_url', 'citation_domain']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def write_results_to_json(enhanced_results: Dict, output_file: str):
    """Write enhanced results to JSON file"""
    json_file = output_file.replace('.csv', '.json')
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced results written to: {json_file}")

def create_enhanced_results_structure(prompts: List[Dict], timestamp: str) -> Dict:
    """Create the enhanced JSON structure for detailed results"""
    return {
        "metadata": {
            "date": timestamp,
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "benchmark_version": "2.0"
        },
        "prompts": []
    }

def run_benchmark(dry_run=False):
    """Main benchmark runner"""
    print("AI Citation Benchmark Tool")
    print("=" * 30)
    
    if dry_run:
        print("Running in DRY RUN mode (no API calls)")
        print()
    
    # Load prompts
    if not os.path.exists(PROMPTS_FILE):
        print(f"Error: {PROMPTS_FILE} not found!")
        return
    
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = f"results_{timestamp}.csv"
    
    # Initialize both legacy and enhanced results
    results = []  # Legacy CSV format
    enhanced_results = create_enhanced_results_structure(prompts, timestamp)
    
    total_calls = len(prompts) * 3  # 3 providers
    current_call = 0
    
    # API providers and their functions
    if dry_run:
        providers = [
            ("ChatGPT", "gpt-4-turbo", call_stub_chatgpt),
            ("Gemini", "gemini-pro", call_stub_gemini),
            ("Perplexity", "llama-3.1-sonar-small-128k-online", call_stub_perplexity)
        ]
    else:
        providers = [
            ("ChatGPT", "gpt-4-turbo", call_openai_api),
            ("Gemini", "gemini-pro", call_gemini_api),
            ("Perplexity", "llama-3.1-sonar-small-128k-online", call_perplexity_api)
        ]
    
    for prompt_data in prompts:
        prompt_id = prompt_data['prompt_id']
        prompt_text = prompt_data['prompt']
        category = prompt_data.get('category', 'general')
        
        print(f"\nProcessing prompt {prompt_id}: {prompt_text[:60]}...")
        
        # Enhanced JSON structure for this prompt
        enhanced_prompt_data = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "category": category,
            "providers": {}
        }
        
        for provider_name, model, api_function in providers:
            current_call += 1
            print(f"  [{current_call}/{total_calls}] Calling {provider_name}...")
            
            try:
                response_text, citations = api_function(prompt_text)
                
                # Categorize citations for enhanced JSON
                categorized_citations = categorize_citations(citations)
                
                # Enhanced provider data
                enhanced_prompt_data["providers"][provider_name] = {
                    "model": model,
                    "response_text": response_text,
                    "learn_citations": categorized_citations["learn"],
                    "other_citations": categorized_citations["other"],
                    "citation_count": {
                        "learn": len(categorized_citations["learn"]),
                        "other": len(categorized_citations["other"]),
                        "total": len(categorized_citations["learn"]) + len(categorized_citations["other"])
                    }
                }
                
                # Legacy CSV format (for backward compatibility)
                if citations:
                    # One row per citation
                    for citation in citations:
                        domain = normalize_domain(citation)
                        results.append({
                            'date': timestamp,
                            'provider': provider_name,
                            'model': model,
                            'prompt_id': prompt_id,
                            'prompt': prompt_text,
                            'response': response_text[:500] + "..." if len(response_text) > 500 else response_text,  # Truncate for CSV
                            'citation_url': citation,
                            'citation_domain': domain
                        })
                else:
                    # No citations for legacy CSV
                    results.append({
                        'date': timestamp,
                        'provider': provider_name,
                        'model': model,
                        'prompt_id': prompt_id,
                        'prompt': prompt_text,
                        'response': response_text[:500] + "..." if len(response_text) > 500 else response_text,
                        'citation_url': 'NONE',
                        'citation_domain': 'NONE'
                    })
                
                # Rate limiting delay
                if current_call < total_calls and not dry_run:
                    time.sleep(DELAY_BETWEEN_CALLS)
                    
            except Exception as e:
                print(f"    Error calling {provider_name}: {str(e)}")
                
                # Record error in both formats
                error_response = f"ERROR: {str(e)}"
                
                # Enhanced format
                enhanced_prompt_data["providers"][provider_name] = {
                    "model": model,
                    "response_text": error_response,
                    "learn_citations": [],
                    "other_citations": [],
                    "citation_count": {"learn": 0, "other": 0, "total": 0},
                    "error": str(e)
                }
                
                # Legacy CSV format
                results.append({
                    'date': timestamp,
                    'provider': provider_name,
                    'model': model,
                    'prompt_id': prompt_id,
                    'prompt': prompt_text,
                    'response': error_response,
                    'citation_url': 'ERROR',
                    'citation_domain': 'ERROR'
                })
        
        # Add this prompt's data to enhanced results
        enhanced_results["prompts"].append(enhanced_prompt_data)
    
    # Write both formats
    write_results_to_csv(results, output_file)
    write_results_to_json(enhanced_results, output_file)
    
    print(f"\nBenchmark complete! Results written to: {output_file}")
    print(f"Total results: {len(results)} rows")
    
    # Enhanced summary from JSON data
    total_learn = sum(
        prompt_data["providers"][provider]["citation_count"]["learn"]
        for prompt_data in enhanced_results["prompts"]
        for provider in prompt_data["providers"]
    )
    total_other = sum(
        prompt_data["providers"][provider]["citation_count"]["other"] 
        for prompt_data in enhanced_results["prompts"]
        for provider in prompt_data["providers"]
    )
    total_citations = total_learn + total_other
    
    if total_citations > 0:
        learn_percentage = (total_learn / total_citations) * 100
        print(f"Learn.microsoft.com citations: {total_learn}/{total_citations} ({learn_percentage:.1f}%)")
        print(f"Other citations: {total_other}/{total_citations} ({100-learn_percentage:.1f}%)")
    else:
        print("No citations found in responses.")

def main():
    parser = argparse.ArgumentParser(description='AI Citation Benchmark Tool')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run with stub data (no API keys required)')
    args = parser.parse_args()
    
    run_benchmark(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
