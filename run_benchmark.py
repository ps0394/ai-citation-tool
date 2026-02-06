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

    client = OpenAI(api_key=api_key)

    # Use current chat completions API
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please provide accurate information and include citations with source URLs when possible. Format URLs clearly in your response."},
        {"role": "user", "content": f"{prompt} Please include relevant source URLs and documentation links in your response."}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )

    # Get response text
    text = resp.choices[0].message.content or ""

    # Extract URLs from the response text (since current API doesn't return structured citations)
    citations = extract_urls_from_text(text)

    return text, citations

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
    if not GEMINI_API_KEY or not requests:
        return "API key not configured or requests not available", []
    
    # Note: This is a simplified implementation
    # Actual Gemini API with grounding may require different endpoints
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": f"Please answer this question with citations: {prompt}"
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 1000
        }
    }
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            # Extract URLs from the response text
            citations = extract_urls_from_text(response_text)
            return response_text, citations
        else:
            return f"API Error: {response.status_code}", []
            
    except Exception as e:
        return f"Error: {str(e)}", []

def call_perplexity_api(prompt: str) -> Tuple[str, List[str]]:
    """Call Perplexity API"""
    if not PERPLEXITY_API_KEY or not requests:
        return "API key not configured or requests not available", []
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "return_citations": True
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            # Extract citations from Perplexity response
            citations = []
            if 'citations' in result:
                citations = result['citations']
            else:
                # Fall back to extracting URLs from text
                citations = extract_urls_from_text(response_text)
            
            return response_text, citations
        else:
            return f"API Error: {response.status_code}", []
            
    except Exception as e:
        return f"Error: {str(e)}", []

def write_results_to_csv(results: List[Dict], output_file: str):
    """Write results to CSV file"""
    fieldnames = ['date', 'provider', 'model', 'prompt_id', 'prompt', 'response', 'citation_url', 'citation_domain']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

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
    
    results = []
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
        
        print(f"\nProcessing prompt {prompt_id}: {prompt_text[:60]}...")
        
        for provider_name, model, api_function in providers:
            current_call += 1
            print(f"  [{current_call}/{total_calls}] Calling {provider_name}...")
            
            try:
                response_text, citations = api_function(prompt_text)
                
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
                            'response': response_text,
                            'citation_url': citation,
                            'citation_domain': domain
                        })
                else:
                    # No citations
                    results.append({
                        'date': timestamp,
                        'provider': provider_name,
                        'model': model,
                        'prompt_id': prompt_id,
                        'prompt': prompt_text,
                        'response': response_text,
                        'citation_url': 'NONE',
                        'citation_domain': 'NONE'
                    })
                
                # Rate limiting delay
                if current_call < total_calls and not dry_run:
                    time.sleep(DELAY_BETWEEN_CALLS)
                    
            except Exception as e:
                print(f"    Error calling {provider_name}: {str(e)}")
                # Record the error
                results.append({
                    'date': timestamp,
                    'provider': provider_name,
                    'model': model,
                    'prompt_id': prompt_id,
                    'prompt': prompt_text,
                    'response': f"ERROR: {str(e)}",
                    'citation_url': 'ERROR',
                    'citation_domain': 'ERROR'
                })
    
    # Write results
    write_results_to_csv(results, output_file)
    print(f"\nBenchmark complete! Results written to: {output_file}")
    print(f"Total results: {len(results)} rows")
    
    # Quick summary
    learn_citations = sum(1 for r in results if r['citation_domain'] == 'learn.microsoft.com')
    total_citations = sum(1 for r in results if r['citation_url'] not in ['NONE', 'ERROR'])
    
    if total_citations > 0:
        learn_percentage = (learn_citations / total_citations) * 100
        print(f"Learn.microsoft.com citations: {learn_citations}/{total_citations} ({learn_percentage:.1f}%)")
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