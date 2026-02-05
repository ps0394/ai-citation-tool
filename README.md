# AI Citation Tool

Benchmark tool that tests how **learn.microsoft.com** is cited by ChatGPT, Gemini, and Perplexity.

## Run Instructions

**Dry run (no API keys needed):**
```
python run_benchmark.py --dry-run
```

**Real run (requires API keys):**
```
set OPENAI_API_KEY=your-key
set GEMINI_API_KEY=your-key
set PERPLEXITY_API_KEY=your-key
python run_benchmark.py
```

## Output

Generates `results_YYYY-MM-DD.csv` with columns:
- `date` - Run date  
- `provider` - ChatGPT/Gemini/Perplexity
- `model` - Model name
- `prompt_id` - Question ID
- `prompt` - Question text
- `response` - AI response
- `citation_url` - Cited URL
- `citation_domain` - Domain (normalized)

Normalizes `docs.microsoft.com` → `learn.microsoft.com`
