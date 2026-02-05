# AI Citation Tool Setup

## Quick Start

1. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Set API Keys as Environment Variables**
   ```powershell
   $env:OPENAI_API_KEY = "your-openai-key-here"
   $env:GEMINI_API_KEY = "your-gemini-key-here"
   $env:PERPLEXITY_API_KEY = "your-perplexity-key-here"
   ```

3. **Run the Benchmark**
   ```powershell
   python run_benchmark.py
   ```

## API Key Setup

### OpenAI/ChatGPT
- Get your key from: https://platform.openai.com/api-keys
- For Microsoft employees: Use Azure OpenAI for faster access

### Gemini
- Get your key from: https://aistudio.google.com/app/apikey
- Enable "Grounding with Google Search" in your project

### Perplexity
- Get your key from: https://www.perplexity.ai/settings/api
- Use the online/search models for citations

## Output

Results are saved to `results/results_YYYY-MM-DD.csv` with this schema:
- `date` - Run date
- `provider` - ChatGPT, Gemini, or Perplexity  
- `model` - Specific model used
- `prompt_id` - Question identifier
- `prompt` - The question asked
- `response` - Full AI response
- `citation_url` - Cited URL (or NONE)
- `citation_domain` - Domain from URL (normalized)

Learn citations show as `learn.microsoft.com` domain.

## SharePoint Integration

To save results directly to SharePoint/OneDrive:
1. Map SharePoint as a network drive
2. Modify `RESULTS_DIR` in the script to point to your mapped drive
3. Example: `RESULTS_DIR = "Z:\AI-Citation-Benchmark"`

## Analysis in Excel

1. Open the CSV in Excel
2. Create a Pivot Table:
   - Rows: `provider`
   - Values: Count of `citation_domain` where equals "learn.microsoft.com"
3. Add calculated field for percentage of total citations

This gives you immediate insights into Learn citation performance across AI providers.