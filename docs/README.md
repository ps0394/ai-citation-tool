# AI Citation Benchmark - GitHub Pages Setup

## Quick Setup

1. **Fork/Clone this repository to GitHub**

2. **Set up Secrets** (in GitHub repo settings > Secrets and variables > Actions):
   ```
   OPENAI_API_KEY=your-openai-key
   GEMINI_API_KEY=your-gemini-key  
   PERPLEXITY_API_KEY=your-perplexity-key
   ```

3. **Enable GitHub Pages** (in repo settings > Pages):
   - Source: Deploy from branch
   - Branch: main
   - Folder: /docs

4. **Enable Actions** (in repo settings > Actions):
   - Allow all actions and reusable workflows

## What You Get

- **📊 Live Dashboard**: View citation analytics at `https://yourusername.github.io/ai-citation-tool`
- **🤖 Automated Runs**: Daily benchmark at 9 AM UTC
- **📈 Historical Data**: CSV results stored in `docs/data/`
- **🔄 Manual Triggers**: Run benchmarks on-demand via Actions tab

## Dashboard Features

- **Citation Rate Metrics** - % of responses citing learn.microsoft.com
- **Provider Comparison** - ChatGPT vs Gemini vs Perplexity 
- **Domain Analysis** - Top cited domains across all responses
- **Results Table** - Recent benchmark results
- **Manual Benchmark** - Trigger new runs (requires backend setup)

## GitHub Actions Workflow

The workflow:
1. Runs `python run_benchmark.py` on schedule or manual trigger
2. Saves results to `docs/data/results_YYYY-MM-DD.csv`
3. Commits and pushes results to trigger GitHub Pages deployment
4. Dashboard automatically loads latest data

## Local Development

```bash
# Run locally
python run_benchmark.py --dry-run

# Serve GitHub Pages locally
cd docs
python -m http.server 8000
# Visit http://localhost:8000
```

## Customization

Edit files:
- `prompts.csv` - Change benchmark questions
- `docs/index.html` - Modify dashboard design
- `.github/workflows/benchmark.yml` - Adjust schedule/configuration

Production-ready for Microsoft Learn citation tracking!