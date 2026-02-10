# AI Citation Tool

<!-- Updated: 2026-02-10 12:00 UTC -->
Benchmark tool that tests how **learn.microsoft.com** is cited by ChatGPT, Gemini, and Perplexity.

## 🚀 Quick Start

**View live dashboard:** [GitHub Pages Dashboard](https://ps0394.github.io/ai-citation-tool)

**Run manually:**
```bash
# Dry run (no API keys needed)
python run_benchmark.py --dry-run

# Real run (requires API keys in .env file)
python run_benchmark.py
```

## 📋 Setup Instructions

### 1. **Clone Repository**
```bash
git clone https://github.com/ps0394/ai-citation-tool.git
cd ai-citation-tool
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Configure API Keys**

#### For Local Development:
Create a `.env` file:
```env
OPENAI_API_KEY=sk-proj-your-openai-key
GEMINI_API_KEY=your-gemini-key  
PERPLEXITY_API_KEY=your-perplexity-key
```

#### For GitHub Actions (Required):
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add repository secrets:
   - **Name:** `OPENAI_API_KEY` **Value:** Your OpenAI API key
   - **Name:** `GEMINI_API_KEY` **Value:** Your Gemini API key (optional)
   - **Name:** `PERPLEXITY_API_KEY` **Value:** Your Perplexity API key (optional)

⚠️ **Important:** Without GitHub Secrets, the automated benchmarks will run stub functions instead of real API calls.

### 4. **Enable GitHub Pages**
1. Go to **Settings** → **Pages**
2. **Source:** Deploy from a branch
3. **Branch:** main
4. **Folder:** /docs

## ⚡ Automation Features

### **Daily Benchmark Runs**
- Runs automatically at 9:00 AM UTC daily
- Results stored in repository as `results_YYYY-MM-DD.csv`
- Updates dashboard automatically

### **Manual Trigger**
- Go to **Actions** → **AI Citation Benchmark** → **Run workflow**
- Choose normal run or dry-run mode

### **Web Dashboard**
- Displays citation analytics and trends
- Interactive charts powered by Chart.js
- Automatically loads latest results
- Fallback to sample data if no results exist

## 📊 Output

Generates `results_YYYY-MM-DD.csv` with columns:
- `date` - Run date  
- `provider` - ChatGPT/Gemini/Perplexity
- `model` - Model name used
- `prompt_id` - Question ID (001-010)
- `prompt` - Question text
- `response` - AI response text
- `citation_url` - Cited URL (if found)
- `citation_domain` - Domain (learn.microsoft.com normalized)

**Domain Normalization:** `docs.microsoft.com` → `learn.microsoft.com`

## 🔧 Troubleshooting

### **No API Usage in OpenAI Dashboard**
**Symptoms:** Benchmark completes but no OpenAI API calls appear in dashboard  
**Cause:** GitHub Secrets not configured  
**Fix:** Add `OPENAI_API_KEY` to repository secrets (see Setup step 3)

### **"No citations found" Results**
**Symptoms:** CSV shows empty citation columns  
**Cause:** Using stub functions (API keys missing)  
**Fix:** Verify API keys are configured in GitHub Secrets

### **IndentationError in GitHub Actions**
**Symptoms:** Workflow fails with Python indentation errors  
**Cause:** Git merge conflicts in code  
**Fix:** Reset local branch: `git reset --hard origin/main`

### **Debug API Calls**
Run with debug logging to see API call details:
```bash
python run_benchmark.py  # Debug output included automatically
```

## 🛠️ Development

**File Structure:**
- `run_benchmark.py` - Main benchmark script
- `prompts.csv` - 10 test questions (Azure/Microsoft focused)
- `docs/index.html` - Interactive dashboard
- `.github/workflows/benchmark.yml` - GitHub Actions automation
- `results_*.csv` - Historical benchmark results

**Adding New Questions:**
Edit `prompts.csv` with format: `prompt_id,category,prompt`

**Modifying Providers:**
Update provider functions in `run_benchmark.py` (ChatGPT, Gemini, Perplexity)
