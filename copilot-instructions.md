
Fast, Self-Built Learn Citation Benchmark
Audience: Microsoft employee (individual contributor)
Goal: Quickly build a repeatable benchmark that shows how learn.microsoft.com is cited by major AI answer engines (ChatGPT, Gemini, Perplexity).
Design principle: Bias toward speed and ease, not durability or production readiness.
Name of this tool: AI Citation Tool


What You Will End Up With

A fixed prompt set focused on Learn‑eligible questions
Real responses from ChatGPT, Gemini, and Perplexity (via APIs)
Extracted citations and domains
A daily CSV stored in SharePoint / OneDrive
Simple analysis in Excel or Power BI
This is intentionally lightweight and solo‑buildable.


Explicit Non‑Goals (to stay fast)

No Google AI Overviews automation (UI scraping is fragile and slow)
No Copilot Studio required to run the benchmark
No retries, durability, or fault tolerance
No claims about training data


Step 1 — Create the Prompt Set
Create a file called prompts.csv.
Schema:
prompt_id,category,prompt


Example:
001,how-to,How do I deploy an Azure Function using Bicep?
002,concept,What is Azure Virtual Network peering?
003,comparison,Azure Functions vs Logic Apps — when should I use each?
004,troubleshooting,Why does my Azure DevOps pipeline fail with error MSB1009?
005,certification,What skills are covered in the AZ-104 certification?


Guidelines:

Bias toward Azure, .NET, DevOps, certifications
Avoid trivia or generic AI questions
50 prompts is sufficient for v1 (100 optional)
This file should remain stable over time


Step 2 — Get API Keys (One Time)
You need three API keys.
ChatGPT

Use Azure OpenAI (fastest inside Microsoft)
Enable the Responses API
Allow web search / grounding so citations are returned
Gemini

Create a Gemini API key
Use Gemini with Google Search grounding enabled
This returns grounding metadata and citation URLs
Perplexity

Create a Perplexity API key
Use a search/online model (citations included)
Optionally also run an offline model for comparison


Step 3 — Create a Single Runner Script
Create one file called run_benchmark.py.
Do not abstract. Do not over‑engineer.
What the script does
For each prompt in prompts.csv:

Call ChatGPT
Call Gemini
Call Perplexity
Capture: Full response text
Citation URLs (if present)
Write results to a CSV
Output schema
date,provider,model,prompt_id,prompt,response,citation_url,citation_domain


Rules:

One citation = one row
If a response has no citations, write NONE


Step 4 — Normalize Learn Citations
Apply minimal normalization in code.
Rules:

docs.microsoft.com → learn.microsoft.com
learn.microsoft.com → learn.microsoft.com
Do not normalize anything else
This is enough to answer leadership questions.


Step 5 — Write CSV to SharePoint / OneDrive
Fastest method:

Mount OneDrive or a SharePoint library locally
Write files directly to the filesystem
Example path:
/AI-Citation-Benchmark/results_YYYY-MM-DD.csv


No Graph API, no auth complexity.


Step 6 — Run It Manually (Your First Demo)
From a terminal:
python run_benchmark.py


Open the CSV in Excel.
You can now immediately answer:

Is learn.microsoft.com being cited?
Which providers cite Learn most often?
What other domains compete with Learn?
This is already demo‑worthy.


Step 7 — Make It Daily (Easiest Options)
Option A — Local scheduler (fastest)

Windows Task Scheduler or cron
Runs once per day
Appends a new CSV
Option B — Power Automate (optional)

Scheduled flow
Calls a local endpoint or Azure Function
Uploads CSV to SharePoint
Use Option B only if you already like Power Automate.


Step 8 — Analyze in Excel (First Pass)
Create a Pivot Table:

Rows: provider
Values: count of rows where citation_domain = learn.microsoft.com
Add:

% of total citations
Trend by date
This is sufficient for leadership discussion.


Optional — Add a Copilot Agent (Later)
Once CSVs exist:

Create a Copilot agent that reads the files
Ask questions like: “Is Learn citation share increasing?”
“Who beats Learn most often in Gemini?”
Important:

The Copilot agent does not run the benchmark
It only analyzes outputs


What This Approach Proves

Uses real model responses
Shows observable citation behavior
Identifies competing sources
Is repeatable and trendable
What It Does Not Claim

No claims about training data
No attribution or traffic claims
No guarantees across all users


Minimal Checklist

✅ prompts.csv
✅ one runner script
✅ three API keys
✅ CSV in SharePoint
✅ Excel pivot
This is the fastest credible path to a Learn citation benchmark you can personally own.
