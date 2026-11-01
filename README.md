# Uber Ireland · Customer Experience Dashboard

Interactive Streamlit + Plotly dashboard for the Google Forms survey
"Uber Customer Service Survey – Ireland". Reads the response sheet live, so new
answers appear after a refresh (cached 5 minutes; use **↻ Refresh responses** to force it).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The Google Sheet must be shared as **Anyone with the link → Viewer**. If it can't be
reached, the app falls back to `data/survey_snapshot.csv` (or a CSV you upload).

## What's inside
| Tab | Covers |
|---|---|
| Executive summary | Service scorecard, top highlights/lowlights, loyalty snapshot, service-recovery funnel |
| ✦ AI insights | Auto-written verdict, highlights, lowlights, watch-outs, key-driver matrix, prioritised actions, optional Claude narrative, Markdown export |
| Respondent profile | Q1–Q3, sunburst drill-down, age × residency, response timeline |
| Service ratings | Q5 diverging distribution, radar + heatmap by segment, single-aspect drill-down |
| Problems & support | Q4, Q4b (multi-select parsed), co-occurrence, Q6, problem vs no-problem ratings |
| Loyalty & trust | Q7, Q8, Q9 distribution/bands, trust by segment, correlation matrix, rating → loyalty |
| Competition & VoC | Q10 + normalised competitor names, Q11 theme tagging, searchable comments |
| Response explorer | Click/box-select riders on a CX-vs-trust map, full answer table, respondent card, CSV exports |

Sidebar filters (date, age, residency, usage, problem, competitor view, CX Index) apply everywhere;
"Compare segments by" switches every segment chart at once.

## AI narrative (optional)
The highlights/lowlights engine (`insights.py`) runs without any key. To also get a narrative
written by Claude, add your key to `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-sonnet-5-5"
```
Only aggregated, anonymous results are sent — never raw rows. Don't commit `secrets.toml`.

## Deploy free on Streamlit Community Cloud
Push this folder to GitHub → share.streamlit.io → New app → pick the repo and `app.py` →
add the key under *Advanced settings → Secrets* if you want the Claude narrative.

## Metric definitions
- **CX Index** – mean of the five Q5 ratings (Very poor = 0 … Excellent = 100).
- **Good+** – share rating Good or Excellent.
- **Key-driver importance** – Spearman correlation of each touchpoint with Overall experience.
- With fewer than 30 responses the AI tab labels findings as directional.
