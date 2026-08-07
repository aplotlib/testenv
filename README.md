# Vive Health — Returns Categorizer

Internal Streamlit app with three focused tools:

| Tool | Input | Output |
|---|---|---|
| **Amazon Return Categorizer** | Amazon FBA customer-returns export (complaint text in column I) | Same file with the quality category filled into column K, plus `Category Summary` and `By SKU` sheets |
| **B2B Report** | Raw Odoo Helpdesk ticket export (`Display Name`, `Description`) | `B2B Report` sheet: Display Name, Description, SKU, Category, Reason |
| **Zendesk B2C Report** | Zendesk quality-issues export (`Ticket created - Date`, `Ticket ID`, `SKU`, `Issue`, `Ticket Type`) | Quality report aggregated by parent SKU (first 7 chars) |

All three assign categories from one shared 24-value taxonomy
(`MEDICAL_DEVICE_CATEGORIES` in `enhanced_ai_analysis.py`).

## Stack

- **UI:** Streamlit — `app.py` is the single entry point
- **AI:** Anthropic (Claude) only — Haiku / Sonnet / Opus, selected per task.
  No other AI providers are used.
- **Data:** analyst-uploaded CSV/XLSX, processed in memory. No business data is
  stored in this repository.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configuration is via Streamlit secrets (`.streamlit/secrets.toml` locally,
**Settings → Secrets** on Streamlit Cloud). See `.streamlit/secrets.toml.example`:

| Secret | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | All AI features |
| `APP_PASSCODE` | No | Enables an app-level passcode gate when set |

## How the Amazon categorizer decides a category

Cheapest path first, so most rows never reach the API:

1. **Corrections memory** — exact match on a complaint a reviewer previously
   corrected. Free, instant, and authoritative.
2. **Pattern tier** (`quick_categorize`) — regex rules, accepted **only** when
   exactly one category matches. Ambiguous rows defer to the AI. Free.
3. **Claude** — full taxonomy prompt with the reviewer's past corrections
   injected as few-shot examples.

Two rules are load-bearing — both previously caused a real accuracy regression:

- The Amazon **reason-code** column (`UNWANTED_ITEM`, `DEFECTIVE`, …) must never
  be mistaken for, or override, the free-text complaint. It is a weak hint only.
- Complaint text is **preprocessed** (HTML-entity decode, contraction expansion)
  *before* pattern matching. Skipping this silently degrades accuracy.

### Accuracy benchmark

`validate_categorizer.py` measures the categorizer against a human-reviewed
export. Run it after any change to the pipeline:

```bash
# free — pattern tier only, no API calls
python validate_categorizer.py --truth categorized_20260630.xlsx

# include the AI tier on a 200-row sample (costs money)
python validate_categorizer.py --truth categorized_20260630.xlsx --ai --sample 200
```

Last run against the June 2026 reviewed export (4,558 rows):
**pattern tier resolved 3,024 rows (66.3%) at 100.0% precision** — zero
disagreements with the reviewed categories. The remaining ~1,530 rows go to Claude.

The script exits non-zero if precision falls below `--min-precision` (default
0.80), so it works as a regression guard.

## Teaching the tool

In the Amazon tab, **Correct categories** lets a reviewer override any assigned
category. Saved corrections are applied as exact matches on future runs and
injected as few-shot examples into the AI prompt, so the same complaint text is
never miscategorized twice. Corrections persist in `~/.quality_app/`, outside
the repository.

## Security posture (for IT review)

- **Secrets:** resolved from Streamlit secrets or environment variables only;
  never hardcoded, logged, or echoed. No secrets exist in the git history
  (full-history scan performed 2026-07-08).
- **Data at rest:** uploads are processed in memory; exports are generated in
  memory for download. Reviewer corrections are stored under `~/.quality_app/`
  on the host — never in the repository. Business data files (`*.csv`, `*.xlsx`)
  are gitignored and were purged from the entire git history.
- **Network egress:** the Anthropic API only. All HTTP calls carry timeouts;
  TLS verification is never disabled.
- **Access control:** the app has no user-account system. Restrict access via
  the Streamlit Cloud viewer allowlist and/or the `APP_PASSCODE` gate. Do not
  host it as a public app.

## Repository layout

```
app.py                      # entry point — the three tools and their UI
enhanced_ai_analysis.py     # Claude engine, taxonomy, pattern tier  (accuracy-critical)
corrections_memory.py       # persistent reviewer corrections
b2b_zendesk_reporting.py    # Zendesk B2C aggregate reporting tool
validate_categorizer.py     # accuracy benchmark / regression guard
legacy/app_v33_full.py      # previous 10k-line Quality Suite, kept for reference only
```

`legacy/` is not imported and does not run. It holds the former all-in-one app
(CAPA, FMEA, screening wizard, VoC, regulatory intelligence) in case any of
those tools need to be revived. Its extra dependencies are **not** in
`requirements.txt`.

- `.gitignore` blocks data files, credentials, local settings, and caches.
- The repository must remain **private**.
