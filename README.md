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
**pattern tier resolved 3,237 rows (71.0%) at 100.0% precision** — zero
disagreements with the reviewed categories. The remaining ~1,320 rows go to Claude.

The script exits non-zero if precision falls below `--min-precision` (default
0.80), so it works as a regression guard.

#### Guarding against overfitting

The collision-resolution rules were derived by inspecting the reviewed export,
so measuring them on that same file would flatter them. `validate_holdout.py`
splits the data 50/50 on a fixed seed and reports both halves:

```bash
python validate_holdout.py categorized_20260630.xlsx
```

Current result — **100.00% precision on both** the train half and the held-out
half at 71.0% coverage. Matching numbers mean the rules generalize rather than
memorizing this month's file. **If you tune the patterns, re-run this**; a train
score much higher than held-out means you have overfitted and next month's
export will regress.

### Full test suite

```bash
python test_time_savings.py        # time-savings maths
python test_partial_recovery.py    # interrupt / resume / sample-run durability
python validate_categorizer.py --truth categorized_20260630.xlsx
python validate_holdout.py categorized_20260630.xlsx
```

## Time savings

Every run reports the analyst time it replaced, using per-item manual handling
baselines that are **adjustable in the sidebar** so the assumption can be
challenged live rather than taken on faith:

| Item | Default manual time |
|---|---|
| Amazon return reason | 7 seconds |
| B2B / Zendesk support ticket | 25 seconds |

The file's worth is shown *before* you run it ("8h 51m of manual work in this
file"), and the saved time is shown as the headline afterwards. For the June
2026 export: **4,558 returns = 8h 51m by hand**, about 1.1 workdays.

## If a run stops early

Long runs are interruptible without losing work. Partial results are durable by
design: the DataFrame is published to session state before the first API call
and mutated in place, and the downloadable export is regenerated after **every
chunk**. So if a run stops — Stop pressed, tab closed, session timed out —
everything already categorized is kept.

- **⏸ Stop and keep what's done** — halts the run, keeps every completed row.
- **Partial downloads are labelled in the filename**
  (`categorized_PARTIAL_1620of4558_20260807.xlsx`) so a half-finished file can
  never be mistaken for a complete one downstream.
- **▶️ Resume** — processes only the rows still missing a category. No rework,
  no double API spend.
- **🧪 Test on first 100** — a cheap dry run to confirm a new file parses
  correctly before committing to thousands of API calls.

`test_partial_recovery.py` covers all of this, including simulating a killed
run mid-flight.

One limitation to be aware of: durability is per **session**. It survives
interrupts, reruns, and Stop, but not the Streamlit Cloud container being
recycled. Download the partial file if you need to step away for a long time.

## Teaching the tool

In the Amazon tab, **Correct categories** lets a reviewer override any assigned
category. Saved corrections are applied as exact matches on future runs and
injected as few-shot examples into the AI prompt, so the same complaint text is
never miscategorized twice. Corrections persist in `~/.quality_app/`, outside
the repository.

Rows are listed **least-confident first**, not in file order — so review time
goes to the rows the AI actually had doubt about, rather than to rows the
pattern tier already resolved with certainty.

## For the analyst

- **By SKU** flags products with ≥60% quality-issue returns on meaningful
  volume (⚠️), which is the CAPA shortlist. Downloadable as CSV.
- **Search complaints** does free-text search across the whole file, so you
  don't have to export to Excel just to find every mention of "strap".
- **Top products by ticket volume** in the B2B tab, with the dominant issue per
  SKU.
- **Cost tracking** in the sidebar shows spend and how many rows were resolved
  free of charge.

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
