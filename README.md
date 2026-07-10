---
title: Vive Health Quality Suite
emoji: 🏥
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# Vive Health Quality Suite

Internal Streamlit application for medical-device quality management:
AI-powered Amazon return categorization, B2B/Zendesk reporting, CAPA
workflows, quality screening, VoC analysis, and regulatory intelligence.

## Stack

- **UI:** Streamlit (`app.py` is the single entry point)
- **AI:** Anthropic (Claude) only — Haiku 4.5 / Sonnet 4.6 / Opus 4.6,
  selected per task. No other AI providers are used.
- **Data:** analyst-uploaded CSV/XLSX files, processed in memory.
  No business data is stored in this repository.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configuration is via Streamlit secrets (`.streamlit/secrets.toml` locally,
**Settings → Secrets** on Streamlit Cloud). See
`.streamlit/secrets.toml.example` for the template:

| Secret | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | All AI features (categorization, summaries, chat) |
| `APP_PASSCODE` | No | Enables an app-level passcode gate when set |

## Security posture (for IT review)

- **Secrets:** resolved from Streamlit secrets or environment variables
  only; never hardcoded, logged, or echoed. No secrets exist in the git
  history (full-history scan performed 2026-07-08).
- **Data at rest:** uploaded files are processed in memory; exports are
  generated in memory for download. User-saved connection configs and
  uploaded service-account credentials are stored under `~/.quality_app/`
  on the host — never inside the repository. Business data files
  (`*.csv`, `*.xlsx`, …) are gitignored and were purged from the entire
  git history.
- **Rendering:** values derived from uploaded files are HTML-escaped
  before any raw-HTML rendering (stored-XSS hardening).
- **Network egress:** Anthropic API, openFDA, and (optionally) Google CSE /
  translation services and user-configured data sources. All HTTP calls
  carry timeouts. TLS verification is never disabled.
- **Access control:** the app has no user-account system. Restrict access
  via the hosting platform (Streamlit Cloud viewer allowlist) and/or the
  `APP_PASSCODE` gate. Do not host it as a public app.
- **Known limitation:** optional data-source connection credentials
  (database passwords, Smartsheet tokens) are stored unencrypted in
  `~/.quality_app/data_connections.json` on the host. If this feature is
  used in production, migrate to a proper secrets manager.

## Repository hygiene

- `.gitignore` blocks data files, credentials, local settings, and caches.
- The repository must remain **private**.
