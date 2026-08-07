"""Verify partial-work durability, resume, and sample-limit behavior.

Simulates the real failure mode: the script gets killed mid-run (Streamlit
rerun / tab close / timeout raises through, it is not a normal Exception).
The categorized rows completed before that point must survive.

Run from the repository root:

    python test_partial_recovery.py
"""
import os
import sys

# Import app.py from this file's own directory, so the test works regardless of
# where it is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import app


# ── minimal Streamlit stand-in ────────────────────────────────────────────────
class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class _Widget:
    def progress(self, *a, **k): pass
    def success(self, *a, **k): pass
    def text(self, *a, **k): pass
    def empty(self, *a, **k): pass
    def metric(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _MockSt:
    def __init__(self):
        self.session_state = _State()

    def __getattr__(self, name):
        # info/caption/warning/error/success/markdown/divider/... all no-op
        return lambda *a, **k: _Widget()

    def progress(self, *a, **k): return _Widget()
    def empty(self, *a, **k): return _Widget()
    def container(self, *a, **k): return _Widget()
    def columns(self, n, *a, **k): return [_Widget() for _ in range(n if isinstance(n, int) else len(n))]
    def button(self, *a, **k): return False       # Stop never auto-pressed
    def expander(self, *a, **k): return _Widget()
    def rerun(self): raise AssertionError("unexpected rerun")


mock = _MockSt()
app.st = mock
mock.session_state.update({
    "chunk_size": 100, "batch_size": 5, "processing_errors": [],
    "row_confidence": {}, "sec_per_return": 7.0, "rows_done": 0, "rows_target": 0,
    "run_state": "idle", "processing_elapsed": 0.0,
})

# ── build a 20-row file shaped like the real export ───────────────────────────
N = 20
df = pd.DataFrame({
    "Date": ["2026-06-01"] * N,
    "Tag": [""] * N,
    "Imported SKU": [f"SKU{i:04d}" for i in range(N)],
    "UDI": [""] * N, "CS": [""] * N, "Order": [""] * N, "Source": [""] * N,
    "Agent": [""] * N,
    "Complaint": [f"complaint number {i}" for i in range(N)],
    "Stars": [""] * N,
    "Category": [""] * N,
    "No": [""] * N,
})
cm = {"complaint": "Complaint", "category": "Category", "sku": "Imported SKU"}

KILL_AFTER = 10


class KilledAnalyzer:
    """Categorizes rows, then dies the way a Streamlit rerun does."""

    def __init__(self, kill_after=None):
        self.done = 0
        self.kill_after = kill_after

    def categorize_batch(self, batch, mode="standard"):
        if self.kill_after is not None and self.done >= self.kill_after:
            raise KeyboardInterrupt("simulated tab close / rerun")
        out = []
        for item in batch:
            out.append({"index": item["index"], "category": "Size: Too Small", "confidence": 0.85})
            self.done += 1
        return out


# ── 1. interrupted run must leave partial work intact ─────────────────────────
killed = False
try:
    app.process_in_chunks(df, KilledAnalyzer(kill_after=KILL_AFTER), cm)
except KeyboardInterrupt:
    killed = True

assert killed, "expected the simulated kill to propagate"

filled = int((df["Category"].astype(str).str.strip() != "").sum())
print(f"[ok] interrupted mid-run: {filled} of {N} rows categorized and RETAINED")
assert filled == KILL_AFTER, f"expected {KILL_AFTER} retained, got {filled}"

# The DataFrame published to session state must be the same object, so the
# partial rows are visible to the UI after the kill.
assert mock.session_state["categorized_data"] is df, "partial df not published to session state"
print("[ok] partial DataFrame is live in session state (same object)")
assert mock.session_state["rows_done"] == KILL_AFTER, mock.session_state["rows_done"]
print(f"[ok] rows_done recorded as {mock.session_state['rows_done']}")
assert mock.session_state["run_state"] == "running", mock.session_state["run_state"]
print("[ok] run_state left as 'running' -> UI will detect this as interrupted")

# ── 2. remaining-work accounting ──────────────────────────────────────────────
remaining = app.count_uncategorized(df, cm)
assert remaining == N - KILL_AFTER, remaining
print(f"[ok] count_uncategorized reports {remaining} rows still to do")

# ── 3. partial filename is unmistakable ───────────────────────────────────────
fn = app._export_filename(partial=True)
assert "PARTIAL" in fn and f"{KILL_AFTER}of" in fn, fn
print(f"[ok] partial export filename: {fn}")
assert "PARTIAL" not in app._export_filename(partial=False)
print(f"[ok] complete export filename: {app._export_filename(partial=False)}")

# ── 4. resume finishes only the leftovers, without redoing work ───────────────
resumer = KilledAnalyzer()
app.process_in_chunks(df, resumer, cm, only_uncategorized=True)
assert resumer.done == N - KILL_AFTER, f"resume did {resumer.done}, expected {N - KILL_AFTER}"
print(f"[ok] resume processed exactly the {resumer.done} remaining rows (no rework)")
assert app.count_uncategorized(df, cm) == 0
assert int((df["Category"].astype(str).str.strip() != "").sum()) == N
print("[ok] all rows categorized after resume")
assert mock.session_state["run_state"] == "complete"
print("[ok] run_state == 'complete'")

# ── 5. sample mode caps the work ──────────────────────────────────────────────
df2 = df.copy()
df2["Category"] = ""
mock.session_state["row_confidence"] = {}
sampler = KilledAnalyzer()
app.process_in_chunks(df2, sampler, cm, limit=7)
assert sampler.done == 7, sampler.done
assert int((df2["Category"].astype(str).str.strip() != "").sum()) == 7
print(f"[ok] sample mode processed exactly 7 rows, left {app.count_uncategorized(df2, cm)} for later")

# ── 6. confidence captured for the review queue ───────────────────────────────
assert len(mock.session_state["row_confidence"]) == 7, mock.session_state["row_confidence"]
print("[ok] per-row confidence captured for the review queue")

print("\nALL PARTIAL/RESUME TESTS PASSED")
