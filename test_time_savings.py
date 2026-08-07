#!/usr/bin/env python
"""Unit tests for time_savings.py — plain asserts, no test framework needed.

    python test_time_savings.py
"""

import sys

from time_savings import (
    SECONDS_PER_AMAZON_RETURN,
    SECONDS_PER_SUPPORT_TICKET,
    estimate_amazon,
    estimate_tickets,
    format_duration,
    format_workdays,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def check_close(label, got, want, tol=0.01):
    if abs(got - want) > tol:
        failures.append(f"{label}: got {got!r}, want ~{want!r}")


# ── format_duration ────────────────────────────────────────────────────────
check("sub-10s precise", format_duration(1.84, precise=True), "1.8s")
check("sub-minute", format_duration(45), "45s")
check("rounds to seconds", format_duration(45.6), "46s")
check("exact minute", format_duration(60), "1m")
check("minutes+seconds", format_duration(750), "12m 30s")
check("exact hour", format_duration(3600), "1h")
check("hours+minutes", format_duration(31906), "8h 51m")  # 8h 51m 46s, minutes floor
check("negative clamps", format_duration(-100), "0s")
check("zero", format_duration(0), "0s")

# ── format_workdays ────────────────────────────────────────────────────────
check("tiny workday", format_workdays(60), "under a tenth of a workday")
check("partial workday", format_workdays(3600 * 4), "0.5 of a workday")
check("one workday", format_workdays(3600 * 9), "1.1 workday")
check("many workdays", format_workdays(3600 * 24), "3.0 workdays")

# ── Amazon: the real June file, 4,558 categorized rows at 7s each ──────────
s = estimate_amazon(4558, actual_seconds=107.0)
check_close("manual seconds", s.manual_seconds, 4558 * 7)
check("manual display", s.manual_display, "8h 51m")
check_close("saved seconds", s.saved_seconds, 4558 * 7 - 107.0)
check("speedup is large", s.speedup > 250, True)
check("speedup display", s.speedup_display, "298x faster")
check("default amazon rate", s.seconds_per_item, SECONDS_PER_AMAZON_RETURN)

# ── Tickets: 176 helpdesk tickets at 25s each ──────────────────────────────
t = estimate_tickets(176, actual_seconds=60.0)
check_close("ticket manual seconds", t.manual_seconds, 176 * 25)
check("ticket manual display", t.manual_display, "1h 13m")
check("default ticket rate", t.seconds_per_item, SECONDS_PER_SUPPORT_TICKET)

# ── Edge cases ─────────────────────────────────────────────────────────────
z = estimate_amazon(0)
check("zero items manual", z.manual_seconds, 0)
check("zero items speedup", z.speedup, 0.0)
check("zero items speedup display", z.speedup_display, "n/a")

unmeasured = estimate_amazon(100)  # no actual_seconds yet (pre-run estimate)
check("unmeasured speedup", unmeasured.speedup, 0.0)
check_close("unmeasured still estimates manual", unmeasured.manual_seconds, 700)
check("unmeasured saved == manual", unmeasured.saved_seconds, 700)

slower = estimate_amazon(1, actual_seconds=999.0)  # tool slower than a human
check("savings never negative", slower.saved_seconds, 0.0)

neg = estimate_amazon(-5)
check("negative items clamp", neg.items, 0)

# ── labor cost ─────────────────────────────────────────────────────────────
c = estimate_amazon(4558, actual_seconds=107.0)
check_close("labor cost at $30/h", c.labor_cost_saved(30), c.saved_seconds / 3600 * 30, 0.01)
check("labor cost negative rate clamps", c.labor_cost_saved(-50), 0.0)

# ── custom rate override (demo can adjust the assumption live) ─────────────
custom = estimate_amazon(1000, actual_seconds=10.0, seconds_per_item=12)
check_close("custom rate", custom.manual_seconds, 12000)

if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all time_savings tests passed")
