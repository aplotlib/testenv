"""
Time-savings estimation.

Turns "we processed N rows in T seconds" into the number a business owner
actually cares about: how much analyst time this replaced.

Baselines are per-item manual handling times, set from how long it takes a
human to read one item and record the category they think applies:

  * Amazon return reason  ->  7 seconds
  * B2B / Zendesk support ticket -> 25 seconds  (longer: real ticket bodies,
    often with quoted email threads, plus SKU lookup)

Both are configurable at the UI layer so the assumption can be challenged and
adjusted live in a demo rather than being baked in.

Pure module — no Streamlit, no I/O — so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Manual handling time per item, in seconds.
SECONDS_PER_AMAZON_RETURN = 7
SECONDS_PER_SUPPORT_TICKET = 25


def format_duration(seconds: float, *, precise: bool = False) -> str:
    """Human-readable duration.

    >>> format_duration(31906)
    '8h 51m'
    >>> format_duration(750)
    '12m 30s'
    >>> format_duration(1.84, precise=True)
    '1.8s'
    """
    seconds = max(0.0, float(seconds))

    if seconds < 60:
        # Sub-minute runtimes are the tool's own speed — worth a decimal.
        return f"{seconds:.1f}s" if precise and seconds < 10 else f"{int(round(seconds))}s"

    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"

    hours, rem = divmod(int(round(seconds)), 3600)
    minutes = rem // 60
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def format_workdays(seconds: float, hours_per_day: float = 8.0) -> str:
    """Express a duration as working days — the framing leadership thinks in."""
    days = seconds / 3600.0 / hours_per_day
    if days < 0.1:
        return "under a tenth of a workday"
    if days < 1:
        return f"{days:.1f} of a workday"
    if days < 2:
        return f"{days:.1f} workday"
    return f"{days:.1f} workdays"


@dataclass(frozen=True)
class Savings:
    """Manual-vs-actual comparison for one processing run."""

    items: int
    seconds_per_item: float
    actual_seconds: float

    @property
    def manual_seconds(self) -> float:
        """Estimated time for a human to do the same work by hand."""
        return self.items * self.seconds_per_item

    @property
    def saved_seconds(self) -> float:
        """Never negative — if the tool were somehow slower, savings are zero."""
        return max(0.0, self.manual_seconds - self.actual_seconds)

    @property
    def speedup(self) -> float:
        """How many times faster than manual. 0.0 when not yet measurable."""
        if self.actual_seconds <= 0:
            return 0.0
        return self.manual_seconds / self.actual_seconds

    # ── display helpers ────────────────────────────────────────────────────
    @property
    def manual_display(self) -> str:
        return format_duration(self.manual_seconds)

    @property
    def actual_display(self) -> str:
        return format_duration(self.actual_seconds, precise=True)

    @property
    def saved_display(self) -> str:
        return format_duration(self.saved_seconds)

    @property
    def workdays_display(self) -> str:
        return format_workdays(self.manual_seconds)

    @property
    def speedup_display(self) -> str:
        x = self.speedup
        if x <= 0:
            return "n/a"
        return f"{x:,.0f}x faster" if x >= 10 else f"{x:.1f}x faster"

    def labor_cost_saved(self, hourly_rate: float) -> float:
        """Value of the saved time at a fully-loaded hourly rate."""
        return self.saved_seconds / 3600.0 * max(0.0, hourly_rate)


def estimate_amazon(items: int, actual_seconds: float = 0.0,
                    seconds_per_item: float = SECONDS_PER_AMAZON_RETURN) -> Savings:
    return Savings(items=max(0, items), seconds_per_item=seconds_per_item,
                   actual_seconds=max(0.0, actual_seconds))


def estimate_tickets(items: int, actual_seconds: float = 0.0,
                     seconds_per_item: float = SECONDS_PER_SUPPORT_TICKET) -> Savings:
    return Savings(items=max(0, items), seconds_per_item=seconds_per_item,
                   actual_seconds=max(0.0, actual_seconds))
