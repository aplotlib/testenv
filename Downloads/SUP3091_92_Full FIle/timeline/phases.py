import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def draw_phase_bands(ax, events, phases):
    """Draw tinted background band + label for each phase.

    Phase bands are derived from event dates so layout adjusts automatically.
    The DECISIONS and INVESTIGATION phases overlap in time; both are drawn at
    12% opacity so the overlap blends naturally.
    """
    X_PAD = 8   # days padding on each side of a phase's event cluster
    BAND_Y_BOTTOM = 0.44  # intentionally above spine only (per design: bands are upper-half context zones)
    BAND_Y_TOP    = 1.00

    for phase in phases:
        phase_events = [e for e in events if e['phase'] == phase['key']]
        if not phase_events:
            continue

        dates_num = [mdates.date2num(e['date']) for e in phase_events]
        x_start = min(dates_num) - X_PAD
        x_end   = max(dates_num) + X_PAD

        rect = mpatches.Rectangle(
            (x_start, BAND_Y_BOTTOM),
            x_end - x_start,
            BAND_Y_TOP - BAND_Y_BOTTOM,
            facecolor=phase['color'],
            alpha=0.12,
            edgecolor='none',
            zorder=1,
        )
        ax.add_patch(rect)

        mid_x = (x_start + x_end) / 2
        r, g, b, _ = mcolors.to_rgba(phase['color'])
        ax.text(
            mid_x, 0.97,
            phase['name'],
            ha='center', va='top',
            fontsize=8.5, fontweight='bold',
            color=(r, g, b, 0.80),
            fontfamily='DejaVu Sans',
            zorder=3,
        )
