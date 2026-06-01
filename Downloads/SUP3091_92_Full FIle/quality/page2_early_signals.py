import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from quality.data import MONTHLY_TICKETS, KEY_DATES, COLORS
from quality.canvas import new_page, add_page_header, add_footer, draw_sidebar_callouts

FOOTER = ('Complaint trend: all helpdesk ticket types (quality + logistics) — trend is a valid signal proxy. '
          'Cold season: Nov–Mar. Key dates from QA/QC records and internal discussion threads.')

SIDEBAR = [
    {
        'accent':       '#2E8B7A',
        'header':       'What QA recommended',
        'body':         'Stop-sale + full rework scope.\nFlagged Nov 2025, reinforced Jan–Feb 2026\nas complaint volume rose and cold-\nembrittlement was confirmed.',
        'header_color': '#FFFFFF',
        'bg_alpha':     0.90,
    },
    {
        'accent':       '#C0392B',
        'header':       'What was decided',
        'body':         'Continue selling.\nRework in parallel. No stop-ship.',
        'header_color': '#FFFFFF',
        'bg_alpha':     0.85,
    },
    {
        'accent': '#4A7DB5',
        'header': '6-month gap',
        'body':   'Between first QA flag (Nov 12, 2025)\nand stop-ship (May 15, 2026).',
    },
]

MONTH_LABELS = list(MONTHLY_TICKETS.keys())

# Cold season: indices 7–11 (Nov '25 – Mar '26)
COLD_START = 7
COLD_END   = 11

# Map date_str to index in MONTH_LABELS
DATE_TO_IDX = {
    '2025-11': 7,
    '2026-01': 9,
    '2026-02': 10,
    '2026-05': 13,
}


def build_page():
    fig = new_page()
    add_page_header(fig, 'EARLY SIGNALS & WHAT WAS RECOMMENDED',
                    'Monthly complaint trend, cold-season correlation, and the recommendation record')
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[0.06, 0.55, 0.39],
                  left=0.10, right=0.97, top=0.92, bottom=0.07,
                  hspace=0.42, wspace=0.10)

    ax_title   = fig.add_subplot(gs[0, :])
    ax_trend   = fig.add_subplot(gs[1, 0])
    ax_cold    = fig.add_subplot(gs[2, 0])
    ax_sidebar = fig.add_subplot(gs[1:, 1])
    ax_title.axis('off')

    tickets = list(MONTHLY_TICKETS.values())
    xs = np.arange(len(tickets))

    # ── Chart A: monthly complaint trend ──────────────────────────────────
    ax_trend.axvspan(COLD_START - 0.5, COLD_END + 0.5,
                     alpha=0.08, color=COLORS['cold_shading'], zorder=1)
    ax_trend.plot(xs, tickets, color='#2C2C2C', linewidth=1.4, zorder=3)
    ax_trend.scatter(xs, tickets, s=28, color='#2C2C2C', zorder=4)

    # Annotate key dates — deduplicate by index
    seen = set()
    y_offsets = {7: 3.5, 9: 3.0, 10: -5.0, 13: 3.0}
    for d in KEY_DATES:
        idx = DATE_TO_IDX.get(d['date_str'])
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        y_off = y_offsets.get(idx, 3.0)
        ax_trend.annotate(
            d['label'],
            xy=(idx, tickets[idx]),
            xytext=(idx, tickets[idx] + y_off),
            fontsize=5.5, color='#444444', fontfamily='DejaVu Sans',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.6),
        )

    ax_trend.text((COLD_START + COLD_END) / 2, 1.5,
                  'Cold season  Nov–Mar',
                  ha='center', va='bottom', fontsize=6,
                  color='#E07B39', fontfamily='DejaVu Sans', style='italic')
    ax_trend.set_xticks(xs[::2])
    ax_trend.set_xticklabels(MONTH_LABELS[::2], fontsize=6.5, rotation=30,
                              ha='right', fontfamily='DejaVu Sans')
    ax_trend.set_ylabel('Helpdesk tickets / month', fontsize=7.5,
                         color='#666666', fontfamily='DejaVu Sans')
    ax_trend.set_title('Monthly complaint volume with cold-season overlay',
                        fontsize=9, fontweight='bold', color='#2C2C2C',
                        fontfamily='DejaVu Sans', loc='left', pad=6)
    ax_trend.set_facecolor('#F8F9FA')
    for spine in ax_trend.spines.values():
        spine.set_visible(False)
    ax_trend.yaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_trend.set_axisbelow(True)
    ax_trend.tick_params(axis='y', labelsize=7, colors='#888888')
    ax_trend.set_ylim(0, 30)

    # ── Chart B: cold vs. warm month plastic defect rate ──────────────────
    bar_labels  = ['Warm months\n(Apr–Oct)', 'Cold months\n(Nov–Mar)']
    bar_rates   = [1.0, 1.57]
    bar_colors  = ['#AAAAAA', COLORS['plastic']]
    bars = ax_cold.bar(bar_labels, bar_rates, color=bar_colors, width=0.45,
                       edgecolor='white', linewidth=0.5)
    for bar, rate in zip(bars, bar_rates):
        ax_cold.text(bar.get_x() + bar.get_width() / 2, rate + 0.05,
                     f'{rate:.2f}×', ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#2C2C2C',
                     fontfamily='DejaVu Sans')
    ax_cold.annotate('', xy=(1, 1.57), xytext=(1, 1.0),
                     arrowprops=dict(arrowstyle='->', color=COLORS['plastic'], lw=1.2))
    ax_cold.text(1.25, 1.28, '+57%\nin cold', fontsize=7, color=COLORS['plastic'],
                 fontfamily='DejaVu Sans', va='center')
    ax_cold.text(0.5, 0.12, 'PA6/PP-GF cold-embrittlement confirmed Feb 9, 2026',
                 ha='center', va='bottom', fontsize=6, color='#666666',
                 fontfamily='DejaVu Sans', style='italic',
                 transform=ax_cold.transAxes)
    ax_cold.set_title('Plastic defect rate: cold vs. warm months',
                       fontsize=9, fontweight='bold', color='#2C2C2C',
                       fontfamily='DejaVu Sans', loc='left', pad=6)
    ax_cold.set_ylabel('Relative rate (warm = 1.0)', fontsize=7.5,
                        color='#666666', fontfamily='DejaVu Sans')
    ax_cold.set_ylim(0, 2.2)
    ax_cold.set_facecolor('#F8F9FA')
    for spine in ax_cold.spines.values():
        spine.set_visible(False)
    ax_cold.yaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_cold.set_axisbelow(True)
    ax_cold.tick_params(labelsize=7.5, colors='#888888')

    draw_sidebar_callouts(ax_sidebar, SIDEBAR)
    add_footer(fig, FOOTER)
    return fig
