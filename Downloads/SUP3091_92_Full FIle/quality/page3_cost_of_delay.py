import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from quality.data import (
    HALF_RATES, MONTHLY_TICKETS, MONTHLY_RETURNS,
    REWORK_WAREHOUSES, COLORS,
)
from quality.canvas import new_page, add_page_header, add_footer, draw_sidebar_callouts

FOOTER = ('H1 = Jun–Nov 2025; H2 = Dec 2025–May 2026. '
          'Return data: all-cause (includes wrong size, preference changes). '
          'Rework data: Master Stockout Analysis 2026-04-17. '
          'Defect rate +95% HoH; +77% excluding bulk claims.')

SIDEBAR = [
    {
        'accent': '#E07B39',
        'header': 'Defect rate nearly doubled',
        'body':   'From 1.9 to 3.7 per 1,000 in the second\nhalf — while product kept shipping.',
    },
    {
        'accent': '#C0392B',
        'header': 'JX warehouse gap',
        'body':   'Defective units marked non-sellable in\nOdoo but NOT placed in QC hold.\nDiscovered late May 2026.',
    },
    {
        'accent': '#4A7DB5',
        'header': 'Rework burden',
        'body':   '4 warehouses, 10,892 sleeves, 3 direction\nchanges before final phased plan.\nTeams reworking while fulfilling orders.',
    },
]

# Month labels for May'25–May'26 (13 months, matching MONTHLY_RETURNS)
MONTH_LABELS_13 = [
    "May '25", "Jun '25", "Jul '25", "Aug '25", "Sep '25", "Oct '25", "Nov '25",
    "Dec '25", "Jan '26", "Feb '26", "Mar '26", "Apr '26", "May '26",
]


def build_page():
    fig = new_page()
    add_page_header(fig, 'THE COST OF SELLING THROUGH',
                    'Defect rate escalation, the hidden signal in returns, and rework burden')
    gs = GridSpec(4, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[0.05, 0.29, 0.32, 0.34],
                  left=0.12, right=0.97, top=0.92, bottom=0.07,
                  hspace=0.50, wspace=0.10)

    ax_title   = fig.add_subplot(gs[0, :])
    ax_half    = fig.add_subplot(gs[1, 0])
    ax_diverge = fig.add_subplot(gs[2, 0])
    ax_rework  = fig.add_subplot(gs[3, 0])
    ax_sidebar = fig.add_subplot(gs[1:, 1])
    ax_title.axis('off')

    # ── Chart A: H1 vs H2 defect rate ────────────────────────────────────
    labels = list(HALF_RATES.keys())
    rates  = list(HALF_RATES.values())
    bars = ax_half.bar(labels, rates,
                       color=['#AAAAAA', COLORS['plastic']],
                       width=0.38, edgecolor='white', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax_half.text(bar.get_x() + bar.get_width() / 2, rate + 0.06,
                     f'{rate:.1f}/1,000', ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#2C2C2C',
                     fontfamily='DejaVu Sans')
    ax_half.annotate('', xy=(1, 3.7), xytext=(1, 1.9),
                     arrowprops=dict(arrowstyle='->', color=COLORS['plastic'], lw=1.5))
    ax_half.text(1.22, 2.8, '+95%', fontsize=9, fontweight='bold',
                 color=COLORS['plastic'], fontfamily='DejaVu Sans', va='center')
    ax_half.set_title('Defect rate: first half vs. second half of period',
                       fontsize=9, fontweight='bold', color='#2C2C2C',
                       fontfamily='DejaVu Sans', loc='left', pad=5)
    ax_half.set_ylabel('Defects per 1,000 units', fontsize=7.5,
                        color='#666666', fontfamily='DejaVu Sans')
    ax_half.set_ylim(0, 5.2)
    ax_half.set_facecolor('#F8F9FA')
    for spine in ax_half.spines.values():
        spine.set_visible(False)
    ax_half.yaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_half.set_axisbelow(True)
    ax_half.tick_params(labelsize=7.5, colors='#888888')

    # ── Chart B: returns vs complaints divergence ────────────────────────
    # Align: MONTHLY_RETURNS is May'25–May'26 (13 months)
    # MONTHLY_TICKETS is Apr'25–May'26 (14 months); skip Apr'25 to align
    ticket_vals = list(MONTHLY_TICKETS.values())[1:]   # May'25 onward (13)
    return_vals = list(MONTHLY_RETURNS.values())        # May'25–May'26 (13)
    xs = np.arange(len(ticket_vals))

    ax_diverge.plot(xs, return_vals, color=COLORS['returns_line'], linewidth=1.2,
                    label='All-cause returns', linestyle='--')
    ax_diverge.plot(xs, ticket_vals, color='#2C2C2C', linewidth=1.4,
                    label='Helpdesk complaints')
    ax_diverge.fill_between(
        xs, return_vals, ticket_vals,
        where=[r > t for r, t in zip(return_vals, ticket_vals)],
        alpha=0.07, color=COLORS['returns_line'],
    )
    ax_diverge.fill_between(
        xs, ticket_vals, return_vals,
        where=[t >= r for t, r in zip(ticket_vals, return_vals)],
        alpha=0.09, color=COLORS['plastic'],
    )
    # Show only every other label to avoid crowding
    tick_indices = xs[::2]
    tick_labels = [MONTH_LABELS_13[i] for i in tick_indices]
    ax_diverge.set_xticks(tick_indices)
    ax_diverge.set_xticklabels(tick_labels, fontsize=6.5, rotation=30, ha='right', fontfamily='DejaVu Sans')
    ax_diverge.set_title('Returns fell while complaints rose — returns were masking the signal',
                          fontsize=9, fontweight='bold', color='#2C2C2C',
                          fontfamily='DejaVu Sans', loc='left', pad=5)
    ax_diverge.set_ylabel('Count per month', fontsize=7.5,
                           color='#666666', fontfamily='DejaVu Sans')
    ax_diverge.legend(fontsize=6.5, loc='upper right',
                      framealpha=0.9, edgecolor='#DDDDDD')
    ax_diverge.set_facecolor('#F8F9FA')
    for spine in ax_diverge.spines.values():
        spine.set_visible(False)
    ax_diverge.yaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_diverge.set_axisbelow(True)
    ax_diverge.tick_params(axis='y', labelsize=7, colors='#888888')

    # ── Chart C: rework sleeve allocation by warehouse ───────────────────
    wh_names   = [wh['wh']     for wh in REWORK_WAREHOUSES]
    wh_sleeves = [wh['sleeves'] for wh in REWORK_WAREHOUSES]
    y_pos = np.arange(len(wh_names))
    bars_rw = ax_rework.barh(y_pos, wh_sleeves,
                              color=COLORS['adhesive'],
                              height=0.5, edgecolor='white', linewidth=0.5)
    ax_rework.set_yticks(y_pos)
    ax_rework.set_yticklabels(wh_names, fontsize=7.5, fontfamily='DejaVu Sans',
                               color='#2C2C2C')
    ax_rework.set_xlabel('Sleeves allocated', fontsize=7.5,
                          color='#666666', fontfamily='DejaVu Sans')
    ax_rework.set_title(
        'Rework sleeve allocation by warehouse\n'
        '(10,892 sleeves / 5,446 Tall boots — rework plan changed direction 3×)',
        fontsize=9, fontweight='bold', color='#2C2C2C',
        fontfamily='DejaVu Sans', loc='left', pad=5,
    )
    ax_rework.set_facecolor('#F8F9FA')
    for spine in ax_rework.spines.values():
        spine.set_visible(False)
    ax_rework.xaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_rework.set_axisbelow(True)
    ax_rework.tick_params(axis='x', labelsize=7, colors='#888888')
    ax_rework.set_xlim(0, 6500)
    for bar, val in zip(bars_rw, wh_sleeves):
        ax_rework.text(bar.get_width() + 60, bar.get_y() + bar.get_height() / 2,
                       f'{val:,}', va='center', ha='left', fontsize=7.5,
                       color='#2C2C2C', fontfamily='DejaVu Sans')

    draw_sidebar_callouts(ax_sidebar, SIDEBAR)
    add_footer(fig, FOOTER)
    return fig
