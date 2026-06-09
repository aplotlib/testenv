import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from quality.data import CORRECTIVE_ACTIONS, COLORS
from quality.canvas import new_page, add_page_header, add_footer, draw_sidebar_callouts

FOOTER = ('Corrective action statuses as of late May 2026. '
          'Seasonal caveat: cold-weather complaints cluster Nov–Mar; '
          'current decline does not confirm problem resolved. '
          'Validation: next cold season Nov 2026–Mar 2027.')

SIDEBAR = [
    {
        'accent': '#2E8B7A',
        'header': 'Fix validation: next cold season',
        'body':   'Current complaint decline is seasonal,\nnot proof of resolution. Real test:\nNov 2026 – Mar 2027.',
    },
    {
        'accent': '#4A7DB5',
        'header': 'Outstanding items',
        'body':   'Rework completion ETA.\nUpdated inventory counts at all WH.\nRegular WH communication cadence.',
    },
    {
        'accent': '#2E8B7A',
        'header': 'KS: proactive catch working',
        'body':   'Jason flagged return-to-inventory risk\nindependently. Team trained. Pack station\nphotos live. Upstream catch in place.',
    },
]

MILESTONES = [
    ("Nov '25",  "First QA flag",       'flag'),
    ("Jan '26",  "Rework ID'd",          'flag'),
    ("Feb '26",  "Root cause confirmed", 'test'),
    ("Apr '26",  "Rework ordered",       'decision'),
    ("May '26",  "STOP-SHIP",            'stop'),
]

STATUS_COLORS = {
    'VALIDATED':   '#2E8B7A',
    'IN PROGRESS': '#E07B39',
    'DONE':        '#2E8B7A',
}


def build_page():
    fig = new_page()
    add_page_header(fig, 'CURRENT STATE & PATH FORWARD',
                    'Stop-ship timeline, corrective action status, and what remains')
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[0.05, 0.20, 0.75],
                  left=0.08, right=0.97, top=0.92, bottom=0.07,
                  hspace=0.28, wspace=0.10)

    ax_title      = fig.add_subplot(gs[0, :])
    ax_milestones = fig.add_subplot(gs[1, 0])
    ax_table      = fig.add_subplot(gs[2, 0])
    ax_sidebar    = fig.add_subplot(gs[1:, 1])
    ax_title.axis('off')

    # ── Chart A: milestone timeline (horizontal) ─────────────────────────
    ax_milestones.set_xlim(-0.5, 4.5)
    ax_milestones.set_ylim(0, 1)
    ax_milestones.axis('off')

    dot_colors = {
        'flag':     COLORS['plastic'],
        'test':     COLORS['adhesive'],
        'decision': COLORS['bladder'],
        'stop':     COLORS['decision'],
    }
    spine_y = 0.50
    ax_milestones.axhline(spine_y, xmin=0.04, xmax=0.96,
                           color='#CCCCCC', linewidth=1.2, zorder=1)

    # Gap bracket
    ax_milestones.annotate('', xy=(4.0, 0.80), xytext=(0.0, 0.80),
                            arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=1.2))
    ax_milestones.text(2.0, 0.88, '6 months between first QA flag and stop-ship',
                       ha='center', va='bottom', fontsize=7, color='#C0392B',
                       fontfamily='DejaVu Sans', fontweight='bold')

    for i, (label_date, label_text, etype) in enumerate(MILESTONES):
        x = float(i)
        ax_milestones.scatter([x], [spine_y], s=90,
                               color=dot_colors.get(etype, '#888888'),
                               zorder=3, linewidths=0.5, edgecolors='white')
        below = (i % 2 == 0)
        y  = spine_y - 0.16 if below else spine_y + 0.08
        va = 'top'          if below else 'bottom'
        ax_milestones.text(x, y, f'{label_date}\n{label_text}',
                           ha='center', va=va, fontsize=6.5,
                           fontfamily='DejaVu Sans', color='#2C2C2C',
                           linespacing=1.2)

    ax_milestones.set_title('Stop-ship arrived 6 months after first QA flag',
                             fontsize=9, fontweight='bold', color='#2C2C2C',
                             fontfamily='DejaVu Sans', loc='left', pad=4)

    # ── Chart B: corrective actions table ────────────────────────────────
    ax_table.axis('off')
    ax_table.set_title('Corrective action status', fontsize=9, fontweight='bold',
                        color='#2C2C2C', fontfamily='DejaVu Sans', loc='left', pad=5)

    col_xs    = [0.00, 0.42, 0.64]
    col_heads = ['Action', 'Status', 'Notes']
    header_y  = 0.97
    row_height = 0.115

    for cx, head in zip(col_xs, col_heads):
        ax_table.text(cx, header_y, head,
                      transform=ax_table.transAxes,
                      fontsize=7.5, fontweight='bold', color='#2C2C2C',
                      fontfamily='DejaVu Sans', va='top')

    rule_y = header_y - 0.045
    ax_table.add_patch(mpatches.Rectangle(
        (0, rule_y), 1.0, 0.004,
        facecolor='#DDDDDD', edgecolor='none',
        transform=ax_table.transAxes,
    ))

    for i, action in enumerate(CORRECTIVE_ACTIONS):
        y   = rule_y - 0.018 - i * row_height
        bg  = '#F0FAF7' if i % 2 == 0 else 'white'
        ax_table.add_patch(mpatches.Rectangle(
            (-0.01, y - row_height * 0.72), 1.02, row_height * 0.90,
            facecolor=bg, edgecolor='none',
            transform=ax_table.transAxes, zorder=1,
        ))
        ax_table.text(col_xs[0], y, action['action'],
                      transform=ax_table.transAxes,
                      fontsize=6.8, color='#2C2C2C', fontfamily='DejaVu Sans',
                      va='top', zorder=2)
        ax_table.text(col_xs[1], y, action['status'],
                      transform=ax_table.transAxes,
                      fontsize=6.8, color=STATUS_COLORS.get(action['status'], '#888888'),
                      fontfamily='DejaVu Sans', fontweight='bold',
                      va='top', zorder=2)
        ax_table.text(col_xs[2], y, action['notes'],
                      transform=ax_table.transAxes,
                      fontsize=6.2, color='#555555', fontfamily='DejaVu Sans',
                      va='top', zorder=2)

    draw_sidebar_callouts(ax_sidebar, SIDEBAR)
    add_footer(fig, FOOTER)
    return fig
