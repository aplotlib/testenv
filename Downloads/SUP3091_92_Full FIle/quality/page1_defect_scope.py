import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from quality.data import FAILURE_MODES, VARIANT_RATES, COLORS
from quality.canvas import new_page, add_page_header, add_footer, draw_sidebar_callouts

FOOTER = ('Defect counts: curated plastic/adhesive log + quality-attributed helpdesk tickets. '
          'Conservative floor 132 — realistic estimate 159. '
          'Denominator: 46,573 B2B units shipped Jun 2025–May 2026.')

SIDEBAR = [
    {
        'accent': '#E07B39',
        'header': '1 in 291 boots confirmed defective (0.34%)',
        'body':   '~159 quality failures across 46,573 units\nshipped Jun 2025 – May 2026.',
    },
    {
        'accent': '#4A7DB5',
        'header': 'SUP3091 Tall: the outlier',
        'body':   '6.6 defects per 1,000 — 4× the Short rate\nand 26× SUP3092 Short.',
    },
    {
        'accent': '#7B5EA7',
        'header': 'SUP3091 = 62% of channel volume',
        'body':   'A sub-1% rate concentrated in the dominant\nproduct creates outsized brand exposure.',
    },
]


def build_page():
    fig = new_page()
    add_page_header(fig, 'THE DEFECT PICTURE',
                    'SUP3091 / SUP3092 Coretech Walker Boot — Quality Case Overview')
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[0.06, 0.47, 0.47],
                  left=0.10, right=0.97, top=0.92, bottom=0.07,
                  hspace=0.42, wspace=0.10)

    ax_title   = fig.add_subplot(gs[0, :])
    ax_chart_a = fig.add_subplot(gs[1, 0])
    ax_chart_b = fig.add_subplot(gs[2, 0])
    ax_sidebar = fig.add_subplot(gs[1:, 1])
    ax_title.axis('off')

    # ── Chart A: failure mode counts (horizontal bar) ─────────────────────
    names  = [m['name']  for m in FAILURE_MODES]
    counts = [m['count'] for m in FAILURE_MODES]
    colors = [m['color'] for m in FAILURE_MODES]
    y_pos  = np.arange(len(names))

    bars = ax_chart_a.barh(y_pos, counts, color=colors,
                            height=0.55, edgecolor='white', linewidth=0.5)
    ax_chart_a.set_yticks(y_pos)
    ax_chart_a.set_yticklabels(names, fontsize=7.5, fontfamily='DejaVu Sans',
                                color='#2C2C2C')
    ax_chart_a.set_xlabel('Confirmed defects (realistic count)', fontsize=7.5,
                           color='#666666', fontfamily='DejaVu Sans')
    ax_chart_a.set_title('Defect count by failure mode', fontsize=9,
                          fontweight='bold', color='#2C2C2C',
                          fontfamily='DejaVu Sans', loc='left', pad=6)
    ax_chart_a.set_xlim(0, 52)
    ax_chart_a.tick_params(axis='x', labelsize=7, colors='#888888')
    ax_chart_a.set_facecolor('#F8F9FA')
    for spine in ax_chart_a.spines.values():
        spine.set_visible(False)
    ax_chart_a.xaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_chart_a.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax_chart_a.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                        str(count), va='center', ha='left', fontsize=7.5,
                        color='#2C2C2C', fontfamily='DejaVu Sans')

    # ── Chart B: defect rates by variant — grouped bars (plastic + adhesive) ─
    variants       = [v['variant']       for v in VARIANT_RATES]
    plastic_rates  = [v['plastic_rate']  for v in VARIANT_RATES]
    adhesive_rates = [v['adhesive_rate'] for v in VARIANT_RATES]
    total_rates    = [v['total_rate']    for v in VARIANT_RATES]
    y_pos2 = np.arange(len(variants))
    bar_h  = 0.28

    ax_chart_b.barh(y_pos2 + bar_h / 2, plastic_rates, height=bar_h,
                    color=COLORS['plastic'], label='Plastic', edgecolor='white', linewidth=0.3)
    ax_chart_b.barh(y_pos2 - bar_h / 2, adhesive_rates, height=bar_h,
                    color=COLORS['adhesive'], label='Adhesive/Velcro', edgecolor='white', linewidth=0.3)

    # Total rate marker — only for variants where total_rate is not None
    for yi, tr in zip(y_pos2, total_rates):
        if tr is not None:
            ax_chart_b.scatter([tr], [yi], marker='|', s=120,
                               color='#2C2C2C', zorder=5, linewidths=1.5)
    # Dummy handle for legend
    import matplotlib.lines as mlines
    total_handle = mlines.Line2D([], [], marker='|', color='#2C2C2C',
                                  linewidth=1.5, markersize=8, label='Total (all modes)')

    ax_chart_b.set_yticks(y_pos2)
    ax_chart_b.set_yticklabels(variants, fontsize=7.5, fontfamily='DejaVu Sans',
                                color='#2C2C2C')
    ax_chart_b.set_xlabel('Rate per 1,000 units sold', fontsize=7.5,
                           color='#666666', fontfamily='DejaVu Sans')
    ax_chart_b.set_title('Defect rate per 1,000 units sold by variant',
                          fontsize=9, fontweight='bold', color='#2C2C2C',
                          fontfamily='DejaVu Sans', loc='left', pad=6)
    ax_chart_b.set_facecolor('#F8F9FA')
    for spine in ax_chart_b.spines.values():
        spine.set_visible(False)
    ax_chart_b.xaxis.grid(True, color='#DDDDDD', linewidth=0.5)
    ax_chart_b.set_axisbelow(True)
    ax_chart_b.tick_params(axis='x', labelsize=7, colors='#888888')
    ax_chart_b.set_xlim(0, 8.5)
    # Add legend with explicit handles
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLORS['plastic'],  label='Plastic'),
        Patch(facecolor=COLORS['adhesive'], label='Adhesive/Velcro'),
        total_handle,
    ]
    ax_chart_b.legend(handles=handles, fontsize=6.5, loc='lower right',
                      framealpha=0.9, edgecolor='#DDDDDD')

    draw_sidebar_callouts(ax_sidebar, SIDEBAR)
    add_footer(fig, FOOTER)
    return fig
