import matplotlib.patches as mpatches
import matplotlib.lines as mlines


CALLOUTS = [
    {
        'accent': '#E07B39',
        'header': 'Not catastrophic — strategic risk.',
        'body':   'SUP3091 is ~62% of channel; low defect\nrate carries outsized brand exposure.',
    },
    {
        'accent': '#4A7DB5',
        'header': 'Recent decline is likely seasonal.',
        'body':   'Cold-weather complaints fall every spring.\nReal validation is next cold season.',
    },
    {
        'accent': '#2E8B7A',
        'header': 'Containment in progress.',
        'body':   'JX gap corrected; KS proactive catches\nin place; unchecked stock pulled from sale.',
    },
]

# Axes-fraction coordinates (transAxes)
BOX_X       = 0.695
BOX_WIDTH   = 0.295
BOX_HEIGHT  = 0.115
BOX_Y_START = 0.345   # top of the first (topmost) box
BOX_GAP     = 0.012
ACCENT_W    = 0.006


def draw_callouts(ax):
    """Draw three framing callout boxes in the lower-right quadrant."""
    trans = ax.transAxes

    for i, callout in enumerate(CALLOUTS):
        y_top = BOX_Y_START - i * (BOX_HEIGHT + BOX_GAP)
        y_bot = y_top - BOX_HEIGHT

        # Border rectangle
        border = mpatches.FancyBboxPatch(
            (BOX_X, y_bot), BOX_WIDTH, BOX_HEIGHT,
            boxstyle='round,pad=0.008',
            facecolor='white', edgecolor='#DDDDDD',
            linewidth=0.5, zorder=6,
            transform=trans, clip_on=False,
        )
        ax.add_patch(border)

        # Accent bar
        accent = mpatches.Rectangle(
            (BOX_X, y_bot), ACCENT_W, BOX_HEIGHT,
            facecolor=callout['accent'], edgecolor='none',
            zorder=7,
            transform=trans, clip_on=False,
        )
        ax.add_patch(accent)

        text_x = BOX_X + ACCENT_W + 0.008

        # Header
        ax.text(text_x, y_top - 0.010,
                callout['header'],
                ha='left', va='top',
                fontsize=6.8, fontweight='bold',
                color='#2C2C2C', fontfamily='DejaVu Sans',
                transform=trans, zorder=8, clip_on=False)

        # Body
        ax.text(text_x, y_top - 0.036,
                callout['body'],
                ha='left', va='top',
                fontsize=6.2, color='#555555',
                fontfamily='DejaVu Sans',
                linespacing=1.35,
                transform=trans, zorder=8, clip_on=False)


def draw_legend(ax, colors, markers):
    """Draw event-type legend in the lower-left corner."""
    items = [
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=colors['detection'],   markersize=7,
                      label='Detection / signal'),
        mlines.Line2D([], [], marker='D', color='w',
                      markerfacecolor=colors['rca'],         markersize=7,
                      label='Investigation / RCA'),
        mlines.Line2D([], [], marker='s', color='w',
                      markerfacecolor=colors['decision'],    markersize=7,
                      label='Decision'),
        mlines.Line2D([], [], marker='h', color='w',
                      markerfacecolor=colors['containment'], markersize=7,
                      label='Containment / corrective action'),
        mlines.Line2D([], [], marker='*', color='w',
                      markerfacecolor='#27AE60',             markersize=9,
                      label='Positive team action'),
    ]
    legend = ax.legend(
        handles=items,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        fontsize=6.5,
        framealpha=0.92,
        edgecolor='#DDDDDD',
        fancybox=True,
        handlelength=1.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.5)
