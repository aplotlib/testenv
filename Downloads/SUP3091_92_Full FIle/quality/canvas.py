import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import os

FIG_W = 8.5
FIG_H = 11.0
DPI   = 150

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def new_page():
    """Return a new 8.5×11 portrait figure with off-white background."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('#F8F9FA')
    return fig


def add_page_header(fig, title, subtitle=''):
    """Add bold page title and optional subtitle at top of figure."""
    fig.text(0.06, 0.965, title,
             fontsize=13, fontweight='bold', color='#2C2C2C',
             fontfamily='DejaVu Sans', va='top')
    if subtitle:
        fig.text(0.06, 0.950, subtitle,
                 fontsize=7.5, color='#666666',
                 fontfamily='DejaVu Sans', va='top')


def add_footer(fig, text):
    """Add thin gray rule and italic source note at bottom of figure."""
    import matplotlib.lines as mlines
    fig.add_artist(
        mlines.Line2D(
            [0.06, 0.97], [0.040, 0.040],
            transform=fig.transFigure,
            color='#DDDDDD', linewidth=0.6,
        )
    )
    fig.text(0.06, 0.034, text,
             fontsize=5.5, color='#888888', style='italic',
             fontfamily='DejaVu Sans', va='top')


def draw_sidebar_callouts(ax, callouts):
    """Draw stacked callout boxes on a cleared axes using transAxes coords.

    Each callout dict requires: accent (hex str), header (str), body (str).
    Optional: bg_alpha (float, default 0 = white background),
              header_color (hex str, default '#2C2C2C').
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    n           = len(callouts)
    gap         = 0.015
    box_height  = (0.92 - gap * (n - 1)) / n
    start_y     = 0.97
    accent_w    = 0.055
    pad         = 0.010

    for i, c in enumerate(callouts):
        y_top    = start_y - i * (box_height + gap)
        y_bot    = y_top - box_height
        bg_alpha = c.get('bg_alpha', 0)

        if bg_alpha > 0:
            border = mpatches.FancyBboxPatch(
                (0.02, y_bot), 0.96, box_height,
                boxstyle='round,pad=0.01',
                facecolor=c['accent'], alpha=bg_alpha,
                edgecolor=c['accent'], linewidth=0.8,
                transform=ax.transAxes, zorder=6, clip_on=False,
            )
        else:
            border = mpatches.FancyBboxPatch(
                (0.02, y_bot), 0.96, box_height,
                boxstyle='round,pad=0.01',
                facecolor='white', alpha=1.0,
                edgecolor='#DDDDDD', linewidth=0.5,
                transform=ax.transAxes, zorder=6, clip_on=False,
            )
        ax.add_patch(border)

        if bg_alpha == 0:
            accent_bar = mpatches.Rectangle(
                (0.02, y_bot), accent_w, box_height,
                facecolor=c['accent'], edgecolor='none',
                transform=ax.transAxes, zorder=7, clip_on=False,
            )
            ax.add_patch(accent_bar)

        hdr_color = c.get('header_color', '#2C2C2C')
        text_x    = 0.02 + (0 if bg_alpha > 0 else accent_w) + pad

        ax.text(text_x, y_top - 0.008,
                c['header'],
                ha='left', va='top',
                fontsize=6.5, fontweight='bold',
                color=hdr_color, fontfamily='DejaVu Sans',
                transform=ax.transAxes, zorder=8, clip_on=False)
        ax.text(text_x, y_top - 0.035,
                c['body'],
                ha='left', va='top',
                fontsize=5.8, color='#444444' if bg_alpha == 0 else '#FFFFFF',
                fontfamily='DejaVu Sans', linespacing=1.35,
                transform=ax.transAxes, zorder=8, clip_on=False)


def save_all(pages, stem='SUP3091_3092_Quality_Analysis'):
    """Save list of figures as multi-page PDF and individual page PNGs.
    Returns the pdf_path string.
    """
    pdf_path = os.path.join(OUTPUT_DIR, stem + '.pdf')
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(pages, 1):
            pdf.savefig(fig, bbox_inches='tight', facecolor='#F8F9FA')
            png_path = os.path.join(OUTPUT_DIR, f'{stem}_p{i}.png')
            fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
                        facecolor='#F8F9FA')
            plt.close(fig)
    return pdf_path
