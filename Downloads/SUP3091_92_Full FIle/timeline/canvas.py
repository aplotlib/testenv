import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
import os

FIG_WIDTH  = 20
FIG_HEIGHT = 8
DPI        = 300
SPINE_Y    = 0.42   # y in data coords (ax ylim 0–1); slightly below center
X_PAD_DAYS = 20     # padding in days beyond first/last event

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_canvas(events):
    """Return (fig, ax) with background, cleared spines, date x-axis, and spine line drawn."""
    dates_num = [mdates.date2num(e['date']) for e in events]
    x_min = min(dates_num) - X_PAD_DAYS
    x_max = max(dates_num) + X_PAD_DAYS

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis='x', colors='#AAAAAA', labelsize=7,
                   length=0, pad=4)

    spine_line = Line2D(
        [x_min + X_PAD_DAYS / 2, x_max - X_PAD_DAYS / 2],
        [SPINE_Y, SPINE_Y],
        color='#AAAAAA', linewidth=1.5, zorder=2,
    )
    ax.add_line(spine_line)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.10)
    return fig, ax


def save_output(fig, stem='SUP3091_3092_Timeline'):
    """Save fig as PNG (300 DPI) and PDF to OUTPUT_DIR. Returns (png_path, pdf_path)."""
    png_path = os.path.join(OUTPUT_DIR, stem + '.png')
    pdf_path = os.path.join(OUTPUT_DIR, stem + '.pdf')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='#F8F9FA')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='#F8F9FA', format='pdf')
    plt.close(fig)
    return png_path, pdf_path
