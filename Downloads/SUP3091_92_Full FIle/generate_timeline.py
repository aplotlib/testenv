import matplotlib
matplotlib.use('Agg')

from timeline.data import EVENTS, PHASES, COLORS, MARKERS
from timeline.canvas import setup_canvas, save_output
from timeline.phases import draw_phase_bands
from timeline.events import draw_events
from timeline.callouts import draw_callouts, draw_legend


def main():
    fig, ax = setup_canvas(EVENTS)
    draw_phase_bands(ax, EVENTS, PHASES)
    draw_events(ax, EVENTS, COLORS, MARKERS)
    draw_callouts(ax)
    draw_legend(ax, COLORS, MARKERS)
    png_path, pdf_path = save_output(fig)
    print(f"Saved PNG : {png_path}")
    print(f"Saved PDF : {pdf_path}")


if __name__ == '__main__':
    main()
