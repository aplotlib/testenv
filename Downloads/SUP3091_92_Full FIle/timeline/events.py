import matplotlib.dates as mdates

# y positions (data coords, ylim 0–1)
SPINE_Y       = 0.42   # NOTE: canvas.py defines its own SPINE_Y = 0.42, keep in sync
MARKER_ABOVE  = 0.56
MARKER_BELOW  = 0.28
LABEL_ABOVE   = 0.77
LABEL_BELOW   = 0.05
NOTE_ABOVE    = 0.63
NOTE_BELOW    = 0.21
MARKER_SIZE   = 130
BADGE_SIZE    = 45
TICK_LW       = 0.8
LABEL_ROT     = 32   # degrees; positive = counterclockwise from horizontal


def draw_events(ax, events, colors, markers):
    """Plot type-coded event markers, connecting ticks, rotated labels, and notes.

    Events are sorted by date before rendering so the above/below stagger
    (even index above, odd below) always matches the left-to-right visual order.
    """
    sorted_events = sorted(events, key=lambda e: e['date'])

    for i, event in enumerate(sorted_events):
        x = mdates.date2num(event['date'])
        above = (i % 2 == 0)

        marker_y = MARKER_ABOVE if above else MARKER_BELOW
        label_y  = LABEL_ABOVE  if above else LABEL_BELOW
        note_y   = NOTE_ABOVE   if above else NOTE_BELOW
        rot      = LABEL_ROT    if above else -LABEL_ROT
        va_label = 'bottom'     if above else 'top'
        va_note  = 'bottom'     if above else 'top'

        color  = colors[event['type']]
        shape  = markers[event['type']]

        # Tick line from spine to marker
        ax.plot([x, x], [SPINE_Y, marker_y],
                color='#AAAAAA', linewidth=TICK_LW, zorder=2)

        # Event marker
        ax.scatter([x], [marker_y],
                   s=MARKER_SIZE, marker=shape,
                   color=color, zorder=4,
                   linewidths=0.8, edgecolors='white')

        # Positive-action star badge (overlaid)
        if event.get('positive'):
            ax.scatter([x], [marker_y],
                       s=BADGE_SIZE, marker='*',
                       color=colors['positive_badge'], zorder=5)

        # Primary label
        ax.text(x, label_y, event['label'],
                ha='left', va=va_label,
                fontsize=7.5, color='#2C2C2C',
                fontfamily='DejaVu Sans',
                rotation=rot,
                rotation_mode='anchor',
                zorder=5,
                linespacing=1.3)

        # One-line note (italic, only for key events)
        if event.get('note'):
            ax.text(x, note_y, event['note'],
                    ha='left', va=va_note,
                    fontsize=6.5, color='#666666',
                    fontfamily='DejaVu Sans',
                    style='italic',
                    rotation=rot,
                    rotation_mode='anchor',
                    zorder=5)
