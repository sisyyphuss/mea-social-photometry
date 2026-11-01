"""
Publication-ready visualization styles and figure generators.
"""

import matplotlib.pyplot as plt

COLOR_LEFT = "#1b9e77"   # Left MeA (Teal)
COLOR_RIGHT = "#7570b3"  # Right MeA (Purple)
COLOR_YOUNG = "#2c7bb6"  # Young Cohort (Blue)
COLOR_OLD = "#d7191c"    # Old Cohort (Red)


def set_publication_style():
    """Apply clean publication styling parameters to Matplotlib."""
    plt.rcParams.update({
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
