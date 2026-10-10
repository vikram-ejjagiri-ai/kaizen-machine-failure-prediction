"""One consistent, readable chart style for every figure in the project."""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

# Colorblind-safe pair (validated palette): blue = normal, orange = failure
NORMAL = "#2a78d6"
FAILURE = "#eb6834"
THIRD = "#1baf7a"
NEUTRAL = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Diverging map for correlations: blue (negative) - gray (zero) - red (positive)
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8302f"]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    """Save a figure into reports/figures and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
