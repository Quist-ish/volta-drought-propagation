"""
Shared publication style for all manuscript and supplementary figures.

Typography follows journal artwork guidance: a sans-serif with Arial metrics (Liberation Sans),
5-7 pt text at final size, 8 pt bold lowercase panel labels, 0.5 pt axes. Widths follow Elsevier
(single 90 mm, 1.5-column 140 mm, double 190 mm). Every figure is written as vector PDF
(TrueType fonts embedded) plus 600 dpi PNG and LZW-compressed TIFF.

Colour roles are fixed across the paper, so an entity always has the same colour:
  Ebro reference  blue  #2a78d6      Volta gauges  orange #eb6834     reanalysis  aqua #1baf7a
(the three validate as colour-blind safe under all pairings; validate_palette.js --pairs all).
Drought severity uses one validated single-hue ordinal ramp (moderate, severe, extreme).
Author: I. Quist (NUIST)
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 90 * MM, 140 * MM, 190 * MM

INK, INK2, MUTED, GRID, PAPER = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e9e9e7", "#ffffff"
EBRO, VOLTA, REAN = "#2a78d6", "#eb6834", "#1baf7a"
NAVY = "#24303f"
SEV = {"moderate": "#f0a070", "severe": "#d9592a", "extreme": "#9c3410"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("pub_blues", BLUE_RAMP)
REJECT = "#b2312b"          # used only with a cross glyph and a text label, never colour alone


def setup():
    mpl.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],      # per-glyph fallback for ✓ ✕ ⁻ etc.
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "mathtext.default": "it",
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "normal",
        "axes.titlepad": 3, "axes.labelpad": 2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": 0.9, "lines.markersize": 3, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3, "legend.columnspacing": 1.0,
        "grid.color": GRID, "grid.linewidth": 0.4,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.cmap": "pub_blues",
    })
    if "pub_blues" not in plt.colormaps():
        plt.colormaps.register(BLUES)


def panel(ax, s, x=-0.02, y=1.0, fig=None, **kw):
    """Bold lowercase panel label placed at the top-left of an axes, outside the frame."""
    t = ax if fig is None else fig
    tr = ax.transAxes if fig is None else fig.transFigure
    t.text(x, y, s, transform=tr, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK, **kw)


def hgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


MIN_PT = 6.0   # Elsevier artwork guidance: lettering no smaller than 6 pt at final size


def enforce_min_text(fig, min_pt=MIN_PT):
    """Raise any text below min_pt (tick labels, annotations, legends) to min_pt before export."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        try:
            if t.get_text() and t.get_fontsize() < min_pt:
                t.set_fontsize(min_pt)
        except Exception:
            pass


def save(fig, name, min_pt=MIN_PT):
    """Vector PDF (TrueType embedded), 600 dpi PNG for previews and 1000 dpi LZW TIFF for production.
    Dashed and dotted lines are not used anywhere in the paper; the export refuses a figure that has one."""
    enforce_min_text(fig, min_pt)
    from pubviz import assert_solid
    assert_solid(fig, name)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 1000, "pil_kwargs": {"compression": "tiff_lzw"}})):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, facecolor=PAPER, **kw)
        paths.append(p)
    plt.close(fig)
    return paths
