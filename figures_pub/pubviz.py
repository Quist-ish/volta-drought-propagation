"""
Distribution and uncertainty marks shared by the redesigned figures (Figs. 4, 6 and 8).

The marks follow published, evaluated designs rather than ad hoc styling:
  * raincloud plots (half density, raw observations and a summary; Allen et al., 2021, Wellcome Open Res. 4, 63),
  * Wilkinson dot plots, one dot per model or observation, binned on the value axis,
  * gradient intervals, whose opacity follows the confidence curve 1 - |2F - 1| of the estimate's
    confidence distribution (Kay, IEEE Trans. Vis. Comput. Graph., VIS 2023, ggdist), so the eye is drawn to the estimate and the
    interval fades towards its bounds instead of ending at an arbitrary cap.
No mark uses dashed or dotted lines; reference levels are solid hairlines or shaded regions.
Author: I. Quist (NUIST)
"""
import numpy as np
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde, norm


def tint(c, f):
    """Mix colour c with white; f = 0 keeps c, f = 1 gives white."""
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def dotstrip(ax, x0, values, binw, dot_w, side="center", **kw):
    """Wilkinson dot plot along the y axis at position x0: values are binned in bins of width binw and the
    dots of a bin are laid side by side with spacing dot_w (x data units). Returns the x positions."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    xs = np.empty_like(v)
    bins = np.floor(v[order] / binw)
    for b in np.unique(bins):
        idx = order[bins == b]
        n = len(idx)
        if side == "center":
            off = (np.arange(n) - (n - 1) / 2) * dot_w
        elif side == "right":
            off = np.arange(n) * dot_w
        else:
            off = -np.arange(n) * dot_w
        xs[idx] = x0 + off
    ax.scatter(xs, v, **kw)
    return xs


def half_violin(ax, x0, values, width, side="left", grid=None, bw=None, **kw):
    """One-sided kernel density of values (vertical axis) drawn at x0, scaled to a maximum half-width."""
    v = np.asarray(values, float)
    k = gaussian_kde(v, bw_method=bw)
    g = np.linspace(v.min() - 0.15 * np.ptp(v), v.max() + 0.15 * np.ptp(v), 200) if grid is None else grid
    d = k(g)
    d = d / d.max() * width
    if side == "left":
        ax.fill_betweenx(g, x0 - d, x0, **kw)
    else:
        ax.fill_betweenx(g, x0, x0 + d, **kw)
    return g, d


def _conf_curve(x, mu, s_lo, s_hi):
    """Confidence curve 1 - |2F - 1| of a split-normal confidence distribution (log scale handled by caller)."""
    z = np.where(x < mu, (x - mu) / s_lo, (x - mu) / s_hi)
    return 1 - np.abs(2 * norm.cdf(z) - 1)


def gradient_interval(ax, y, lo, mid, hi, color, height, logx=False, q=0.9, ends=None, alpha_max=0.95, zorder=2,
                      n=400):
    """Horizontal gradient interval at height y. lo, mid and hi are the q-interval bounds and the point
    estimate (e.g. q = 0.9 for 10th-90th percentiles, 0.95 for a 95% CI); opacity follows the confidence curve of a
    split normal fitted to them (on log10 if logx). The gradient is drawn between `ends` (default lo, hi)."""
    t = (lambda a: np.log10(a)) if logx else (lambda a: np.asarray(a, float))
    ti = (lambda a: 10 ** a) if logx else (lambda a: a)
    zq = norm.ppf(0.5 + q / 2)
    mu = t(mid)
    s_lo, s_hi = (mu - t(lo)) / zq, (t(hi) - mu) / zq
    a0, a1 = (t(lo), t(hi)) if ends is None else (t(ends[0]), t(ends[1]))
    xs = np.linspace(a0, a1, n)
    al = _conf_curve(xs, mu, max(s_lo, 1e-9), max(s_hi, 1e-9)) * alpha_max
    rgba = np.zeros((1, n, 4))
    rgba[0, :, :3] = to_rgb(color)
    rgba[0, :, 3] = al
    if logx:
        # imshow cannot stretch on a log axis, so draw narrow rectangles instead
        from matplotlib.patches import Rectangle
        edges = ti(np.linspace(a0, a1, n + 1))
        for i in range(n):
            ax.add_patch(Rectangle((edges[i], y - height / 2), edges[i + 1] - edges[i], height, lw=0,
                                   fc=(*to_rgb(color), al[i]), zorder=zorder))
    else:
        ax.imshow(rgba, extent=(a0, a1, y - height / 2, y + height / 2), aspect="auto", interpolation="bilinear",
                  zorder=zorder)
    return xs, al


def gradient_interval_v(ax, x, lo, mid, hi, color, width, q=0.9, alpha_max=0.95, zorder=2, n=300):
    """Vertical gradient interval at position x on a linear y axis (see gradient_interval)."""
    zq = norm.ppf(0.5 + q / 2)
    s_lo, s_hi = (mid - lo) / zq, (hi - mid) / zq
    ys = np.linspace(lo, hi, n)
    al = _conf_curve(ys, mid, max(s_lo, 1e-9), max(s_hi, 1e-9)) * alpha_max
    rgba = np.zeros((n, 1, 4))
    rgba[:, 0, :3] = to_rgb(color)
    rgba[:, 0, 3] = al
    ax.imshow(rgba[::-1], extent=(x - width / 2, x + width / 2, lo, hi), aspect="auto", interpolation="bilinear",
              zorder=zorder)


def assert_solid(fig, name=""):
    """Fail if any visible line, collection or patch edge in the figure is dashed or dotted."""
    import matplotlib.lines as ml
    import matplotlib.collections as mc
    import matplotlib.patches as mp
    bad = []
    for l in fig.findobj(ml.Line2D):
        if l.get_visible() and l.get_linewidth() > 0 and l.get_linestyle() not in ("-", "None", "", " "):
            bad.append(f"line {l.get_label()!r} {l.get_linestyle()}")
    for c in fig.findobj(mc.Collection):
        if not c.get_visible():
            continue
        lws = np.atleast_1d(c.get_linewidth())
        for (off, seq), lw in zip(c.get_linestyle(), np.resize(lws, len(c.get_linestyle()))):
            if seq is not None and lw > 0:
                bad.append(f"collection {type(c).__name__}")
                break
    for p in fig.findobj(mp.Patch):
        if p.get_visible() and p.get_linewidth() > 0 and p._dash_pattern[1] is not None \
                and p.get_edgecolor()[3] > 0:
            bad.append(f"patch {type(p).__name__}")
    if bad:
        raise ValueError(f"{name}: dashed or dotted lines are not allowed: {bad[:6]}")
