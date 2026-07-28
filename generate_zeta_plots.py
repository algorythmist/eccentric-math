"""Generates the grid-transformation plots of the Riemann zeta function used in ramanujan II.ipynb.

Follows the approach used in 3Blue1Brown's zeta function video (github.com/3b1b/videos, _2016/zeta.py):
take a dense grid of straight vertical and horizontal lines in the s-plane, each colored with a
gradient, and plot where zeta sends each of them in the w=zeta(s) plane. With enough lines, the result
reads as a continuously warped colored fabric rather than a handful of discrete curves.

Run this script to regenerate zeta-direct.png and zeta-continuation.png. The continuation plot uses
mpmath.zeta pointwise (as in the original source, at reduced precision for speed), which is too slow to
be worth running interactively inside the notebook, so the images are precomputed here instead.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpmath as mp

mp.mp.dps = 7  # matches the precision used in 3b1b's original source, for speed


def zeta_direct(s, n_terms=3000):
    total = np.zeros_like(s, dtype=complex)
    for n in range(1, n_terms + 1):
        total += n ** (-s)
    return total


def zeta_continued(s):
    return complex(mp.zeta(complex(s)))


def plot_dense_grid(vert_zeta, horiz_zeta, sigma_range, t_range, sigma_step, t_step,
                     epsilon_sigma, epsilon_t, n_pts, xlim, ylim, title, filename,
                     mark_zeta_minus_one=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sigmas = np.arange(sigma_range[0], sigma_range[1] + sigma_step / 2, sigma_step)
    sigmas = [s for s in sigmas if abs(s - 1) > epsilon_sigma]
    for sigma in sigmas:
        t = np.linspace(*t_range, n_pts)
        w = vert_zeta(sigma, t)
        frac = (sigma - sigma_range[0]) / (sigma_range[1] - sigma_range[0])
        color = cm.cool(0.15 + 0.7 * frac)
        ax.plot(w.real, w.imag, color=color, linewidth=0.5, alpha=0.95)

    ts = np.arange(t_range[0], t_range[1] + t_step / 2, t_step)
    ts = [t for t in ts if abs(t) > epsilon_t]
    for tv in ts:
        sigma = np.linspace(*sigma_range, n_pts)
        w = horiz_zeta(sigma, tv)
        frac = (tv - t_range[0]) / (t_range[1] - t_range[0])
        color = cm.autumn(0.05 + 0.9 * frac)
        ax.plot(w.real, w.imag, color=color, linewidth=0.5, alpha=0.95)

    if mark_zeta_minus_one:
        zm1 = zeta_continued(-1)
        ax.scatter([zm1.real], [zm1.imag], color='white', zorder=5, s=25)
        ax.annotate(r'$\zeta(-1)=-1/12$', (zm1.real, zm1.imag),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color='white')

    ax.grid(True, color='#3b5cc4', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axhline(0, color='#5b7fe0', linewidth=0.9, zorder=0)
    ax.axvline(0, color='#5b7fe0', linewidth=0.9, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\mathrm{Re}(\zeta(s))$', color='white')
    ax.set_ylabel(r'$\mathrm{Im}(\zeta(s))$', color='white')
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#3b5cc4')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(filename, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    # Direct series: only makes sense for Re(s) > 1. Fast (vectorized with numpy), so we can
    # afford a denser grid here.
    plot_dense_grid(
        vert_zeta=lambda sigma, t: zeta_direct(sigma + 1j * t),
        horiz_zeta=lambda sigma, tv: zeta_direct(sigma + 1j * tv),
        sigma_range=(1.05, 4),
        t_range=(-8, 8),
        sigma_step=0.15,
        t_step=0.25,
        epsilon_sigma=0.03,
        epsilon_t=0.05,
        n_pts=300,
        xlim=(-1, 5),
        ylim=(-3, 3),
        title=r'Image of a dense grid under $\zeta$, direct series ($\mathrm{Re}(s)>1$)',
        filename='zeta-direct.png',
    )

    # Analytic continuation: mpmath.zeta is not vectorized, so this grid must be coarser to
    # keep runtime reasonable, while still being dense enough to read as a warped fabric.
    plot_dense_grid(
        vert_zeta=lambda sigma, t: np.array([zeta_continued(sigma + 1j * tv) for tv in t]),
        horiz_zeta=lambda sigma, tv: np.array([zeta_continued(sv + 1j * tv) for sv in sigma]),
        sigma_range=(-2, 4),
        t_range=(-4, 4),
        sigma_step=0.2,
        t_step=0.2,
        epsilon_sigma=0.05,
        epsilon_t=0.05,
        n_pts=200,
        xlim=(-4, 4),
        ylim=(-4, 4),
        title=r'Image of a dense grid under $\zeta$, via analytic continuation (mpmath)',
        filename='zeta-continuation.png',
        mark_zeta_minus_one=True,
    )