import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


colors = ["teal", "darkorange", "firebrick", "limegreen", "magenta", "cyan", "navy"]


def plot_B_field_profile(
    inner_S, inner_B, inner_Xpoint: int, outer_S, outer_B, outer_Xpoint: int
) -> plt.Axes:
    r"""Plot $B_{tot}$ as a function of $S_{\parallel}$ for the inner and outer divertors

    Note that $S_{\parallel}$ should go from target to midplane

    Parameters
    ----------
    inner_S : array
        $S_{\parallel}$ for inner divertor
    inner_B : array
        $B_{tot}$ for inner divertor
    inner_Xpoint : int
        Index of X-point in ``inner_S``
    outer_S : array
        $S_{\parallel}$ for outer divertor
    outer_B : array
        $B_{tot}$ for outer divertor
    outer_Xpoint : int
        Index of X-point in ``outer_S``

    Returns
    -------
    plt.Axes

    """

    fig, ax = plt.subplots()
    size = 100

    def plot_side(s, btot, xpoint, color, label):
        ax.plot(s, btot, color=color, label=label)
        ax.scatter(s[xpoint], btot[xpoint], color=color, marker="x", s=size)
        ax.scatter(s[0], btot[0], color=color, marker="o", s=size)
        ax.scatter(s[-1], btot[-1], color=color, marker="d", s=size)

    plot_side(inner_S, inner_B, inner_Xpoint, colors[0], "Inner")
    plot_side(outer_S, outer_B, outer_Xpoint, colors[1], "Outer")

    ax.set_xlabel(r"$S_{\parallel}$ (m from target)")
    ax.set_ylabel(r"$B_{tot}$ (T)")
    ax.legend()

    h, _ = ax.get_legend_handles_labels()
    kwargs = {"color": "grey", "linewidth": 0, "markersize": 10}
    extra_handles = [
        Line2D([0], [0], marker="x", label="X-point", **kwargs),
        Line2D([0], [0], marker="o", label="Target", **kwargs),
        Line2D([0], [0], marker="d", label="Midplane", **kwargs),
    ]

    ax.legend(fontsize=12, handles=h + extra_handles)

    return ax
