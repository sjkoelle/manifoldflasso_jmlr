import seaborn as sns

def plot_cosines(cosines, ax, colors):
    p = cosines.shape[0]
    sns.heatmap(cosines, ax=ax, vmin=0., vmax=1.)
    #    ax = sns.heatmap(x, cmap=cmap)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=500 / p)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=500 / p)

    ax.set_ylabel(r"$g_{j'}$", fontsize=70)
    ax.set_xlabel(r"$g_{j}$", fontsize=70)
    # ax.set_title(r"$\text{hi}$")
    ax.set_title(
        r"$\frac{1}{n'} \sum_{i = 1}^{n'} \frac{ |\langle grad_{\mathcal M} g_j (\xi_i) ,grad_{\mathcal M} g_{j'} (\xi_i)\rangle|}{\|grad_{\mathcal M} g_j (\xi_i) \| \| grad_{\mathcal M} g_{j'}(\xi_i) \|}$",
        fontsize=70)