import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import math
rcParams['figure.figsize'] = 25, 10

def width(p,w):
    if p > 1.:
        output = 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
    else:
        output = w
    return(output)

def plot_reg_path_ax_lambdasearch_customcolors(axes, coeffs, xaxis, fig, colors):
    p = coeffs.shape[3]
    q = coeffs.shape[1]
    gnames = np.asarray(list(range(p)), dtype=str)

    # xlabel = r"$\displaystyle \lambda$"
    # ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)

    # maxes = np.zeros(q)
    # for k in range(q):
    #     maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    # normax = maxes.max()
    normax = np.sqrt(np.sum(np.sum(np.sum(coeffs ** 2, axis=1), axis=1), axis=1).max())

    for k in range(q):
        for j in range(p):
            toplot = np.linalg.norm(coeffs[:, k, :, j], axis=1)
            w = .15
            widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
            # axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[k + 1].plot(xaxis, toplot, 'go--', linewidth=10, markersize=0, alpha=1.,
                             color=colors[j], label=gnames[j])
    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        axes[0].plot(xaxis, toplot, 'go--', linewidth=10, markersize=0, alpha=.5,
                     color=colors[j], label=gnames[j])

    kkk = xaxis.copy()
    kkk.sort()

    # xupperindex = np.min(np.where(np.sum(np.sum(np.sum(coeffs**2, axis = 1), axis = 1), axis = 1) ==0)[0])

    for k in range(1 + q):
        axes[k].tick_params(labelsize=50)
        #axes[k].set_xscale('symlog')
        axes[k].set_yscale('symlog')
        axes[k].set_ylim(bottom=0, top=normax)
        # axes[k].set_xlim(left = 0, right = xaxis[xupperindex])
        if (k == 0):
            tixx = np.hstack(
                [np.asarray([0]), 10 ** np.linspace(math.floor(np.log10(normax)), math.floor(np.log10(normax)) + 1, 2)])
        if k != 0:
            # axes[k].set_yticks(tixx)
            axes[k].set_yticklabels([])
        if k != q:
            axes[k + 1].set_title(r"$\phi_{{{}}}$".format(k + 1), fontsize=50)
            # axes[k + 1].set_title(r"$\phi_{{{}}}$.format(k)")
        if k == 0:
            axes[k].set_title("Combined", fontdict={'fontsize': 50})
    for k in range(1 + q):
        axes[k].grid(True, which="both", alpha=True)
        axes[k].set_xlabel(r"$\lambda$", fontsize=50)
        axes[k].set_xticklabels([])
        axes[k].set_xticks([])

    axes[0].set_ylabel(r"$||\beta_j||$", fontsize=50)


def plot_watch_custom(to_plot, p, ax, colors,nreps):

    # fig, ax = plt.subplots(figsize = (15,15))
    # %matplotlib inline

    # fig, ax = plt.subplots(figsize = (15,15))
    theta = np.linspace(0, 2 * np.pi, 10000)
    cmap = plt.get_cmap('twilight_shifted', p)

    angles = np.linspace(0, 2 * np.pi, p + 1)

    radius = 1.

    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    # figure, axes = plt.subplots(figsize = (15,15))

    # axes.plot(a, b, color= 'gray')
    ax.scatter(a, b, color='gray', s=.2,
               alpha=.1)  # , '-', color = 'gray')#, s= .1, alpha = .1)#, type = 'line')#,cmap=plt.get_cmap('twilight')) #'hsv','twilight_shifted

    # for i in range(to_plot.shape)
    if len(to_plot.shape) > 1:
        totes = np.sum(to_plot, axis=0)
    else:
        totes = to_plot

    for j in range(p):
        print(np.cos(angles[j]), np.sin(angles[j]))  # r'$test \frac{1}{}$'.format(g)
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=cmap.colors[j], marker='x')
        ax.text(x=1.1 * np.cos(angles[j]),
                y=1.1 * np.sin(angles[j]),
                s=r"$g_{{{}}}$".format(j), color=colors[j],  # cmap.colors[j],
                fontdict={'fontsize': 70},
                horizontalalignment='center',
                verticalalignment='center')

        ax.text(x=.9 * np.cos(angles[j]), y=.9 * np.sin(angles[j]), s=str(totes[j] / nreps), fontdict={'fontsize': 40},
                horizontalalignment='center',
                verticalalignment='center')

    for j in range(p):
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='o', s=200 * totes[j])

    if len(to_plot.shape) > 1:
        for i in range(p):
            for j in range(p):

                # point1 = [1, 2]
                # point2 = [3, 4]

                x_values = [np.cos(angles[j]), np.cos(angles[i])]
                # gather x-values

                y_values = [np.sin(angles[j]), np.sin(angles[i])]
                # gather y-values

                ax.plot(x_values, y_values, linewidth=to_plot[i, j], color='black')

                if to_plot[i, j] > 0:
                    ax.text(x=np.mean(x_values),
                            y=np.mean(y_values),
                            s=str(to_plot[i, j] / nreps),
                            fontdict={'fontsize': 40})  # ,
                # horizontalalignment='left',
                # verticalalignment='bottom')

                # axes.axline((x1, y1), (x2, y2))
    ax.set_aspect(1)
    ax.set_axis_off()
    ax.set_title(r"$\omega = 25$")


def plot_reg_path_ax_lambdasearch_customcolors_norm(ax, coeffs, xaxis, fig, colors):
    p = coeffs.shape[3]
    q = coeffs.shape[1]
    gnames = np.asarray(list(range(p)), dtype=str)

    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)

    normax = np.sqrt(np.sum(np.sum(np.sum(coeffs ** 2, axis=1), axis=1), axis=1).max())

    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        ax.plot(xaxis, toplot, 'go--', linewidth=5, markersize=0, alpha=1.,
                color=colors[j], label=gnames[j])

    kkk = xaxis.copy()
    kkk.sort()

    # xupperindex = np.min(np.where(np.sum(np.sum(np.sum(coeffs**2, axis = 1), axis = 1), axis = 1) ==0)[0])

    # for k in range(1 + q):
    ax.tick_params(labelsize=50)
    # ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_ylim(bottom=0, top=normax)
    # axes[k].set_xlim(left = 0, right = xaxis[xupperindex])
    # if (k == 0):
    tixx = np.hstack(
        [np.asarray([0]), 10 ** np.linspace(math.floor(np.log10(normax)), math.floor(np.log10(normax)) + 1, 2)])
    #    if k != 0:
    # axes[k].set_yticks(tixx)
    # ax.set_ylabel(r"$\displaystyle \|\hat \beta_{j}\|_2$", fontsize = 70)
    # ax.set_xlabel(r"$\lambda  \sqrt{nm}$", fontsize = 70)
    # ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    # ax.l
    # if k == 0:
    # ax.set_title("Combined", fontdict={'fontsize': 50})
    ax.grid(True, which="both", alpha=True)




def plot_watch3(to_plot, names, colors, ax,nreps):

    p = to_plot.shape[0]
    # fig, ax = plt.subplots(figsize = (15,15))
    # %matplotlib inline

    # fig, ax = plt.subplots(figsize = (15,15))
    theta = np.linspace(0, 2 * np.pi, 10000)
    # cmap = plt.get_cmap('twilight_shifted',p)

    angles = np.linspace(0, 2 * np.pi, p + 1)

    radius = 1.

    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    # figure, axes = plt.subplots(figsize = (15,15))

    # axes.plot(a, b, color= 'gray')
    ax.scatter(a, b, color='gray', s=.2,
               alpha=.1)  # , '-', color = 'gray')#, s= .1, alpha = .1)#, type = 'line')#,cmap=plt.get_cmap('twilight')) #'hsv','twilight_shifted

    # for i in range(to_plot.shape)
    if len(to_plot.shape) > 1:
        totes = np.sum(to_plot, axis=0)
    else:
        totes = to_plot

    for j in range(p):
        print(np.cos(angles[j]), np.sin(angles[j]))  # r'$test \frac{1}{}$'.format(g)
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='x')
        ax.text(x=1.1 * np.cos(angles[j]),
                y=1.1 * np.sin(angles[j]),
                s=names[j], color=colors[j],
                fontdict={'fontsize': 40},
                horizontalalignment='center',
                verticalalignment='center')

        ax.text(x=.9 * np.cos(angles[j]), y=.9 * np.sin(angles[j]), s=str(totes[j] / nreps), fontdict={'fontsize': 30},
                horizontalalignment='center',
                verticalalignment='center')

    for j in range(p):
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='o', s=100 * totes[j])

    if len(to_plot.shape) > 1:
        for i in range(p):
            for j in range(p):

                # point1 = [1, 2]
                # point2 = [3, 4]

                x_values = [np.cos(angles[j]), np.cos(angles[i])]
                # gather x-values

                y_values = [np.sin(angles[j]), np.sin(angles[i])]
                # gather y-values

                ax.plot(x_values, y_values, linewidth=to_plot[i, j], color='black')

                if to_plot[i, j] > 0:
                    ax.text(x=np.mean(x_values),
                            y=np.mean(y_values),
                            s=str(to_plot[i, j] / nreps),
                            fontdict={'fontsize': 20})  # ,
                # horizontalalignment='left',
                # verticalalignment='bottom')
                # axes.axline((x1, y1), (x2, y2))
    ax.set_aspect(1)
    ax.set_axis_off()



