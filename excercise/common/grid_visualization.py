import numpy as np
import matplotlib.pyplot as plt

threshold = -1.0


def visualize_value_function(ax,  # matplotlib axes object
                             v_pi: np.array,
                             nx: int,
                             ny: int,
                             plot_cbar=True):
    hmap = ax.imshow(v_pi.reshape(nx, ny),
                     interpolation='nearest')
    if plot_cbar:
        cbar = ax.figure.colorbar(hmap, ax=ax)

    # disable x,y ticks for better visibility
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    # annotate value of value functions on heat map
    for i in range(ny):
        for j in range(nx):
            cell_v = v_pi.reshape(nx, ny)[j, i]
            text_color = "w" if cell_v < threshold else "black"
            cell_v = "{:.2f}".format(cell_v)
            ax.text(i, j, cell_v, ha="center", va="center", color="w")


def visualize_policy(ax, pi: np.array, nx: int, ny: int):
    d_symbols = ['↑', '→', '↓', '←']
    pi = np.array(list(map(np.argmax, pi))).reshape(nx, ny)

    ax.imshow(pi, interpolation='nearest', cmap=plt.get_cmap('Paired'))

    ax.set_xticks(np.arange(pi.shape[1]))
    ax.set_yticks(np.arange(pi.shape[0]))
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(pi.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(pi.shape[0] + 1) - .5, minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(ny):
        for j in range(nx):
            direction = pi[j, i]
            direction = d_symbols[direction]
            ax.text(i, j, direction, ha="center", va="center", color="black", fontsize=20)
