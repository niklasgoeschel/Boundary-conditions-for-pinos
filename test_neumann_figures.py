import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True
})

from matplotlib import cm, colors

with open('boundary-conditions-for-pinos-code/results_test_neumann/single_distance_function.npy', 'rb') as f:
    test_neumann_single_distance_function = np.load(f)

with open('boundary-conditions-for-pinos-code/results_test_neumann/multiple_distance_functions.npy', 'rb') as f:
    test_neumann_multiple_distance_functions = np.load(f)

with open('boundary-conditions-for-pinos-code/results_test_neumann/our_approach.npy', 'rb') as f:
    test_neumann_our_approach = np.load(f)

res_size = 201

x = np.array([[(i)/(res_size-1) for j in range(res_size)] for i in range(res_size)] )
y = np.array([[(j)/(res_size-1) for j in range(res_size)] for i in range(res_size)] )

test_neumann_reference = np.cos(np.pi*x) * np.cos(np.pi*y)


def plot_test_neumann(pred, path):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x[1:-1,1:-1], y[1:-1,1:-1], pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y)$')
    fig.savefig(path, format='pdf')

plot_test_neumann(test_neumann_reference[1:-1,1:-1], 'boundary-conditions-for-pinos-code/figures/test_neumann_analytic.pdf')
plot_test_neumann(test_neumann_single_distance_function, 'boundary-conditions-for-pinos-code/figures/test_neumann_single_distance_function.pdf')
plot_test_neumann(test_neumann_multiple_distance_functions, 'boundary-conditions-for-pinos-code/figures/test_neumann_multiple_distance_functions.pdf')
plot_test_neumann(test_neumann_our_approach, 'boundary-conditions-for-pinos-code/figures/test_neumann_our_approach.pdf')