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



fig = plt.figure(figsize=(5.5,4.5))
gs = fig.add_gridspec(4,1, hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row')

def plot_errors_losses_navier_stokes(axes, path, label_name, color):
    with open(path, 'rb') as f:
        losses = np.load(f)
        errors_u = np.load(f)
        errors_v = np.load(f)
        errors_p = np.load(f)

    axes[1].plot(np.array([i for i in range(4000)]), errors_u, color=color, label=label_name)
    axes[2].plot(np.array([i for i in range(4000)]), errors_v, color=color, label=label_name)
    axes[3].plot(np.array([i for i in range(4000)]), errors_p, color=color, label=label_name)
    axes[0].plot(np.array([i for i in range(4000)]), losses, color=color, label=label_name)

plot_errors_losses_navier_stokes(axes, 'boundary-conditions-for-pinos-code/results_Navier_stokes/generalized_local_solution_structure/losses_errors.npy', 'GLSS', 'blue')
plot_errors_losses_navier_stokes(axes, 'boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/losses_errors.npy', 'OP', 'red')
plot_errors_losses_navier_stokes(axes, 'boundary-conditions-for-pinos-code/results_Navier_stokes/semi_weak/losses_errors.npy', 'Semi-weak', 'orange')
plot_errors_losses_navier_stokes(axes, 'boundary-conditions-for-pinos-code/results_Navier_stokes/weak/losses_errors.npy', 'Weak', 'green')

axes[0].legend(loc="upper right")

axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')
axes[3].set_yscale('log')

axes[1].set_ylabel('error $u$')
axes[2].set_ylabel('error $v$')
axes[3].set_ylabel('error $p$')
axes[0].set_ylabel('loss')

axes[3].set_xlabel('Epochs')

axes[0].grid(True, which="both", ls="-", color='0.65')
axes[1].grid(True, which="both", ls="-", color='0.65')
axes[2].grid(True, which="both", ls="-", color='0.65')
axes[3].grid(True, which="both", ls="-", color='0.65')

fig.savefig('boundary-conditions-for-pinos-code/figures/Navier_stokes_errors_losses.pdf', format='pdf')