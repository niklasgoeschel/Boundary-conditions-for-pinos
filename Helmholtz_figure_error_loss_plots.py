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

with open('boundary-conditions-for-pinos-code/results_Helmholtz/generalized_local_solution_structure/losses_errors.npy', 'rb') as f:
    losses_glss = np.load(f)
    errors_glss = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Helmholtz/orthogonal_projections/losses_errors.npy', 'rb') as f:
    losses_op = np.load(f)
    errors_op = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Helmholtz/semi_weak/losses_errors.npy', 'rb') as f:
    losses_semi_weak = np.load(f)
    errors_semi_weak = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Helmholtz/weak/losses_errors.npy', 'rb') as f:
    losses_weak = np.load(f)
    errors_weak = np.load(f)

fig = plt.figure(figsize=(5.5,4.5))
gs = fig.add_gridspec(2,1, hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row')


axes[0].plot(np.array([i for i in range(2000)]), losses_glss, color='blue', label='GLSS')
axes[1].plot(np.array([i for i in range(2000)]), errors_glss, color='blue', label='GLSS')

axes[0].plot(np.array([i for i in range(2000)]), losses_op, color='red', label='OP')
axes[1].plot(np.array([i for i in range(2000)]), errors_op, color='red', label='OP')

axes[0].plot(np.array([i for i in range(2000)]), losses_semi_weak, color='orange', label='Semi-weak')
axes[1].plot(np.array([i for i in range(2000)]), errors_semi_weak, color='orange', label='Semi-weak')

axes[0].plot(np.array([i for i in range(2000)]), losses_weak, color='green', label='Weak')
axes[1].plot(np.array([i for i in range(2000)]), errors_weak, color='green', label='Weak')


axes[0].legend(loc="upper right")

axes[0].set_yscale('log')
axes[1].set_yscale('log')

axes[1].set_ylabel('error')
axes[0].set_ylabel('loss')

axes[1].set_xlabel('Epochs')

axes[0].grid(True, which="both", ls="-", color='0.65')
axes[1].grid(True, which="both", ls="-", color='0.65')


fig.savefig('boundary-conditions-for-pinos-code/figures/Helmholtz_errors_losses.pdf', format='pdf')