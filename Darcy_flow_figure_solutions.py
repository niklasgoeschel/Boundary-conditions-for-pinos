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


def pcolormesh_row(fig, ax_row, pred, sol, set_title=False):
    ax_row[0].set_aspect('equal')
    ax_row[1].set_aspect('equal')
    ax_row[2].set_aspect('equal')

    if set_title:
        ax_row[0].set_title('Prediction')
        ax_row[1].set_title('Exact Solution')
        ax_row[2].set_title('Error')



    vmax = np.nanmax( np.concatenate((pred,sol)) )
    vmin = np.nanmin( np.concatenate((pred,sol)) )
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    res_size = pred.shape[1]

    ax_row[0].pcolormesh(np.array([i/(res_size-1) for i in range(res_size)]), np.array([i/(res_size-1) for i in range(res_size)]), pred, cmap = 'viridis', norm=norm )
    pcm_sol = ax_row[1].pcolormesh(np.array([i/(res_size-1) for i in range(res_size)]), np.array([i/(res_size-1) for i in range(res_size)]), sol, cmap = 'viridis', norm=norm )
    
    fig.colorbar(pcm_sol, ax=ax_row[0:2], norm=norm)
    
    error_map = np.maximum(np.abs(sol-pred), 0.001)
    error_map = np.minimum(error_map, 1)
    pcm_diff = ax_row[2].pcolormesh(np.array([i/(res_size-1) for i in range(res_size)]), np.array([i/(res_size-1) for i in range(res_size)]), error_map, cmap = 'viridis', norm=colors.LogNorm(vmin=0.001, vmax=1) )
    
    fig.colorbar(pcm_diff, ax=ax_row[2])

def load_predictions_solutions(path):
    with open(path, 'rb') as f:
        np.load(f)
        np.load(f)
        all_predictions = np.load(f)
        all_solutions = np.load(f)
                
    res_size = all_predictions.shape[1]
    all_predictions[:, -int(res_size/2)::, -int(res_size/2)::] = np.nan
    all_solutions[:, -int(res_size/2)::, -int(res_size/2)::] = np.nan

    return all_predictions, all_solutions


def plot_best_worst_middle(path, save, title):
    with open(path, 'rb') as f:
        losses = np.load(f)
        errors = np.load(f)
        all_predictions = np.load(f)
        all_solutions = np.load(f)
    
    sorted_errors = np.sort(errors)
    min_error = sorted_errors[0]
    max_error = sorted_errors[-1]
    median_error = sorted_errors[49]
    i_min = np.where(errors==min_error)[0][0]
    i_max = np.where(errors==max_error)[0][0]
    i_median = np.where(errors==median_error)[0][0]
                
    res_size = all_predictions.shape[1]
    all_predictions[:, -int(res_size/2)::, -int(res_size/2)::] = np.nan
    all_solutions[:, -int(res_size/2)::, -int(res_size/2)::] = np.nan

    fig, ax = plt.subplots(3,3, layout='constrained')
    pcolormesh_row(fig, ax[0,:], all_predictions[i_min, ...], all_solutions[i_min, ...], set_title=True)
    pcolormesh_row(fig, ax[1,:], all_predictions[i_median, ...], all_solutions[i_median, ...])
    pcolormesh_row(fig, ax[2,:], all_predictions[i_max, ...], all_solutions[i_max, ...])
    fig.suptitle(title)
    fig.savefig(save, format='pdf')

plot_best_worst_middle('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_2/predicted_solutions.npy', 'boundary-conditions-for-pinos-code/figures/Darcy_flow_examples_orthogonal_projections.pdf', 'Orthogonal projections')
plot_best_worst_middle('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_2/predicted_solutions.npy', 'boundary-conditions-for-pinos-code/figures/Darcy_flow_examples_generalized_local_solution_structure.pdf', 'Generalized local solution structure')
plot_best_worst_middle('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_2/predicted_solutions.npy', 'boundary-conditions-for-pinos-code/figures/Darcy_flow_examples_semi_weak.pdf', 'Semi-weak')
plot_best_worst_middle('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_2/predicted_solutions.npy', 'boundary-conditions-for-pinos-code/figures/Darcy_flow_examples_weak.pdf', 'Weak')