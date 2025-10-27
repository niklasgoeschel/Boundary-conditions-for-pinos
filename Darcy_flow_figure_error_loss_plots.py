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



def plot_operator_training_and_finetune_error(axes, path_operator, checkpoints_operator, path_finetune, path_pinnlike, label_name, color):
    for i in range(checkpoints_operator):
        with open(path_operator + '/ckp_' + str(int(i+1)) + '/losses_errors.npy', 'rb') as f:
            epochs = np.load(f)
            losses = np.load(f)
            errors = np.load(f)
        axes[1,1].plot(epochs, errors, color=color)
        axes[0,1].plot(epochs, losses, color=color)

    losses = np.zeros((1000,100))
    errors = np.zeros((1000,100))
    with open(path_finetune, 'rb') as f:
        for i in range(100):
            losses[:,i] = np.load(f)
            errors[:,i] = np.load(f)

    axes[1,2].plot(np.array([i for i in range(1000)]), np.mean(errors, 1), color=color, label=label_name)
    axes[1,2].plot(np.array([i for i in range(1000)]), np.ones((1000))*np.mean(errors, 1)[0], '--', color=color, alpha=0.5)

    axes[0,2].plot(np.array([i for i in range(1000)]), np.mean(losses, 1), color=color, label=label_name)    
    axes[0,2].plot(np.array([i for i in range(1000)]), np.ones((1000))*np.mean(losses, 1)[0], '--', color=color, alpha=0.5)

    losses = np.zeros((1000,100))
    errors = np.zeros((1000,100))
    with open(path_pinnlike, 'rb') as f:
        for i in range(100):
            losses[:,i] = np.load(f)
            errors[:,i] = np.load(f)

    axes[1,0].plot(np.array([i for i in range(1000)]), np.mean(errors, 1), color=color, label=label_name)

    axes[0,0].plot(np.array([i for i in range(1000)]), np.mean(losses, 1), color=color, label=label_name)    



fig = plt.figure(figsize=(8,5))
gs = fig.add_gridspec(2,3, hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row')

plot_operator_training_and_finetune_error(axes, 'boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure', 2, 'boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/generalized_local_solution_structure.npy', 'boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/generalized_local_solution_structure.npy', 'GLSS', 'blue')
plot_operator_training_and_finetune_error(axes, 'boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections', 2, 'boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/orthogonal_projections.npy', 'boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/orthogonal_projections.npy', 'OP', 'red')
plot_operator_training_and_finetune_error(axes, 'boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak', 2, 'boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/semi_weak.npy', 'boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/semi_weak.npy', 'Semi-weak', 'orange')
plot_operator_training_and_finetune_error(axes, 'boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak', 2, 'boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/weak.npy', 'boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/weak.npy', 'Weak', 'green')



axes[0,2].legend(loc="upper right")

axes[0,0].set_yscale('log')
axes[1,0].set_yscale('log')

axes[0,1].spines['left'].set_linewidth(4)
axes[1,1].spines['left'].set_linewidth(4)

axes[0,0].set_ylabel('loss')
axes[1,0].set_ylabel('Relative $L^2$-error')

axes[1,0].set_xlabel('Epochs')
axes[1,1].set_xlabel('Epochs')
axes[1,2].set_xlabel('Epochs')

axes[0,0].grid(True, which="both", ls="-", color='0.65')
axes[0,1].grid(True, which="both", ls="-", color='0.65')
axes[1,0].grid(True, which="both", ls="-", color='0.65')
axes[1,1].grid(True, which="both", ls="-", color='0.65')
axes[0,2].grid(True, which="both", ls="-", color='0.65')
axes[1,2].grid(True, which="both", ls="-", color='0.65')

axes[0,0].set_title('PINN-like training')
axes[0,1].set_title('Operator training')
axes[0,2].set_title('Finetuning')

fig.savefig('boundary-conditions-for-pinos-code/figures/Darcy_flow_errors_losses.pdf', format='pdf')