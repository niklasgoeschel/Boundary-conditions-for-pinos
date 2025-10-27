import numpy as np

def get_best_worst_average_finetuning(path, epoch, name):
    losses = np.zeros((1000,100))
    errors = np.zeros((1000,100))
    with open(path, 'rb') as f:
        for i in range(100):
            losses[:,i] = np.load(f)
            errors[:,i] = np.load(f)

    errors_at_epoch = errors[epoch,:]
    std_dev = np.std(errors_at_epoch)

    sorted_errors = np.sort(errors_at_epoch)
    min_error = sorted_errors[0]
    max_error = sorted_errors[-1]
    average_error = np.mean(sorted_errors)

    print(name+':', ' Average: ', '{0:0.4f}'.format(round(average_error, 4)), ' standard deviation: ', '{0:0.4f}'.format(round(std_dev, 4)), ' Lowest: ', '{0:0.4f}'.format(round(min_error, 4)), ' Highest: ', '{0:0.4f}'.format(round(max_error, 4)) )

print('Before Finetuning:')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/orthogonal_projections.npy', 0, 'OP')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/generalized_local_solution_structure.npy', 0, 'GLSS')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/semi_weak.npy', 0, 'Semi-weak')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/weak.npy', 0, 'Weak')

print('After Finetuning:')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/orthogonal_projections.npy', -1, 'OP')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/generalized_local_solution_structure.npy', -1, 'GLSS')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/semi_weak.npy', -1, 'Semi-weak')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/weak.npy', -1, 'Weak')

print('PINN-like Training:')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/orthogonal_projections.npy', -1, 'OP')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/generalized_local_solution_structure.npy', -1, 'GLSS')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/semi_weak.npy', -1, 'Semi-weak')
get_best_worst_average_finetuning('boundary-conditions-for-pinos-code/results_Darcy_flow/PINN_like_training/weak.npy', -1, 'Weak')