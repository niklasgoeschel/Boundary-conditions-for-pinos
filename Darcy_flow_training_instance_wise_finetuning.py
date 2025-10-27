from train import train_L_shape, train_L_shape_weak, train_L_shape_semi_weak
from Darcy_flow_ansatz_functions import ansatz_orthogonal_projections, ansatz_continous_different_phis

import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instance-wise finetuning
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# Exact boundary conditions (generalized local solution structure)

start_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/generalized_local_solution_structure.npy', 'wb') as f:
    for i in range(100):
        print("Offset: ", i)
        losses, errors = train_L_shape(epochs=1000, 
                                       res_size=101,
                                       ansatz_function=ansatz_continous_different_phis,
                                       out_dim=4,
                                       num=1,
                                       batch_size=1,
                                       offset=400+i,
                                       epochs_per_milestone=200,
                                       save=False,
                                       checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_2',
                                       return_errors_losses=True,
                                       lr=0.0025)
        np.save(f, losses)
        np.save(f, errors)

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/generalized_local_solution_structure_runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

# Exact boundary conditions (orthogonal projections)

start_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/orthogonal_projections.npy', 'wb') as f:
    for i in range(100):
        print("Offset: ", i)
        losses, errors = train_L_shape(epochs=1000, 
                                       res_size=101,
                                       ansatz_function=ansatz_orthogonal_projections,
                                       out_dim=2,
                                       num=1,
                                       batch_size=1,
                                       offset=400+i,
                                       epochs_per_milestone=200,
                                       save=False,
                                       checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_2',
                                       return_errors_losses=True,
                                       lr=0.0025)
        np.save(f, losses)
        np.save(f, errors)

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/orthogonal_projections_runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

# Semi-weak boundary conditions

start_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/semi_weak.npy', 'wb') as f:
    for i in range(100):
        print("Offset: ", i)
        losses, errors = train_L_shape_semi_weak(epochs=1000, 
                                       res_size=101,
                                       num=1,
                                       batch_size=1,
                                       offset=400+i,
                                       epochs_per_milestone=200,
                                       lam=1,
                                       save=False,
                                       checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_2',
                                       return_errors_losses=True,
                                       lr=0.0025)
        np.save(f, losses)
        np.save(f, errors)

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/semi_weak_runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

# Weak boundary conditions

start_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/weak.npy', 'wb') as f:
    for i in range(100):
        print("Offset: ", i)
        losses, errors = train_L_shape_weak(epochs=1000, 
                                       res_size=101,
                                       num=1,
                                       batch_size=1,
                                       offset=400+i,
                                       epochs_per_milestone=200,
                                       lam=1,
                                       save=False,
                                       checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_2',
                                       return_errors_losses=True,
                                       lr=0.0025)
        np.save(f, losses)
        np.save(f, errors)

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/instance_wise_finetuning/weak_runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)