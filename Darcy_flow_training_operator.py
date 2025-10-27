import torch
from train import train_L_shape, train_L_shape_weak, train_L_shape_semi_weak
from Darcy_flow_ansatz_functions import ansatz_orthogonal_projections, ansatz_continous_different_phis
import time
import numpy as np

""" Running this file will train the neural operator and save the data in the 'results' folder for each method.
For the training, a dataset consisting of 400 parameters is used.
At the end, the accuracy of each method is evaluated by testing on a different dataset consisting of 100 parameters. """


#########################################################################################################
# Training
#########################################################################################################

# Weak boundary conditions


start_time = time.time()
train_L_shape_weak(epochs=500, 
                   res_size=101,
                   num=400,
                   batch_size=10,
                   epochs_per_milestone=100,
                   lam=1,
                   save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_1',
                   checkpoint=False)


train_L_shape_weak(epochs=500, 
                   res_size=101,
                   num=400,
                   batch_size=10,
                   epochs_per_milestone=100,
                   lam=1,
                   save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_2',
                   checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_1')

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

# Semi-weak boundary conditions

start_time = time.time()
train_L_shape_semi_weak(epochs=500, 
                   res_size=101,
                   num=400,
                   batch_size=10,
                   epochs_per_milestone=100,
                   lam=1,
                   save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_1',
                   checkpoint=False) 

train_L_shape_semi_weak(epochs=500, 
                   res_size=101,
                   num=400,
                   batch_size=10,
                   epochs_per_milestone=100,
                   lam=1,
                   save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_2',
                   checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_1')

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

# Exact boundary conditions (Generalized local solution structures)

start_time = time.time()
train_L_shape(epochs=500, 
              res_size=101,
              ansatz_function=ansatz_continous_different_phis,
              out_dim=4,
              num=400,
              batch_size=10,
              epochs_per_milestone=100,
              save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_1',
              checkpoint=False)

train_L_shape(epochs=500, 
              res_size=101,
              ansatz_function=ansatz_continous_different_phis,
              out_dim=4,
              num=400,
              batch_size=10,
              epochs_per_milestone=100,
              save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_2',
              checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_1')

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)


# Exact boundary conditions (Orthogonal projections)

start_time = time.time()
train_L_shape(epochs=500, 
              res_size=101,
              ansatz_function=ansatz_orthogonal_projections,
              out_dim=2,
              num=400,
              batch_size=10,
              epochs_per_milestone=100,
              save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_1',
              checkpoint=False)


train_L_shape(epochs=500, 
              res_size=101,
              ansatz_function=ansatz_orthogonal_projections,
              out_dim=2,
              num=400,
              batch_size=10,
              epochs_per_milestone=100,
              save='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_2',
              checkpoint='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_1')

end_time = time.time()
with open('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)