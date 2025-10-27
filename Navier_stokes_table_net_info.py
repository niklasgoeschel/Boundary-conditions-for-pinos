import torch
import numpy as np
import os

def table_data(method):
    with open('boundary-conditions-for-pinos-code/results_Navier_stokes/'+method+'/runtime.npy', 'rb') as f:
        runtime_operator_training = np.load(f)
        runtime_operator_training = runtime_operator_training.item()

    with open('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/'+method+'/ckp_2/predicted_solutions.npy', 'rb') as f:
        np.load(f)
        np.load(f)
        np.load(f)
        np.load(f)
        eval_time_100_parameters = np.load(f)
        eval_time_100_parameters = eval_time_100_parameters.item()
    
    model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/'+method+'/net.pt', weights_only=False)
    total_params = sum(p.numel() for p in model.parameters())

    ckp_size = os.path.getsize('boundary-conditions-for-pinos-code/results_Navier_stokes/'+method+'/net.pt')

    print(method+': ', "Training time: {0:0.2f} (min)".format(round(runtime_operator_training/60, 2)), " Trainable parameters: "+str(total_params), ' Checkpoint size: '+str(ckp_size)+' (bytes)')

table_data('generalized_local_solution_structure')
table_data('orthogonal_projections')
table_data('semi_weak')
table_data('weak')