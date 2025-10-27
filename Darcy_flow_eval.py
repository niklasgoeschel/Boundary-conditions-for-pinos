from eval import eval_L_shape, eval_L_shape_weak, eval_L_shape_semi_weak
from Darcy_flow_ansatz_functions import ansatz_orthogonal_projections, ansatz_continous_different_phis

eval_L_shape_weak(res_size=101,
                  path='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/weak/ckp_2',
                  lam=1,
                  num=100,
                  offset=400)

eval_L_shape_semi_weak(res_size=101,
                       path='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_2',
                       num=100,
                       offset=400)

eval_L_shape(res_size=101,
             ansatz_function=ansatz_continous_different_phis,
             path='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_2',
             num=100,
             offset=400)


eval_L_shape(res_size=101,
             ansatz_function=ansatz_orthogonal_projections,
             path='boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/orthogonal_projections/ckp_2',
             num=100,
             offset=400)