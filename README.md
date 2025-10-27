# Boundary Conditions for PINOs


The resulting data we generated in our experiments can be found in
https://figshare.com/s/d836ba71c037c530327d.

We used the conda environment [environment.yml](environment.yml) and installed pytorch with the command
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

We used code for training the FNOs from the repository https://github.com/neuraloperator/physics_informed/tree/Grad-ckpt.

For generating the reference solution to the Navier-Stokes example, we used an adapted version of the code from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html.

# Reproducing the experiments

For reproducing the experiments we conducted, run the files from top to bottom for each problem. Make sure that the files are executed from where you see the 'boundary-conditions-for-pinos-code' directory.

## Poisson equation (test neumann example)

### Training

- [test_neumann_train_single_distance_function.py](test_neumann_train_single_distance_function.py)
- [test_neumann_train_multiple_distance_functions.py](test_neumann_train_multiple_distance_functions.py)
- [test_neumann_train_our_approach.py](test_neumann_train_our_approach.py)

### Figures

- [test_neumann_figures.py](test_neumann_figures.py) generates the subfigures in Figure 1.

## Darcy Flow

### Data generation
- [Darcy_flow_data_generation.py](Darcy_flow_data_generation.py)

### Training
Operator training + instance-wise finetuning:
- [Darcy_flow_training_operator.py](Darcy_flow_training_operator.py)
- [Darcy_flow_training_instance_wise_finetuning.py](Darcy_flow_training_instance_wise_finetuning.py)

PINN-like training:
- [Darcy_flow_training_PINN_like.py](Darcy_flow_training_PINN_like.py)

### Eval
- [Darcy_flow_eval.py](Darcy_flow_eval.py) evaluates the trained operator on the validation set.

### Table Data
- [Darcy_flow_table_data.py](Darcy_flow_table_data.py) generates data for Table 2.
- [Darcy_flow_table_net_info.py](Darcy_flow_table_net_info.py) generates data for Table 1. The inference times are determined by running [Darcy_flow_inference_time_GLSS.py](Darcy_flow_inference_time_GLSS.py), [Darcy_flow_inference_time_OP.py](Darcy_flow_inference_time_OP.py), [Darcy_flow_inference_time_semi_weak.py](Darcy_flow_inference_time_semi_weak.py) and [Darcy_flow_inference_time_weak.py](Darcy_flow_inference_time_weak.py).

### Figures
- [Darcy_flow_figure_error_loss_plots.py](Darcy_flow_figure_error_loss_plots.py) generates Figure 3.
- [Darcy_flow_figure_solutions.py](Darcy_flow_figure_solutions.py) generates Figures 7-10.


## Navier-Stokes equations

### Generation of the reference solution
We generated a reference solution with [fenics_reference_solution/DFG_2D_1.py](fenics_reference_solution/DFG_2D_1.py) (you do not need to run this) and saved the solution in [fenics_reference_solution/DFG_2D_1/pred_441_83_3.npy](fenics_reference_solution/DFG_2D_1/pred_441_83_3.npy). The code used for it is an adapted version of the code from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html. 

### Training
Generalized local solution structure:
- [Navier_stokes_train_generalized_local_solution_structure.py](Navier_stokes_train_generalized_local_solution_structure.py)

Orthogonal projections:
- [Navier_stokes_train_orthogonal_projections.py](Navier_stokes_train_orthogonal_projections.py)

Semi-weak:
- [Navier_stokes_train_semi_weak.py](Navier_stokes_train_semi_weak.py)

Weak:
- [Navier_stokes_train_weak.py](Navier_stokes_train_weak.py)

### Figures

- [Navier_stokes_figure_error_loss_plots.py](Navier_stokes_figure_error_loss_plots.py) generates Figure 4.
- [Navier_stokes_figure_solutions.py](Navier_stokes_figure_solutions.py) generates Figure 12.

### Table data

- [Navier_stokes_table_numerical_quantities.py](Navier_stokes_table_numerical_quantities.py) generates data for Table 3.
- [Navier_stokes_table_net_info.py](Navier_stokes_table_net_info.py) generates data for Table 1. The inference times are determined through running [Navier_stokes_inference_time_GLSS.py](Navier_stokes_inference_time_GLSS.py), [Navier_stokes_inference_time_OP.py](Navier_stokes_inference_time_OP.py), [Navier_stokes_inference_time_semi_weak.py](Navier_stokes_inference_time_semi_weak.py) and [Navier_stokes_inference_time_weak.py](Navier_stokes_inference_time_weak.py).
