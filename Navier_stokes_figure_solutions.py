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

with open('boundary-conditions-for-pinos-code/fenics_reference_solution/DFG_2D_1/pred_441_83_3.npy', 'rb') as f:
    navier_stokes_reference_solution = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/generalized_local_solution_structure/prediction_441_83_3.npy', 'rb') as f:
    navier_stokes_prediction_exact = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/prediction_441_83_3.npy', 'rb') as f:
    navier_stokes_prediction_orthogonal_projections = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/semi_weak/prediction_441_83_3.npy', 'rb') as f:
    navier_stokes_prediction_semi_weak = np.load(f)

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/weak/prediction_441_83_3.npy', 'rb') as f:
    navier_stokes_prediction_weak = np.load(f)

interior = np.ones((441,83))
for i in range(441):
    for j in range(83):
        x = 2.2 * i / 440
        y = 0.41 * j / 82
        if np.sqrt( (x-0.2)**2 + (y-0.2)**2 ) < 0.05:
            interior[i,j] = np.nan

sqrt_nu = np.sqrt(0.001)

velocity_reference_solution = np.sqrt( navier_stokes_reference_solution[...,0]**2 + navier_stokes_reference_solution[...,1]**2 ) * interior
pressure_reference_solution = navier_stokes_reference_solution[...,2]*interior
velocity_exact = np.sqrt( navier_stokes_prediction_exact[...,0]**2 + navier_stokes_prediction_exact[...,1]**2 ) * interior
pressure_exact = navier_stokes_prediction_exact[...,2]*interior*sqrt_nu
velocity_orthogonal_projections = np.sqrt( navier_stokes_prediction_orthogonal_projections[...,0]**2 + navier_stokes_prediction_orthogonal_projections[...,1]**2 ) * interior
pressure_orthogonal_projections = navier_stokes_prediction_orthogonal_projections[...,2]*interior*sqrt_nu
velocity_semi_weak = np.sqrt( navier_stokes_prediction_semi_weak[...,0]**2 + navier_stokes_prediction_semi_weak[...,1]**2 ) * interior
pressure_semi_weak = navier_stokes_prediction_semi_weak[...,2]*interior*sqrt_nu
velocity_weak = np.sqrt( navier_stokes_prediction_weak[...,0]**2 + navier_stokes_prediction_weak[...,1]**2 ) * interior
pressure_weak = navier_stokes_prediction_weak[...,2]*interior*sqrt_nu

vmax_velocity = np.nanmax( np.concatenate((velocity_reference_solution,velocity_exact,velocity_orthogonal_projections,velocity_semi_weak,velocity_weak)) )
vmin_velocity = np.nanmin( np.concatenate((velocity_reference_solution,velocity_exact,velocity_orthogonal_projections,velocity_semi_weak,velocity_weak)) )
norm_velocity = colors.Normalize(vmin=vmin_velocity, vmax=vmax_velocity)

vmax_pressure = np.nanmax( np.concatenate((pressure_reference_solution,pressure_exact,pressure_orthogonal_projections,pressure_semi_weak,pressure_weak)) )
vmin_pressure = np.nanmin( np.concatenate((pressure_reference_solution,pressure_exact,pressure_orthogonal_projections,pressure_semi_weak,pressure_weak)) )
norm_pressure = colors.Normalize(vmin=vmin_pressure, vmax=vmax_pressure)


fig, ax = plt.subplots(5,2, layout='constrained', figsize=(7,6))

def plot_navier_stokes(velocity, pressure, axis, label):

    axis[0].set_aspect('equal')
    c_velocity=axis[0].pcolormesh(np.array([2.2*i/(440) for i in range(253)]), np.array([0.41*i/(82) for i in range(84)]), np.transpose(velocity[0:252,:]), cmap = 'viridis', norm=norm_velocity)
    axis[0].set_title('Velocity('+label+')')
    axis[1].set_aspect('equal')
    c_pressure=axis[1].pcolormesh(np.array([2.2*i/(440) for i in range(253)]), np.array([0.41*i/(82) for i in range(84)]), np.transpose(pressure[0:252,:]), cmap = 'viridis', norm=norm_pressure)
    axis[1].set_title('Pressure('+label+')')

    if label=='Reference solution':
        fig.colorbar(c_velocity, ax=ax[:,0])
        fig.colorbar(c_pressure, ax=ax[:,1])

plot_navier_stokes(velocity_reference_solution, pressure_reference_solution, ax[0,:], 'Reference solution')
plot_navier_stokes(velocity_exact, pressure_exact, ax[1,:], 'GLSS')
plot_navier_stokes(velocity_orthogonal_projections, pressure_orthogonal_projections, ax[2,:], 'OP')
plot_navier_stokes(velocity_semi_weak, pressure_semi_weak, ax[3,:], 'Semi-weak')
plot_navier_stokes(velocity_weak, pressure_weak, ax[4,:], 'Weak')

fig.savefig('boundary-conditions-for-pinos-code/figures/Navier_stokes_solutions.pdf', format='pdf')