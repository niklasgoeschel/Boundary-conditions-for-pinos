import torch
import numpy as np
import math
import time

n_col_circle = 1000

res_size_x = 441
res_size_y = 83

dx = 2.2 / (res_size_x-1)
dy = 0.41 / (res_size_y-1)

x = torch.Tensor([[i*dx for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()
y = torch.Tensor([[j*dy for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()

x_extended = torch.Tensor([[(i-1)*dx for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()
y_extended = torch.Tensor([[(j-1)*dy for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()

########################################################
# Generalized local solution structure

model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/generalized_local_solution_structure/net.pt', weights_only=False).cuda()
model.eval()

phi_1 = x
phi_2 = y
phi_3 = (2.2 - x)**2
phi_4 = 0.41 - y
phi_S = torch.sqrt( (x-0.2)**2 + (y-0.2)**2 ) - 0.05

phi_A = x_extended+y_extended
phi_B = -x_extended+y_extended+2.2
phi_C = -x_extended-y_extended+2.61
phi_D = x_extended-y_extended+0.41

phi_3_tilde = 2.2 - x

den = phi_2 * phi_3 * phi_4 * phi_S + phi_1 * phi_3 * phi_4 * phi_S + phi_1 * phi_2 * phi_4 * phi_S + phi_1 * phi_2 * phi_3 * phi_S + phi_1 * phi_2 * phi_3 * phi_4

w_1 = phi_2 * phi_3 * phi_4 * phi_S / den
w_2 = phi_1 * phi_3 * phi_4 * phi_S / den
w_3 = phi_1 * phi_2 * phi_4 * phi_S / den
w_4 = phi_1 * phi_2 * phi_3 * phi_S / den
w_S = phi_1 * phi_2 * phi_3 * phi_4 / den

for w_i, phi_i in [(w_1, phi_1),(w_2, phi_2),(w_3, phi_3),(w_4, phi_4),(w_S, phi_S)]:
    for i in range(res_size_x):
        for j in range(res_size_y):
            if math.isnan(w_i[i,j]):
                if phi_i[i,j] == 0:
                    w_i[i,j] = 0.5
                else:
                    w_i[i,j] = 0

U = 0.3
g_1 = torch.zeros( (res_size_x,res_size_y,2) ).cuda()
g_1[..., 0] = 4*U*y*(0.41-y)/0.41**2

input = torch.ones( (1, res_size_x+2, res_size_y+2, 3) ).cuda()
input[0, :, :, 1] = x_extended
input[0, :, :, 2] = y_extended

sqrt_nu = np.sqrt(0.001)

start_time = time.time()
output = model(input)

Psi_u = output[..., 0]
Psi_v = output[..., 1]
Psi_p = output[..., 2]
Psi_u_3 = (-x_extended-y_extended+2.61) * (-x_extended+y_extended+2.2) * output[..., 3]
Psi_v_3 = (-x_extended-y_extended+2.61) * (-x_extended+y_extended+2.2) * output[..., 4]

Psi_p_1 = (output[..., 5]*phi_D + output[..., 8]*phi_A) / (phi_A + phi_D)
Psi_p_2 = (output[..., 5]*phi_B + output[..., 6]*phi_A) / (phi_A + phi_B)
Psi_p_3 = (output[..., 6]*phi_C + output[..., 7]*phi_B) / (phi_B + phi_C)
Psi_p_4 = (output[..., 7]*phi_D + output[..., 8]*phi_C) / (phi_C + phi_D)
Psi_p_S = output[..., 9]

Psi_u_3_x = (Psi_u_3[:, 2::, 1:-1] - Psi_u_3[:, 0:-2, 1:-1]) / dx / 2
Psi_v_3_x = (Psi_v_3[:, 2::, 1:-1] - Psi_v_3[:, 0:-2, 1:-1]) / dx / 2

Psi_u = Psi_u[:, 1:-1, 1:-1]
Psi_v = Psi_v[:, 1:-1, 1:-1]
Psi_p = Psi_p[:, 1:-1, 1:-1]

pred_u = w_1 * g_1[..., 0] + w_3 * ( Psi_u_3[:, 1:-1, 1:-1] + phi_3_tilde * (Psi_u_3_x - Psi_p_3[:, 1:-1, 1:-1]/sqrt_nu) ) + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_u
pred_v = w_3 * ( Psi_v_3[:, 1:-1, 1:-1] + phi_3_tilde * Psi_v_3_x ) + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_v
pred_p = w_1*Psi_p_1[:,1:-1,1:-1] + w_2*Psi_p_2[:,1:-1,1:-1] + w_3*Psi_p_3[:,1:-1,1:-1] + w_4*Psi_p_4[:,1:-1,1:-1] + w_S*Psi_p_S[:,1:-1,1:-1] + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_p
end_time = time.time()
print("Eval time (GLSS): {0:0.4f} (s)".format(round(end_time-start_time, 4)))

navier_stokes_prediction_generalized_local_solution_structure = np.zeros((res_size_x,res_size_y,3))
navier_stokes_prediction_generalized_local_solution_structure[...,0] = pred_u.cpu().detach().numpy()
navier_stokes_prediction_generalized_local_solution_structure[...,1] = pred_v.cpu().detach().numpy()
navier_stokes_prediction_generalized_local_solution_structure[...,2] = pred_p.cpu().detach().numpy()


########################################################
# Orthogonal projections

model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/net.pt', weights_only=False).cuda()
model.eval()

phi_1_extended = x_extended
phi_2_extended = y_extended
phi_4_extended = 0.41 - y_extended
phi_S_extended = torch.sqrt( (x_extended-0.2)**2 + (y_extended-0.2)**2 ) - 0.05

start_time = time.time()
output = model(input)

Psi_u = output[..., 0]
Psi_v = output[..., 1]
Psi_p = output[..., 2]
Psi_u_bar = (phi_S_extended)/(phi_1_extended*phi_2_extended*phi_4_extended + phi_S_extended) * 4*U*y_extended*(0.41-y_extended)/0.41**2 + phi_1_extended * phi_2_extended * phi_4_extended * phi_S_extended * output[..., 3]
Psi_v_bar = phi_1_extended * phi_2_extended * phi_4_extended * phi_S_extended * output[..., 4]
Psi_p_bar = output[..., 5]

ones_n1 = torch.ones((res_size_x+2, 1)).cuda()

Psi_u_3_projection = ones_n1 * Psi_u_bar[:,-2,:].reshape(Psi_u_bar.shape[0],1,res_size_y+2)
Psi_v_3_projection = ones_n1 * Psi_v_bar[:,-2,:].reshape(Psi_v_bar.shape[0],1,res_size_y+2)
Psi_p_3_projection = ones_n1 * Psi_p_bar[:,-2,:].reshape(Psi_p_bar.shape[0],1,res_size_y+2)

Psi_u = Psi_u[:, 1:-1, 1:-1]
Psi_v = Psi_v[:, 1:-1, 1:-1]
Psi_p = Psi_p[:, 1:-1, 1:-1]

pred_u = w_1 * g_1[..., 0] + w_3 * ( Psi_u_3_projection[:, 1:-1, 1:-1] + phi_3_tilde * (- Psi_p_bar[:, 1:-1, 1:-1]/sqrt_nu) ) + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_u
pred_v = w_3 * ( Psi_v_3_projection[:, 1:-1, 1:-1] ) + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_v
pred_p = Psi_p_bar[:, 1:-1, 1:-1] + (2.2 - x) * Psi_p
end_time = time.time()
print("Eval time (OP): {0:0.4f} (s)".format(round(end_time-start_time, 4)))

navier_stokes_prediction_orthogonal_projections = np.zeros((res_size_x,res_size_y,3))
navier_stokes_prediction_orthogonal_projections[...,0] = pred_u.cpu().detach().numpy()
navier_stokes_prediction_orthogonal_projections[...,1] = pred_v.cpu().detach().numpy()
navier_stokes_prediction_orthogonal_projections[...,2] = pred_p.cpu().detach().numpy()


########################################################
# Semi-weak

model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/semi_weak/net.pt', weights_only=False).cuda()
model.eval()

phi_1 = x_extended
phi_2 = y_extended
phi_4 = 0.41 - y_extended
phi_S = torch.sqrt( (x_extended-0.2)**2 + (y_extended-0.2)**2 ) - 0.05

den = phi_2 * phi_4 * phi_S + phi_1 * phi_4 * phi_S + phi_1 * phi_2 * phi_S + phi_1 * phi_2 * phi_4

w_1 = phi_2 * phi_4 * phi_S / den
w_2 = phi_1 * phi_4 * phi_S / den
w_4 = phi_1 * phi_2 * phi_S / den
w_S = phi_1 * phi_2 * phi_4 / den

for w_i, phi_i in [(w_1, phi_1),(w_2, phi_2),(w_4, phi_4),(w_S, phi_S)]:
    for i in range(res_size_x+2):
        for j in range(res_size_y+2):
            if math.isnan(w_i[i,j]):
                if phi_i[i,j] == 0:
                    w_i[i,j] = 0.5
                else:
                    w_i[i,j] = 0

U = 0.3
g_1 = torch.zeros( (res_size_x+2,res_size_y+2,2) ).cuda()
g_1[..., 0] = 4*U*y_extended*(0.41-y_extended)/0.41**2

start_time = time.time()
output = model(input)

Psi_u = output[..., 0]
Psi_v = output[..., 1]
Psi_p = output[..., 2]

pred_u = w_1 * g_1[..., 0] + phi_1 * phi_2 * phi_4 * phi_S * Psi_u
pred_v = phi_1 * phi_2 * phi_4 * phi_S * Psi_v
pred_p = Psi_p

pred_u = pred_u[:,1:-1,1:-1]
pred_v = pred_v[:,1:-1,1:-1]
pred_p = pred_p[:,1:-1,1:-1]
end_time = time.time()
print("Eval time (Semi-weak): {0:0.4f} (s)".format(round(end_time-start_time, 4)))

navier_stokes_prediction_semi_weak = np.zeros((res_size_x,res_size_y,3))
navier_stokes_prediction_semi_weak[...,0] = pred_u.cpu().detach().numpy()
navier_stokes_prediction_semi_weak[...,1] = pred_v.cpu().detach().numpy()
navier_stokes_prediction_semi_weak[...,2] = pred_p.cpu().detach().numpy()



########################################################
# Weak

model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/weak/net.pt', weights_only=False).cuda()
model.eval()

start_time = time.time()
output = model(input)

Psi_u = output[..., 0]
Psi_v = output[..., 1]
Psi_p = output[..., 2]

pred_u = Psi_u
pred_v = Psi_v
pred_p = Psi_p

pred_u = pred_u[:,1:-1,1:-1]
pred_v = pred_v[:,1:-1,1:-1]
pred_p = pred_p[:,1:-1,1:-1]
end_time = time.time()
print("Eval time (Weak): {0:0.4f} (s)".format(round(end_time-start_time, 4)))

navier_stokes_prediction_weak = np.zeros((res_size_x,res_size_y,3))
navier_stokes_prediction_weak[...,0] = pred_u.cpu().detach().numpy()
navier_stokes_prediction_weak[...,1] = pred_v.cpu().detach().numpy()
navier_stokes_prediction_weak[...,2] = pred_p.cpu().detach().numpy()



print('')
########################################################
# Numerical quantities
########################################################

navier_stokes_prediction_generalized_local_solution_structure[...,2] = navier_stokes_prediction_generalized_local_solution_structure[...,2] * np.sqrt(0.001)
navier_stokes_prediction_orthogonal_projections[...,2] = navier_stokes_prediction_orthogonal_projections[...,2] * np.sqrt(0.001)
navier_stokes_prediction_semi_weak[...,2] = navier_stokes_prediction_semi_weak[...,2] * np.sqrt(0.001)
navier_stokes_prediction_weak[...,2] = navier_stokes_prediction_weak[...,2] * np.sqrt(0.001)

########################################################
# Pressure difference

weights_pressure_difference = np.zeros((2,res_size_x,res_size_y))

for n,x,y in [(0,0.15,0.2),(1,0.25,0.2)]:
    i = int(x/dx)
    j = int(y/dy)
    alpha = x/dx - i
    beta = y/dy - j
    weights_pressure_difference[n,i,j] = (1-alpha) * (1-beta)
    weights_pressure_difference[n,i+1,j] = alpha * (1-beta)
    weights_pressure_difference[n,i,j+1] = (1-alpha) * beta
    weights_pressure_difference[n,i+1,j+1] = alpha * beta

print('Pressure difference (GLSS):', "{0:0.4f}".format(round(np.sum(weights_pressure_difference * navier_stokes_prediction_generalized_local_solution_structure[...,2], axis=(1,2)) @ np.array([1,-1]), 4)) )
print('Pressure difference (OP):', "{0:0.4f}".format(round(np.sum(weights_pressure_difference * navier_stokes_prediction_orthogonal_projections[...,2], axis=(1,2)) @ np.array([1,-1]), 4)) )
print('Pressure difference (semi-weak):', "{0:0.4f}".format(round(np.sum(weights_pressure_difference * navier_stokes_prediction_semi_weak[...,2], axis=(1,2)) @ np.array([1,-1]) , 4)) )
print('Pressure difference (weak):', "{0:0.4f}".format(np.sum(weights_pressure_difference * navier_stokes_prediction_weak[...,2], axis=(1,2)) @ np.array([1,-1]), 4) )


########################################################
# Drag coefficient

def drag_coefficient(pred,n_col_circle):
    F_D = 0
    dS = np.pi * 0.1 / n_col_circle
    for n in range(n_col_circle):
        weights_circle = np.zeros((res_size_x,res_size_y))
        x_col = 0.2 + 0.05*np.cos(2*np.pi*n/n_col_circle)
        y_col = 0.2 + 0.05*np.sin(2*np.pi*n/n_col_circle)
        i = int(x_col/dx)
        j = int(y_col/dy)
        alpha = x_col/dx - i
        beta = y_col/dy - j
        weights_circle[i,j] = (1-alpha) * (1-beta)
        weights_circle[i+1,j] = alpha * (1-beta)
        weights_circle[i,j+1] = (1-alpha) * beta
        weights_circle[i+1,j+1] = alpha * beta

        n_x = np.cos(2*np.pi*n/n_col_circle)
        n_y = np.sin(2*np.pi*n/n_col_circle)

        u_x = (pred[2::, 1:-1,0] - pred[0:-2, 1:-1,0]) / dx / 2
        u_y = (pred[1:-1, 2::,0] - pred[1:-1, 0:-2,0]) / dy / 2
        v_x = (pred[2::, 1:-1,1] - pred[0:-2, 1:-1,1]) / dx / 2
        v_y = (pred[1:-1, 2::,1] - pred[1:-1, 0:-2,1]) / dy / 2

        u_x = np.sum(weights_circle[1:-1,1:-1] * u_x)
        u_y = np.sum(weights_circle[1:-1,1:-1] * u_y) 
        v_x = np.sum(weights_circle[1:-1,1:-1] * v_x) 
        v_y = np.sum(weights_circle[1:-1,1:-1] * v_y)

        u_n = n_x * u_x + n_y * u_y
        v_n = n_x * v_x + n_y * v_y

        ut_n = n_y * u_n - n_x * v_n

        p = np.sum(weights_circle * pred[...,2])
        F_D = F_D + (0.001 * ut_n * n_y - p * n_x) * dS

    return 2*F_D/0.2**2/0.1

print('')

print('Drag coefficient (GLSS):', "{0:0.4f}".format(round(drag_coefficient(navier_stokes_prediction_generalized_local_solution_structure,n_col_circle), 4)) )
print('Drag coefficient (OP):', "{0:0.4f}".format(round(drag_coefficient(navier_stokes_prediction_orthogonal_projections,n_col_circle), 4)) )
print('Drag coefficient (semi weak):', "{0:0.4f}".format(round(drag_coefficient(navier_stokes_prediction_semi_weak,n_col_circle), 4)) )
print('Drag coefficient (weak):', "{0:0.4f}".format(round(drag_coefficient(navier_stokes_prediction_weak,n_col_circle), 4)) )


########################################################
# Lift coefficient

def lift_coefficient(pred,n_col_circle):
    F_D = 0
    dS = np.pi * 0.1 / n_col_circle
    for n in range(n_col_circle):
        weights_circle = np.zeros((res_size_x,res_size_y))
        x_col = 0.2 + 0.05*np.cos(2*np.pi*n/n_col_circle)
        y_col = 0.2 + 0.05*np.sin(2*np.pi*n/n_col_circle)
        i = int(x_col/dx)
        j = int(y_col/dy)
        alpha = x_col/dx - i
        beta = y_col/dy - j
        weights_circle[i,j] = (1-alpha) * (1-beta)
        weights_circle[i+1,j] = alpha * (1-beta)
        weights_circle[i,j+1] = (1-alpha) * beta
        weights_circle[i+1,j+1] = alpha * beta

        n_x = np.cos(2*np.pi*n/n_col_circle)
        n_y = np.sin(2*np.pi*n/n_col_circle)

        u_x = (pred[2::, 1:-1,0] - pred[0:-2, 1:-1,0]) / dx / 2
        u_y = (pred[1:-1, 2::,0] - pred[1:-1, 0:-2,0]) / dy / 2
        v_x = (pred[2::, 1:-1,1] - pred[0:-2, 1:-1,1]) / dx / 2
        v_y = (pred[1:-1, 2::,1] - pred[1:-1, 0:-2,1]) / dy / 2

        u_x = np.sum(weights_circle[1:-1,1:-1] * u_x)
        u_y = np.sum(weights_circle[1:-1,1:-1] * u_y) 
        v_x = np.sum(weights_circle[1:-1,1:-1] * v_x) 
        v_y = np.sum(weights_circle[1:-1,1:-1] * v_y)

        u_n = n_x * u_x + n_y * u_y
        v_n = n_x * v_x + n_y * v_y

        ut_n = n_y * u_n - n_x * v_n

        p = np.sum(weights_circle * pred[...,2])
        F_D = F_D - (0.001 * ut_n * n_x + p * n_y) * dS

    return 2*F_D/0.2**2/0.1

print('')
print('Lift coefficient (GLSS):', "{0:0.4f}".format(round(lift_coefficient(navier_stokes_prediction_generalized_local_solution_structure,n_col_circle), 4)) )
print('Lift coefficient (OP):', "{0:0.4f}".format(round(lift_coefficient(navier_stokes_prediction_orthogonal_projections,n_col_circle), 4)) )
print('Lift coefficient (semi weak):', "{0:0.4f}".format(round(lift_coefficient(navier_stokes_prediction_semi_weak,n_col_circle), 4)) )
print('Lift coefficient (weak):', "{0:0.4f}".format(round(lift_coefficient(navier_stokes_prediction_weak,n_col_circle), 4)) )