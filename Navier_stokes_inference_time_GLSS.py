import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
import time

from matplotlib import cm


model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/generalized_local_solution_structure/net.pt', weights_only=False)

model.eval()

res_size_x = 441
res_size_y = 83

dx = 2.2 / (res_size_x-1)
dy = 0.41 / (res_size_y-1)

x = torch.Tensor([[i*dx for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()
y = torch.Tensor([[j*dy for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()

x_extended = torch.Tensor([[(i-1)*dx for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()
y_extended = torch.Tensor([[(j-1)*dy for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()

phi_1 = x
phi_2 = y
phi_3 = (2.2 - x)**2
phi_4 = 0.41 - y
phi_S = torch.sqrt( (x-0.2)**2 + (y-0.2)**2 ) - 0.05

phi_A = x_extended+y_extended
phi_B = -x_extended+y_extended+2.2
phi_C = -x_extended-y_extended+2.61
phi_D = x_extended-y_extended+0.41

interior = torch.ones((res_size_x,res_size_y)).cuda()
for i in range(res_size_x):
    for j in range(res_size_y):
        if phi_S[i,j] <= 0:
            interior[i,j] = 0

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

inference_times = np.zeros((101))
with torch.no_grad():
    for i in range(101):
        start_time = time.time()
        output = model(input)

        nu = 0.001
        sqrt_nu = np.sqrt(0.001)

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
        inference_times[i] = end_time - start_time


print("Inference time (GLSS): ", np.mean(inference_times[1::]))