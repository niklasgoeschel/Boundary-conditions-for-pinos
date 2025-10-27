import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
import time

from matplotlib import cm


model = torch.load('boundary-conditions-for-pinos-code/results_Navier_stokes/semi_weak/net.pt', weights_only=False)

model.eval()

res_size_x = 441
res_size_y = 83

dx = 2.2 / (res_size_x-1)
dy = 0.41 / (res_size_y-1)

x = torch.Tensor([[i*dx for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()
y = torch.Tensor([[j*dy for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()

x_extended = torch.Tensor([[(i-1)*dx for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()
y_extended = torch.Tensor([[(j-1)*dy for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()

phi_1 = x_extended
phi_2 = y_extended
phi_4 = 0.41 - y_extended
phi_S = torch.sqrt( (x_extended-0.2)**2 + (y_extended-0.2)**2 ) - 0.05

interior = torch.ones((res_size_x,res_size_y)).cuda()
for i in range(res_size_x):
    for j in range(res_size_y):
        if phi_S[i+1,j+1] <= 0:
            interior[i,j] = 0


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

        pred_u = w_1 * g_1[..., 0] + phi_1 * phi_2 * phi_4 * phi_S * Psi_u
        pred_v = phi_1 * phi_2 * phi_4 * phi_S * Psi_v
        pred_p = Psi_p

        pred_u = pred_u[:,1:-1,1:-1]
        pred_v = pred_v[:,1:-1,1:-1]
        pred_p = pred_p[:,1:-1,1:-1]
        end_time = time.time()
        inference_times[i] = end_time - start_time


print("Inference time (semi weak): ", np.mean(inference_times[1::]))