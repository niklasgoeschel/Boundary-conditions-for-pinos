import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
import time

from matplotlib import cm

from torch.utils.data import DataLoader
from pino.datasets import L_shape

model = torch.load('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/generalized_local_solution_structure/ckp_2/net.pt', weights_only=False)

model.eval()

res_size = 101

dx = 1/(res_size-1)

x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

phi_1 = x
phi_2 = y
phi_3 = (1 - x).cuda()
phi_4 = phi_lineseg(x,y, 1, 0.5, 0.5, 0.5).cuda()
phi_5 = phi_lineseg(x,y, 0.5, 0.5, 0.5, 1).cuda()
phi_6 = (1 - y).cuda()

phi_B = torch.sqrt((x-1)**2 + y**2 )
phi_C = torch.sqrt( (x-1)**2 + (y-0.5)**2 )
phi_D = torch.sqrt( (x-0.5)**2 + (y-0.5)**2 )
phi_E = torch.sqrt( (x-0.5)**2 + (y-1)**2 )
phi_F = torch.sqrt( x**2 + (y-1)**2)

mu_1 = 1
mu_2 = 1
mu_3 = 2
mu_4 = 2
mu_5 = 2
mu_6 = 2

den = (phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) + (phi_1**mu_1 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) + (phi_1**mu_1 * phi_2**mu_2 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) + (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_5**mu_5 * phi_6**mu_6) + (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_6**mu_6) + (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5)

phi = (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6)[1:-1,1:-1]

w_1 = ((phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) / den)[1:-1,1:-1]
w_2 = ((phi_1**mu_1 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) / den)[1:-1,1:-1]
w_3 = ((phi_1**mu_1 * phi_2**mu_2 * phi_4**mu_4 * phi_5**mu_5 * phi_6**mu_6) / den)[1:-1,1:-1]
w_4 = ((phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_5**mu_5 * phi_6**mu_6) / den)[1:-1,1:-1]
w_5 = ((phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_6**mu_6) / den)[1:-1,1:-1]
w_6 = ((phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4 * phi_5**mu_5) / den)[1:-1,1:-1]

for w_i, phi_i in [(w_1, phi_1),(w_2, phi_2),(w_3, phi_3),(w_4, phi_4),(w_5, phi_5),(w_6, phi_6)]:
    for i in range(res_size):
        for j in range(res_size):
            if math.isnan(w_i[i,j]):
                if phi_i[i+1,j+1] == 0:
                    w_i[i,j] = 0.5
                else:
                    w_i[i,j] = 0

phi_3_Tilde = (1 - x).cuda()[1:-1,1:-1]
phi_4_Tilde = (0.5 - y).cuda()[1:-1,1:-1]
phi_5_Tilde = (0.5 - x).cuda()[1:-1,1:-1]
phi_6_Tilde = (1 - y).cuda()[1:-1,1:-1]

dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=1, offset=0)
dataloader = DataLoader(dataset, batch_size=1)

inference_times = np.zeros((101))
with torch.no_grad():
    for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
        for i in range(101):
            start_time = time.time()
            output = model(input)

            Psi = output[..., 0]

            Psi_C = output[..., 1]
            Psi_D = output[..., 2]
            Psi_E = output[..., 3]
            
            Psi_3 = ( g2 * phi_C + Psi_C * phi_B ) / (phi_B + phi_C) 
            Psi_4 = ( Psi_C * phi_D + Psi_D * phi_C ) / (phi_C + phi_D) 
            Psi_5 = ( Psi_D * phi_E + Psi_E * phi_D ) / (phi_D + phi_E) 
            Psi_6 = ( Psi_E * phi_F + g1 * phi_E ) / (phi_E + phi_F)   

            Psi_3_x, Psi_3_y = (Psi_3[:, 2::, 1:-1] - Psi_3[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_3[:, 1:-1, 2::] - Psi_3[:, 1:-1, 0:-2]) * (res_size-1) / 2
            Psi_4_x, Psi_4_y = (Psi_4[:, 2::, 1:-1] - Psi_4[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_4[:, 1:-1, 2::] - Psi_4[:, 1:-1, 0:-2]) * (res_size-1) / 2
            Psi_5_x, Psi_5_y = (Psi_5[:, 2::, 1:-1] - Psi_5[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_5[:, 1:-1, 2::] - Psi_5[:, 1:-1, 0:-2]) * (res_size-1) / 2
            Psi_6_x, Psi_6_y = (Psi_6[:, 2::, 1:-1] - Psi_6[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_6[:, 1:-1, 2::] - Psi_6[:, 1:-1, 0:-2]) * (res_size-1) / 2

            u_1 = g1[:,1:-1,1:-1]
            u_2 = g2[:,1:-1,1:-1]
            u_3 = Psi_3[:,1:-1,1:-1] + phi_3_Tilde * ( Psi_3_x - h3[:,1:-1,1:-1] )
            u_4 = Psi_4[:,1:-1,1:-1] + phi_4_Tilde * ( Psi_4_y - h4[:,1:-1,1:-1] )
            u_5 = Psi_5[:,1:-1,1:-1] * ( 1 + phi_5_Tilde ) + phi_5_Tilde * ( Psi_5_x - h5[:,1:-1,1:-1] )
            u_6 = Psi_6[:,1:-1,1:-1] * ( 1 + phi_6_Tilde ) + phi_6_Tilde * ( Psi_6_y - h6[:,1:-1,1:-1] )
            
            pred = w_1 * u_1 + w_2 * u_2 + w_3 * u_3 + w_4 * u_4 + w_5 * u_5 + w_6 * u_6 + phi * Psi[:,1:-1,1:-1]
            
            end_time = time.time()
            inference_times[i] = end_time - start_time


print("Inference time (GLSS): ", np.mean(inference_times[1::]))