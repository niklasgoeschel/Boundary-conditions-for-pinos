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

model = torch.load('boundary-conditions-for-pinos-code/results_Darcy_flow/operator_training/semi_weak/ckp_2/net.pt', weights_only=False)

model.eval()

res_size = 101

dx = 1/(res_size-1)

x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

phi_1 = x
phi_2 = y

w_1 = phi_2 / (phi_1+phi_2)
w_2 = phi_1 / (phi_1+phi_2)

w_1[1,1] = 0.5
w_2[1,1] = 0.5

dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=1, offset=0)
dataloader = DataLoader(dataset, batch_size=1)

inference_times = np.zeros((101))
with torch.no_grad():
    for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
        for i in range(101):
            start_time = time.time()
            output = model(input)

            pred = w_1*g1 + w_2*g2 + phi_1*phi_2*output[..., 0]
            
            end_time = time.time()
            inference_times[i] = end_time - start_time


print("Inference time (semi weak): ", np.mean(inference_times[1::]))