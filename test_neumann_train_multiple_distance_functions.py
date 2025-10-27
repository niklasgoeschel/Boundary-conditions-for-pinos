import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
from pino.after_training import save_model
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam

from torch.utils.data import DataLoader
from pino.datasets import L_shape



from matplotlib import cm


model = FNN2d(modes1=[20, 20, 20, 20],
              modes2=[20, 20, 20, 20],
              fc_dim=128,
              layers=[64, 64, 64, 64, 64],
              activation='gelu',
              out_dim=5).cuda()

optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                 lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[200,500,800],
                                                gamma=0.5)

model.train()

res_size = 201

x = torch.Tensor([[(i)/(res_size-1) for j in range(res_size)] for i in range(res_size)] ).cuda()
y = torch.Tensor([[(j)/(res_size-1) for j in range(res_size)] for i in range(res_size)] ).cuda()


phi_1 = x
phi_2 = (1 - x ).cuda()
phi_3 = y 
phi_4 = (1- y).cuda()


phi_A = (x**2 + y**2)
phi_B = ((x-1)**2 + y**2)
phi_C = ((x-1)**2 + (y-1)**2)
phi_D = (x**2 + (y-1)**2)


mu_1 = 2
mu_2 = 2
mu_3 = 2
mu_4 = 2

den = (phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4) + (phi_1**mu_1 * phi_3**mu_3 * phi_4**mu_4) + (phi_1**mu_1 * phi_2**mu_2 * phi_4**mu_4) + (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3)

phi = (phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4)[1:-1,1:-1]


w_1 = ((phi_2**mu_2 * phi_3**mu_3 * phi_4**mu_4) / den)[1:-1,1:-1]
w_2 = ((phi_1**mu_1 * phi_3**mu_3 * phi_4**mu_4) / den)[1:-1,1:-1]
w_3 = ((phi_1**mu_1 * phi_2**mu_2 * phi_4**mu_4) / den)[1:-1,1:-1]
w_4 = ((phi_1**mu_1 * phi_2**mu_2 * phi_3**mu_3) / den)[1:-1,1:-1]


u_sol = torch.cos(np.pi * x) * torch.cos(np.pi * y) 

input = torch.ones(1, x.size()[0], y.size()[0], 3).cuda()
input[..., 1] = x.reshape((1, x.size()[0], y.size()[0]))
input[..., 2] = y.reshape((1, x.size()[0], y.size()[0]))

f = torch.ones(1, x.size()[0], y.size()[0]).cuda()
f = f * (2*np.pi**2) * torch.cos( np.pi*x ) * torch.cos( np.pi*y )


a = input[..., 0]
f_losses = []
data_losses = []
for i in range(1000):
    optimizer.zero_grad()
    output = model(input)

    Psi_1 = output[...,0]
    Psi_2 = output[...,1]
    Psi_3 = output[...,2]
    Psi_4 = output[...,3]
    Psi = output[...,4]

    Psi_1_x, Psi_1_y = (Psi_1[:, 2::, 1:-1] - Psi_1[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_1[:, 1:-1, 2::] - Psi_1[:, 1:-1, 0:-2]) * (res_size-1) / 2
    Psi_2_x, Psi_2_y = (Psi_2[:, 2::, 1:-1] - Psi_2[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_2[:, 1:-1, 2::] - Psi_2[:, 1:-1, 0:-2]) * (res_size-1) / 2
    Psi_3_x, Psi_3_y = (Psi_3[:, 2::, 1:-1] - Psi_3[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_3[:, 1:-1, 2::] - Psi_3[:, 1:-1, 0:-2]) * (res_size-1) / 2
    Psi_4_x, Psi_4_y = (Psi_4[:, 2::, 1:-1] - Psi_4[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_4[:, 1:-1, 2::] - Psi_4[:, 1:-1, 0:-2]) * (res_size-1) / 2

    u_1 = Psi_1[:,1:-1,1:-1] - phi_1[1:-1,1:-1] * Psi_1_x
    u_2 = Psi_2[:,1:-1,1:-1] + phi_2[1:-1,1:-1] * Psi_2_x 
    u_3 = Psi_3[:,1:-1,1:-1] - phi_3[1:-1,1:-1] * Psi_3_y 
    u_4 = Psi_4[:,1:-1,1:-1] + phi_4[1:-1,1:-1] * Psi_4_y

    pred = w_1*u_1 + w_2*u_2 + w_3*u_3 + w_4*u_4 + phi*Psi[:,1:-1,1:-1]

    pred_x, pred_y = (pred[:, 2::, 1:-1] - pred[:, 0:-2, 1:-1]) * (res_size-1) / 2, (pred[:, 1:-1, 2::] - pred[:, 1:-1, 0:-2]) * (res_size-1) / 2
    a_pred_x, a_pred_y = pred_x, pred_y 
    a_pred_xx, a_pred_yy = (a_pred_x[:, 2::, 1:-1] - a_pred_x[:, 0:-2, 1:-1]) * (res_size-1) / 2, (a_pred_y[:, 1:-1, 2::] - a_pred_y[:, 1:-1, 0:-2]) * (res_size-1) / 2
    residual = - ( a_pred_xx + a_pred_yy ) - f[:,3:-3,3:-3]

    loss = torch.nn.functional.mse_loss(residual,torch.zeros(residual.shape).cuda())

    prediction = pred - torch.sum(pred[0,...]) / (res_size-2)**2
    error = torch.norm(prediction-u_sol[1:-1,1:-1]) / torch.norm(u_sol[1:-1,1:-1])

    loss.backward()
    optimizer.step()

    print("Epoche:", i, " loss:", loss.item(), " L2-error:", error.item())
    scheduler.step()



with open('boundary-conditions-for-pinos-code/results_test_neumann/multiple_distance_functions.npy', 'wb') as f:
    np.save(f, np.array(prediction[0,...].cpu().detach().numpy()))