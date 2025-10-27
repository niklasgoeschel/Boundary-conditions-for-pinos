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
              out_dim=2).cuda()

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


phi = 1 / ( 1/phi_1 + 1/phi_2 + 1/phi_3 + 1/phi_4 )[1:-1,1:-1]

phi_x = (-1 / ( 1/phi_1 + 1/phi_2 + 1/phi_3 + 1/phi_4 )**2 * ( -1/x**2 + 1/(1-x)**2 ))[1:-1,1:-1]
phi_y = (-1 / ( 1/phi_1 + 1/phi_2 + 1/phi_3 + 1/phi_4 )**2 * ( -1/y**2 + 1/(1-y)**2 ))[1:-1,1:-1]


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

    Psi_1_x, Psi_1_y = (Psi_1[:, 2::, 1:-1] - Psi_1[:, 0:-2, 1:-1]) * (res_size-1) / 2, (Psi_1[:, 1:-1, 2::] - Psi_1[:, 1:-1, 0:-2]) * (res_size-1) / 2

    Psi_1 = Psi_1[:,1:-1,1:-1]
    Psi_2 = Psi_2[:,1:-1,1:-1]

    pred = Psi_1 - phi * ( phi_x*Psi_1_x + phi_y*Psi_1_y ) + phi**2 * Psi_2


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


with open('boundary-conditions-for-pinos-code/results_test_neumann/single_distance_function.npy', 'wb') as f:
    np.save(f, np.array(prediction[0,...].cpu().detach().numpy()))