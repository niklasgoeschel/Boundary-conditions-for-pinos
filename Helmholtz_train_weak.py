import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
import time

from matplotlib import cm


model = FNN2d(modes1=[20, 20, 20, 20],
              modes2=[20, 20, 20, 20],
              fc_dim=128,
              layers=[64, 64, 64, 64, 64],
              activation='gelu',
              out_dim=1).cuda()

optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                 lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[200,500,900,1300,1700],
                                                gamma=0.5)

model.train()

res_size = 101
dx = 1/(res_size-1)

x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

interior = torch.ones((res_size+2,res_size+2)).cuda()
for i in range(res_size+2):
    for j in range(res_size+2):
        if x[i,j] >= 0.5 and y[i,j] >= 0.5:
            interior[i,j] = 0

interior_boundary = torch.ones((res_size+2,res_size+2)).cuda()
for i in range(res_size+2):
    for j in range(res_size+2):
        if x[i,j] > 0.5 and y[i,j] > 0.5:
            interior_boundary[i,j] = 0

input = torch.ones( (1, res_size+2, res_size+2, 3) ).cuda()
input[0, :, :, 1] = x
input[0, :, :, 2] = y

# Analytic solution
k = 1
u = torch.sin(np.sqrt(3)/2*k*x) * torch.sin(1/2*k*y)

# Boundary conditions
g1 = 0*x[1:-1,1:-1]
g2 = 0*x[1:-1,1:-1]
h3 = np.sqrt(3)/2*k * np.cos(np.sqrt(3)/2*k) * torch.sin(1/2*k*y[1:-1,1:-1])
h4 = 1/2*k * torch.sin(np.sqrt(3)/2*k*x[1:-1,1:-1]) * np.cos(1/4*k)
h5 = np.sin(np.sqrt(3)/4*k) * torch.sin(1/2*k*y[1:-1,1:-1]) + np.sqrt(3)/2*k * np.cos(np.sqrt(3)/4*k) * torch.sin(1/2*k*y[1:-1,1:-1])
h6 = torch.sin(np.sqrt(3)/2*k*x[1:-1,1:-1]) * np.sin(1/2*k) + 1/2*k * torch.sin(np.sqrt(3)/2*k*x[1:-1,1:-1]) * np.cos(1/2*k)


losses = []
errors = []

epochs = 2000
start_time = time.time()
for i in range(epochs):
    optimizer.zero_grad()
    output = model(input)

    pred = output[...,0]
    pred_x, pred_y = (pred[:, 2::, 1:-1] - pred[:, 0:-2, 1:-1]) / dx / 2, (pred[:, 1:-1, 2::] - pred[:, 1:-1, 0:-2]) / dx / 2
    pred = pred[:,1:-1,1:-1]

    pred_xx = (pred[:, 2::, 1:-1] - 2*pred[:, 1:-1, 1:-1] + pred[:, 0:-2, 1:-1]) / dx**2
    pred_yy = (pred[:, 1:-1, 2::] - 2*pred[:, 1:-1, 1:-1] + pred[:, 1:-1, 0:-2]) / dx**2

    loss_pde = torch.nn.functional.mse_loss((pred_xx + pred_yy + k**2 * pred[:,1:-1,1:-1])*interior[2:-2,2:-2], torch.zeros(pred_xx.shape).cuda())
    residual_1 = pred[0,0,:] - g1[0,:]
    residual_2 = pred[0,:,0] - g2[:,0]
    residual_3 = pred_x[0, -1, 0:int(res_size/2)+1] - h3[-1, 0:int(res_size/2)+1]
    residual_4 = pred_y[0, -(int(res_size/2)+1)::, int(res_size/2)] - h4[-(int(res_size/2)+1)::, int(res_size/2)]
    residual_5 = pred_x[0, int(res_size/2), -(int(res_size/2)+1)::] + pred[0, int(res_size/2), -(int(res_size/2)+1)::] - h5[int(res_size/2), -(int(res_size/2)+1)::]
    residual_6 = pred_y[0, 0:int(res_size/2)+1, -1] + pred[0, 0:int(res_size/2)+1, -1] - h6[0:int(res_size/2)+1, -1]
    residual_bc = torch.cat( (residual_1, residual_2, residual_3, residual_4, residual_5, residual_6) )
    loss_bc = torch.nn.functional.mse_loss(residual_bc, torch.zeros(residual_bc.shape).cuda())
    loss = loss_pde + loss_bc

    error = torch.norm( (pred[0, 1:-1,1:-1] - u[2:-2,2:-2])*interior_boundary[2:-2,2:-2], 2 ) / torch.norm(u[2:-2,2:-2]*interior_boundary[2:-2,2:-2], 2)

    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())
    errors.append(error.item())

    print('epoch: ', i, ' loss: ', loss.item(), ' lr: ', scheduler.get_last_lr()[0], ' error:', error.item())

end_time = time.time()
print("Training time: ", end_time-start_time)

with open('boundary-conditions-for-pinos-code/results_Helmholtz/weak/losses_errors.npy', 'wb') as f:
    np.save(f, np.array(losses))
    np.save(f, np.array(errors))

with open('boundary-conditions-for-pinos-code/results_Helmholtz/weak/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

torch.save(model, 'boundary-conditions-for-pinos-code/results_Helmholtz/weak/net.pt')