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
              out_dim=2).cuda()

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

input = torch.ones( (1, res_size+2, res_size+2, 3) ).cuda()
input[0, :, :, 1] = x
input[0, :, :, 2] = y

# Analytic solution
k = 1
u = torch.sin(np.sqrt(3)/2*k*x) * torch.sin(1/2*k*y)

# Boundary conditions
g1 = 0*x
g2 = 0*x
h3 = np.sqrt(3)/2*k * np.cos(np.sqrt(3)/2*k) * torch.sin(1/2*k*y)
h4 = 1/2*k * torch.sin(np.sqrt(3)/2*k*x) * np.cos(1/4*k)
h5 = np.sin(np.sqrt(3)/4*k) * torch.sin(1/2*k*y) + np.sqrt(3)/2*k * np.cos(np.sqrt(3)/4*k) * torch.sin(1/2*k*y)
h6 = torch.sin(np.sqrt(3)/2*k*x) * np.sin(1/2*k) + 1/2*k * torch.sin(np.sqrt(3)/2*k*x) * np.cos(1/2*k)


ones_1n = torch.ones((1, res_size+2)).cuda()
ones_n1 = torch.ones((res_size+2, 1)).cuda()

losses = []
errors = []

epochs = 2000
start_time = time.time()
for i in range(epochs):
    optimizer.zero_grad()
    output = model(input)

    Psi = output[..., 0]
    Psi_boundary = (phi_2 * g1) / (phi_1 + phi_2) + (phi_1 * g2) / (phi_1 + phi_2) + phi_1 * phi_2 * output[..., 1]
    
    Psi_3 = ones_n1 * Psi_boundary[:,-2,:].reshape(Psi_boundary.shape[0],1,res_size+2)
    Psi_4 = Psi_boundary[:,:,int((res_size+1)/2)].reshape((Psi_boundary.shape[0],res_size+2,1)) * ones_1n
    Psi_5 = ones_n1 * Psi_boundary[:,int((res_size+1)/2),:].reshape(Psi_boundary.shape[0],1,res_size+2)
    Psi_6 = Psi_boundary[:,:,-2].reshape((Psi_boundary.shape[0],res_size+2,1)) * ones_1n

    u_1 = g1[1:-1,1:-1]
    u_2 = g2[1:-1,1:-1]
    u_3 = Psi_3[:,1:-1,1:-1] - phi_3_Tilde * h3[1:-1,1:-1]
    u_4 = Psi_4[:,1:-1,1:-1] - phi_4_Tilde * h4[1:-1,1:-1]
    u_5 = Psi_5[:,1:-1,1:-1] * (1 + phi_5_Tilde) - phi_5_Tilde * h5[1:-1,1:-1]
    u_6 = Psi_6[:,1:-1,1:-1] * (1 + phi_6_Tilde) - phi_6_Tilde * h6[1:-1,1:-1]
    
    pred = w_1 * u_1 + w_2 * u_2 + w_3 * u_3 + w_4 * u_4 + w_5 * u_5 + w_6 * u_6 + phi * Psi[:,1:-1,1:-1]

    pred_xx = (pred[:, 2::, 1:-1] - 2*pred[:, 1:-1, 1:-1] + pred[:, 0:-2, 1:-1]) / dx**2
    pred_yy = (pred[:, 1:-1, 2::] - 2*pred[:, 1:-1, 1:-1] + pred[:, 1:-1, 0:-2]) / dx**2

    loss = torch.nn.functional.mse_loss((pred_xx + pred_yy + k**2 * pred[:,1:-1,1:-1])*interior[2:-2,2:-2], torch.zeros(pred_xx.shape).cuda())
    error = torch.norm( (pred[0, 1:-1,1:-1] - u[2:-2,2:-2])*interior_boundary[2:-2,2:-2], 2 ) / torch.norm(u[2:-2,2:-2]*interior_boundary[2:-2,2:-2], 2)

    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())
    errors.append(error.item())

    print('epoch: ', i, ' loss: ', loss.item(), ' lr: ', scheduler.get_last_lr()[0], ' error:', error.item())

end_time = time.time()
print("Training time: ", end_time-start_time)

with open('boundary-conditions-for-pinos-code/results_Helmholtz/orthogonal_projections/losses_errors.npy', 'wb') as f:
    np.save(f, np.array(losses))
    np.save(f, np.array(errors))

with open('boundary-conditions-for-pinos-code/results_Helmholtz/orthogonal_projections/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

torch.save(model, 'boundary-conditions-for-pinos-code/results_Helmholtz/orthogonal_projections/net.pt')