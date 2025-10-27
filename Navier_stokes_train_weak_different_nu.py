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
              out_dim=3).cuda()

optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                 lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[500,1000,1500,2000,2500,3000,3500],
                                                gamma=0.5)

model.train()

res_size_x = 441
res_size_y = 83

dx = 2.2 / (res_size_x-1)
dy = 0.41 / (res_size_y-1)

x = torch.Tensor([[i*dx for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()
y = torch.Tensor([[j*dy for j in range(res_size_y)] for i in range(res_size_x)] ).cuda()

x_extended = torch.Tensor([[(i-1)*dx for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()
y_extended = torch.Tensor([[(j-1)*dy for j in range(res_size_y+2)] for i in range(res_size_x+2)] ).cuda()

phi_S = torch.sqrt( (x_extended-0.2)**2 + (y_extended-0.2)**2 ) - 0.05

interior = torch.ones((res_size_x,res_size_y)).cuda()
for i in range(res_size_x):
    for j in range(res_size_y):
        if phi_S[i+1,j+1] <= 0:
            interior[i,j] = 0

with open('boundary-conditions-for-pinos-code/fenics_reference_solution/DFG_2D_1_different_nu/pred_100_441_83_3.npy', 'rb') as f:
    reference_solutions = torch.tensor(np.load(f)[0:80,...]).cuda()

with open('boundary-conditions-for-pinos-code/Navier_stokes_coeffs_nu.npy', 'rb') as f:
    nus = torch.tensor(np.load(f)[0:80], dtype=torch.float).cuda()

n_col_circle = int(0.1 * np.pi / dx)
weights_circle = torch.zeros((n_col_circle,10,res_size_x,res_size_y)).cuda()
for n in range(n_col_circle):
    x_col = 0.2 + 0.05*np.cos(2*np.pi*n/n_col_circle)
    y_col = 0.2 + 0.05*np.sin(2*np.pi*n/n_col_circle)
    i = int(x_col/dx)
    j = int(y_col/dy)
    alpha = x_col/dx - i
    beta = y_col/dy - j
    weights_circle[n,:,i,j] = (1-alpha) * (1-beta)
    weights_circle[n,:,i+1,j] = alpha * (1-beta)
    weights_circle[n,:,i,j+1] = (1-alpha) * beta
    weights_circle[n,:,i+1,j+1] = alpha * beta

U = 0.3
g_1 = torch.zeros( (res_size_x+2,res_size_y+2,2) ).cuda()
g_1[..., 0] = 4*U*y_extended*(0.41-y_extended)/0.41**2

ones_x_y_10 = torch.ones( (res_size_x+2, res_size_y+2, 10) ).cuda()

input = torch.ones( (10, res_size_x+2, res_size_y+2, 3) ).cuda()
input[:, :, :, 1] = x_extended
input[:, :, :, 2] = y_extended

losses = []
losses_pde = []
losses_bc = []
errors_u = []
errors_v = []
errors_p = []

epochs = 4000
start_time = time.time()
for i in range(epochs):
    rand_indices = torch.randperm(80)
    rand_nus = nus[rand_indices]
    
    rand_sqrt_nus = torch.sqrt(rand_nus)
    
    losses_batch = []
    losses_pde_batch = []
    losses_bc_batch = []
    errors_u_batch = []
    errors_v_batch = []
    errors_p_batch = []

    for j in range(8):
        input[:, :, :, 0] = (rand_nus[10*j:10*(j+1)] * ones_x_y_10).permute(2,0,1)
        
        optimizer.zero_grad()
        output = model(input)
        
        sqrt_nu = (rand_sqrt_nus[10*j:10*(j+1)] * ones_x_y_10).permute(2,0,1)
        
        Psi_u = output[..., 0]
        Psi_v = output[..., 1]
        Psi_p = output[..., 2]

        pred_u = Psi_u
        pred_v = Psi_v
        pred_p = Psi_p

        pred_u_x_3 = (pred_u[:,-1,2:-2] - pred_u[:,-3,2:-2]) / dx / 2
        pred_v_x_3 = (pred_v[:,-1,2:-2] - pred_v[:,-3,2:-2]) / dx / 2

        pred_u = pred_u[:,1:-1,1:-1]
        pred_v = pred_v[:,1:-1,1:-1]
        pred_p = pred_p[:,1:-1,1:-1]

        pred_u_x = (pred_u[:, 2::, 1:-1] - pred_u[:, 0:-2, 1:-1]) / dx / 2
        pred_u_y = (pred_u[:, 1:-1, 2::] - pred_u[:, 1:-1, 0:-2]) / dy / 2
        pred_v_x = (pred_v[:, 2::, 1:-1] - pred_v[:, 0:-2, 1:-1]) / dx / 2
        pred_v_y = (pred_v[:, 1:-1, 2::] - pred_v[:, 1:-1, 0:-2]) / dy / 2
        pred_u_xx = (pred_u[:, 2::, 1:-1] - 2*pred_u[:, 1:-1, 1:-1] + pred_u[:, 0:-2, 1:-1]) / dx**2
        pred_u_yy = (pred_u[:, 1:-1, 2::] - 2*pred_u[:, 1:-1, 1:-1] + pred_u[:, 1:-1, 0:-2]) / dy**2
        pred_v_xx = (pred_v[:, 2::, 1:-1] - 2*pred_v[:, 1:-1, 1:-1] + pred_v[:, 0:-2, 1:-1]) / dx**2
        pred_v_yy = (pred_v[:, 1:-1, 2::] - 2*pred_v[:, 1:-1, 1:-1] + pred_v[:, 1:-1, 0:-2]) / dy**2
        pred_p_x = (pred_p[:, 2::, 1:-1] - pred_p[:, 0:-2, 1:-1]) / dx / 2
        pred_p_y = (pred_p[:, 1:-1, 2::] - pred_p[:, 1:-1, 0:-2]) / dy / 2

        res_1 = -input[:, 2:-2, 2:-2, 0] * ( pred_u_xx + pred_u_yy ) + sqrt_nu[:,2:-2, 2:-2]*pred_p_x + pred_u[:,1:-1,1:-1] * pred_u_x + pred_v[:,1:-1,1:-1] * pred_u_y 
        res_2 = -input[:, 2:-2, 2:-2, 0] * ( pred_v_xx + pred_v_yy ) + sqrt_nu[:,2:-2, 2:-2]*pred_p_y + pred_u[:,1:-1,1:-1] * pred_v_x + pred_v[:,1:-1,1:-1] * pred_v_y 
        res_3 = pred_u_x + pred_v_y
        
        res_1 = res_1 * interior[1:-1,1:-1]
        res_2 = res_2 * interior[1:-1,1:-1]
        res_3 = res_3 * interior[1:-1,1:-1]

        loss_pde = (torch.nn.functional.mse_loss(res_1, torch.zeros(res_1.shape).cuda()) + torch.nn.functional.mse_loss(res_2, torch.zeros(res_2.shape).cuda()) + torch.nn.functional.mse_loss(res_3, torch.zeros(res_3.shape).cuda()))
        
        res_circle_u = torch.sum(weights_circle * pred_u, dim=[2,3]).reshape((n_col_circle*10))
        res_circle_v = torch.sum(weights_circle * pred_v, dim=[2,3]).reshape((n_col_circle*10))
        res_u_bc = torch.cat(( res_circle_u, (pred_u[:,0,:]-g_1[0,1:-1, 0]).reshape((res_size_y*10)), pred_u[:,1::,0].reshape(((res_size_x-1)*10)), pred_u[:,1::,-1].reshape(((res_size_x-1)*10)) ))
        res_v_bc = torch.cat(( res_circle_v, pred_v[:,0,:].reshape((res_size_y*10)), pred_v[:,1::,0].reshape(((res_size_x-1)*10)), pred_v[:,1::,-1].reshape(((res_size_x-1)*10)) ))
        res_robin = torch.cat(( sqrt_nu[:, -2, 2:-2]*pred_u_x_3 - pred_p[:,-1,1:-1], pred_v_x_3 ))
        loss_bc = torch.nn.functional.mse_loss( res_u_bc, torch.zeros(res_u_bc.shape).cuda() ) + torch.nn.functional.mse_loss( res_v_bc, torch.zeros(res_v_bc.shape).cuda() ) + torch.nn.functional.mse_loss( res_robin, torch.zeros(res_robin.shape).cuda() )
        
        loss = loss_pde + loss_bc
        
        error_u = torch.mean( torch.norm( (pred_u - reference_solutions[rand_indices[10*j:10*(j+1)],:,:,0])*interior, 2, dim=(1,2) ) / torch.norm(reference_solutions[rand_indices[10*j:10*(j+1)],:,:, 0]*interior, 2, dim=(1,2)) )
        error_v = torch.mean( torch.norm( (pred_v - reference_solutions[rand_indices[10*j:10*(j+1)],:,:,1])*interior, 2, dim=(1,2) ) / torch.norm(reference_solutions[rand_indices[10*j:10*(j+1)],:,:, 1]*interior, 2, dim=(1,2)) )
        error_p = torch.mean( torch.norm( (pred_p*sqrt_nu[:,1:-1,1:-1] - reference_solutions[rand_indices[10*j:10*(j+1)],:,:,2])*interior, 2, dim=(1,2) ) / torch.norm(reference_solutions[rand_indices[10*j:10*(j+1)],:,:, 2]*interior, 2, dim=(1,2)) )
        
        losses_batch.append(loss.item())
        losses_pde_batch.append(loss_pde.item())
        losses_bc_batch.append(loss_bc.item())
        errors_u_batch.append(error_u.item())
        errors_v_batch.append(error_v.item())
        errors_p_batch.append(error_p.item())

        loss.backward()
        optimizer.step()

    scheduler.step()

    losses.append(np.mean(losses_batch))
    losses_pde.append(np.mean(losses_pde_batch))
    losses_bc.append(np.mean(losses_bc_batch))
    errors_u.append(np.mean(errors_u_batch))
    errors_v.append(np.mean(errors_v_batch))
    errors_p.append(np.mean(errors_p_batch))
    

    print('epoch: ', i, ' loss: ', losses[-1], ' lr: ', scheduler.get_last_lr()[0], ' error_u:', errors_u[-1], ' error_v:', errors_v[-1], ' error_p:', errors_p[-1])

end_time = time.time()
print("Training time: ", end_time-start_time)

##############################################################
""" interior_plot = torch.ones((res_size_x,res_size_y)).cuda()
for i in range(res_size_x):
    for j in range(res_size_y):
        if phi_S[i,j] <= 0:
            interior_plot[i,j] = np.nan

velocity = torch.sqrt( pred_u**2 + pred_v**2 ) * interior_plot

fig, ax = plt.subplots()
ax.set_aspect('equal')
c = ax.pcolormesh(np.array([2.2*i/(res_size_x-1) for i in range(res_size_x+1)]), np.array([0.41*i/(res_size_y-1) for i in range(res_size_y+1)]), np.transpose(velocity[0, ...].cpu().detach().numpy()), cmap = 'viridis')
fig.colorbar(c, ax = ax)
ax.set_title('Velocity')
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal')
c = ax.pcolormesh(np.array([2.2*i/(res_size_x-1) for i in range(res_size_x+1)]), np.array([0.41*i/(res_size_y-1) for i in range(res_size_y+1)]), np.transpose((interior_plot * pred_p[0, ...]*sqrt_nu).cpu().detach().numpy()), cmap = 'viridis')
fig.colorbar(c, ax = ax)
ax.set_title('Pressure')
plt.show() """
##############################################################

with open('boundary-conditions-for-pinos-code/results_Navier_stokes_different_nu/weak/losses_errors.npy', 'wb') as f:
    np.save(f, np.array(losses))
    np.save(f, np.array(errors_u))
    np.save(f, np.array(errors_v))
    np.save(f, np.array(errors_p))

with open('boundary-conditions-for-pinos-code/results_Navier_stokes_different_nu/weak/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

torch.save(model, 'boundary-conditions-for-pinos-code/results_Navier_stokes_different_nu/weak/net.pt')