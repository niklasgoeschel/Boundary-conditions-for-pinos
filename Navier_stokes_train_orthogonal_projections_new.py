import torch
import numpy as np
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
import time


model = FNN2d(modes1=[20, 20, 20, 20],
              modes2=[20, 20, 20, 20],
              fc_dim=128,
              layers=[64, 64, 64, 64, 64],
              activation='gelu',
              out_dim=7).cuda()

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

phi_1 = x
phi_2 = y
phi_3 = (2.2 - x)**2
phi_4 = 0.41 - y
phi_S = torch.sqrt( (x-0.2)**2 + (y-0.2)**2 ) - 0.05

phi_1_extended = x_extended
phi_2_extended = y_extended
phi_4_extended = 0.41 - y_extended
phi_S_extended = torch.sqrt( (x_extended-0.2)**2 + (y_extended-0.2)**2 ) - 0.05

interior = torch.ones((res_size_x,res_size_y)).cuda()
for i in range(res_size_x):
    for j in range(res_size_y):
        if phi_S[i,j] <= 0:
            interior[i,j] = 0

phi_3_tilde = 2.2 - x

with open('boundary-conditions-for-pinos-code/fenics_reference_solution/DFG_2D_1/pred_441_83_3.npy', 'rb') as f:
    reference_solution = torch.tensor(np.load(f)).cuda()

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

losses = []
errors_u = []
errors_v = []
errors_p = []

epochs = 4000
start_time = time.time()
for i in range(epochs):
    optimizer.zero_grad()
    output = model(input)

    nu = 0.001
    sqrt_nu = np.sqrt(0.001)

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
    pred_p = Psi_p_bar[:, 1:-1, 1:-1] + w_3*(phi_3_tilde * output[:, 1:-1, 1:-1,6]) + phi_1 * phi_2 * phi_3 * phi_4 * phi_S * Psi_p 

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

    res_1 = -nu * ( pred_u_xx + pred_u_yy ) + sqrt_nu*pred_p_x + pred_u[:,1:-1,1:-1] * pred_u_x + pred_v[:,1:-1,1:-1] * pred_u_y 
    res_2 = -nu * ( pred_v_xx + pred_v_yy ) + sqrt_nu*pred_p_y + pred_u[:,1:-1,1:-1] * pred_v_x + pred_v[:,1:-1,1:-1] * pred_v_y 
    res_3 = pred_u_x + pred_v_y

    res_1 = res_1 * interior[1:-1,1:-1]
    res_2 = res_2 * interior[1:-1,1:-1]
    res_3 = res_3 * interior[1:-1,1:-1]

    loss = (torch.nn.functional.mse_loss(res_1, torch.zeros(res_1.shape).cuda()) + torch.nn.functional.mse_loss(res_2, torch.zeros(res_2.shape).cuda()) + torch.nn.functional.mse_loss(res_3, torch.zeros(res_3.shape).cuda()))

    error_u = torch.norm( (pred_u[0, ...] - reference_solution[...,0])*interior, 2 ) / torch.norm(reference_solution[..., 0]*interior, 2)
    error_v = torch.norm( (pred_v[0, ...] - reference_solution[...,1])*interior, 2 ) / torch.norm(reference_solution[..., 1]*interior, 2)
    error_p = torch.norm( (pred_p[0, ...]*sqrt_nu - reference_solution[...,2])*interior, 2 ) / torch.norm(reference_solution[..., 2]*interior, 2)

    losses.append(loss.item())
    errors_u.append(error_u.item())
    errors_v.append(error_v.item())
    errors_p.append(error_p.item())

    loss.backward()
    optimizer.step()
    scheduler.step()

    print('epoch: ', i, ' loss: ', loss.item(), ' lr: ', scheduler.get_last_lr()[0], ' error_u:', error_u.item(), ' error_v:', error_v.item(), ' error_p:', error_p.item())


end_time = time.time()
print("Training time: ", end_time-start_time)

""" with open('boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/losses_errors.npy', 'wb') as f:
    np.save(f, np.array(losses))
    np.save(f, np.array(errors_u))
    np.save(f, np.array(errors_v))
    np.save(f, np.array(errors_p))

predicted_solution = np.zeros((res_size_x,res_size_y,3))
predicted_solution[...,0] = pred_u.cpu().detach().numpy()
predicted_solution[...,1] = pred_v.cpu().detach().numpy()
predicted_solution[...,2] = pred_p.cpu().detach().numpy()

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/prediction_441_83_3.npy', 'wb') as f:
    np.save(f, predicted_solution)

with open('boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/runtime.npy', 'wb') as f:
    np.save(f, end_time-start_time)

torch.save(model, 'boundary-conditions-for-pinos-code/results_Navier_stokes/orthogonal_projections/net.pt') """