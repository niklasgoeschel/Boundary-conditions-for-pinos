import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
from pino.after_training import save_model
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
from pino.losses import pde_loss, L2_error, bc_loss, semi_bc_loss
import time

from torch.utils.data import DataLoader
from pino.datasets import L_shape

def eval_L_shape(res_size,
                 ansatz_function,
                 path,
                 num=100,
                 offset=400):
    
    model = torch.load(path + '/net.pt', weights_only=False).cuda()

    model.eval()

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=1)

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

    errors = []
    losses = []

    all_predictions = np.zeros((num,res_size,res_size))
    all_solutions = np.zeros((num,res_size,res_size))
    i=0
    with torch.no_grad():
        eval_time = 0
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            start_time = time.time()
            output = model(input)

            pred = ansatz_function(output, g1, g2, h3, h4, h5, h6, w_1, w_2, w_3, w_4, w_5, w_6, phi, phi_1, phi_2, phi_3_Tilde, phi_4_Tilde, phi_5_Tilde, phi_6_Tilde, phi_B, phi_C, phi_D, phi_E, phi_F, res_size)
            end_time = time.time()
            eval_time = eval_time + end_time - start_time
            loss = pde_loss(pred, input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            error = L2_error(pred, u[:,1:-1,1:-1],res_size)

            losses.append(loss.item())
            errors.append(error.item())

            all_predictions[i, ...] = pred[0, ...].cpu()
            all_solutions[i, ...] = u[0, 1:-1, 1:-1].cpu()
            i=i+1
    with open(path + '/predicted_solutions.npy', 'wb') as f:
        np.save(f, np.array(losses))
        np.save(f, np.array(errors))
        np.save(f, all_predictions)
        np.save(f, all_solutions)
        np.save(f, eval_time)
        


    print(path, '  loss: ', np.mean(losses), ' L2-error: ', np.mean(errors))



def eval_L_shape_weak(res_size,
                 path,
                 lam=1,
                 num=100,
                 offset=400):
    
    model = torch.load(path + '/net.pt', weights_only=False).cuda()

    model.eval()

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=1)

    errors = []
    losses = []

    all_predictions = np.zeros((num,res_size,res_size))
    all_solutions = np.zeros((num,res_size,res_size))
    i=0
    with torch.no_grad():
        eval_time = 0
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            start_time = time.time()
            output = model(input)

            pred = output[..., 0]
            end_time = time.time()
            eval_time = eval_time + end_time - start_time

            loss_pde = pde_loss(pred[:, 1:-1, 1:-1], input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            loss_bc = bc_loss(pred, g1[:, 1:-1, 1:-1], g2[:, 1:-1, 1:-1], h3[:, 1:-1, 1:-1], h4[:, 1:-1, 1:-1], h5[:, 1:-1, 1:-1], h6[:, 1:-1, 1:-1], res_size)
            loss = loss_pde + lam* loss_bc
            error = L2_error(pred[:, 1:-1, 1:-1], u[:,1:-1,1:-1], res_size)

            losses.append(loss.item())
            errors.append(error.item())

            all_predictions[i, ...] = pred[0, 1:-1, 1:-1].cpu()
            all_solutions[i, ...] = u[0, 1:-1, 1:-1].cpu()
            i=i+1  
    with open(path + '/predicted_solutions.npy', 'wb') as f:
        np.save(f, np.array(losses))
        np.save(f, np.array(errors))
        np.save(f, all_predictions)
        np.save(f, all_solutions)
        np.save(f, eval_time)

    print(path, '  loss: ', np.mean(losses), ' L2-error: ', np.mean(errors))


def eval_L_shape_semi_weak(res_size,
                 path,
                 lam=1,
                 num=100,
                 offset=400):
    
    model = torch.load(path + '/net.pt', weights_only=False).cuda()

    model.eval()

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=1)

    dx = 1/(res_size-1)

    x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
    y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

    phi_1 = x
    phi_2 = y

    w_1 = phi_2 / (phi_1+phi_2)
    w_2 = phi_1 / (phi_1+phi_2)

    w_1[1,1] = 0.5
    w_2[1,1] = 0.5

    errors = []
    losses = []

    all_predictions = np.zeros((num,res_size,res_size))
    all_solutions = np.zeros((num,res_size,res_size))
    i=0
    with torch.no_grad():
        eval_time = 0
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            start_time = time.time()
            output = model(input)

            pred = w_1*g1 + w_2*g2 + phi_1*phi_2*output[..., 0]
            end_time = time.time()
            eval_time = eval_time + end_time - start_time

            loss_pde = pde_loss(pred[:, 1:-1, 1:-1], input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            loss_semi_bc = semi_bc_loss(pred, h3[:, 1:-1, 1:-1], h4[:, 1:-1, 1:-1], h5[:, 1:-1, 1:-1], h6[:, 1:-1, 1:-1], res_size)
            loss = loss_pde + lam * loss_semi_bc
            error = L2_error(pred[:, 1:-1, 1:-1], u[:,1:-1,1:-1], res_size)

            losses.append(loss.item())
            errors.append(error.item())

            all_predictions[i, ...] = pred[0, 1:-1, 1:-1].cpu()
            all_solutions[i, ...] = u[0, 1:-1, 1:-1].cpu()
            i=i+1
    
    with open(path + '/predicted_solutions.npy', 'wb') as f:
        np.save(f, np.array(losses))
        np.save(f, np.array(errors))
        np.save(f, all_predictions)
        np.save(f, all_solutions)
        np.save(f, eval_time)

    print(path, '  loss: ', np.mean(losses), ' L2-error: ', np.mean(errors))