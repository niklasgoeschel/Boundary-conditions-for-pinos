import torch
import numpy as np
import matplotlib.pyplot as plt
from pino.exact_imposition import phi_lineseg
from pino.after_training import save_model
import math
from pino.fourier2d import FNN2d
from pino.adam import Adam
from pino.losses import pde_loss, L2_error, bc_loss, semi_bc_loss

from torch.utils.data import DataLoader
from pino.datasets import L_shape



from matplotlib import cm

def train_L_shape(epochs, 
                  res_size,
                  ansatz_function,
                  out_dim,
                  num,
                  batch_size,
                  offset=0,
                  epochs_per_milestone=100,
                  save=False,
                  checkpoint=False,
                  return_errors_losses=False,
                  lr=False,
                  anchor=False):
    
    if checkpoint == False:
        model = FNN2d(modes1=[20, 20, 20, 20],
                      modes2=[20, 20, 20, 20],
                      fc_dim=128,
                      layers=[64, 64, 64, 64, 64],
                      activation='gelu',
                      out_dim=out_dim).cuda()
        
        if lr==False:
            lr = 0.005
        starting_epoch = 0
    else:
        model = torch.load(checkpoint + '/net.pt', weights_only=False).cuda()
        with open(checkpoint + '/losses_errors.npy', 'rb') as f:
            epochs_prev = np.load(f)
            starting_epoch = epochs_prev[-1]+1
            np.load(f)
            np.load(f)
            if lr==False:
                lr = np.load(f)[0] / 2

    model.train()

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[epochs_per_milestone*(i+1) for i in range(int(epochs/epochs_per_milestone) - 1)],
                                                     gamma=0.5)

    dx = 1/(res_size-1)

    x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
    y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

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

    losses_each_epoch = []
    errors_each_epoch = []
    for i in range(epochs):
        losses = []
        errors = []
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            optimizer.zero_grad()
            output = model(input)

            pred = ansatz_function(output, g1, g2, h3, h4, h5, h6, w_1, w_2, w_3, w_4, w_5, w_6, phi, phi_1, phi_2, phi_3_Tilde, phi_4_Tilde, phi_5_Tilde, phi_6_Tilde, phi_B, phi_C, phi_D, phi_E, phi_F, res_size)
            
            error = L2_error(pred, u[:,1:-1,1:-1],res_size)

            if anchor==False:
                loss = pde_loss(pred, input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            elif i==0:
                anchor_pred = pred.detach()
                loss = pde_loss(pred, input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            else:
                loss = pde_loss(pred, input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size) + anchor * L2_error(pred, anchor_pred, res_size)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            errors.append(error.item())

        loss_each_epoch = np.mean(losses)
        error_each_epoch = np.mean(errors)
        print("Epoche:", i+starting_epoch, " loss:", loss_each_epoch, " L2-error:", error_each_epoch)
        losses_each_epoch.append(loss_each_epoch)
        errors_each_epoch.append(error_each_epoch)
        scheduler.step()

    if save==False:
        pass
    else:
        save_model(model, save, epochs, losses_each_epoch, errors_each_epoch, scheduler.get_last_lr(), starting_epoch)

    if return_errors_losses:
        return losses_each_epoch, errors_each_epoch




def train_L_shape_weak(epochs, 
                       res_size,
                       num,
                       batch_size,
                       offset=0,
                       epochs_per_milestone=100,
                       lam=1,
                       save=False,
                       checkpoint=False,
                       return_errors_losses=False,
                       lr=False,
                       anchor=False):
    
    if checkpoint == False:
        model = FNN2d(modes1=[20, 20, 20, 20],
                      modes2=[20, 20, 20, 20],
                      fc_dim=128,
                      layers=[64, 64, 64, 64, 64],
                      activation='gelu',
                      out_dim=1).cuda()
        
        if lr==False:
            lr = 0.005
        starting_epoch = 0
    else:
        model = torch.load(checkpoint + '/net.pt', weights_only=False).cuda()
        with open(checkpoint + '/losses_errors.npy', 'rb') as f:
            epochs_prev = np.load(f)
            starting_epoch = epochs_prev[-1]+1
            np.load(f)
            np.load(f)
            if lr==False:
                lr = np.load(f)[0] / 2

    model.train()

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[epochs_per_milestone*(i+1) for i in range(int(epochs/epochs_per_milestone) - 1)],
                                                     gamma=0.5)

    dx = 1/(res_size-1)

    x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
    y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    losses_each_epoch = []
    errors_each_epoch = []
    for i in range(epochs):
        losses = []
        errors = []
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            optimizer.zero_grad()
            output = model(input)

            pred = output[..., 0]

            loss_pde = pde_loss(pred[:, 1:-1, 1:-1], input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            loss_bc = bc_loss(pred, g1[:, 1:-1, 1:-1], g2[:, 1:-1, 1:-1], h3[:, 1:-1, 1:-1], h4[:, 1:-1, 1:-1], h5[:, 1:-1, 1:-1], h6[:, 1:-1, 1:-1], res_size)
            
            error = L2_error(pred[:, 1:-1, 1:-1], u[:,1:-1,1:-1], res_size)

            if anchor==False:
                loss = loss_pde + lam* loss_bc
            elif i==0:
                anchor_pred = pred
                loss = loss_pde + lam* loss_bc
            else:
                loss = loss_pde + lam* loss_bc + anchor * L2_error(pred, anchor_pred, res_size)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            errors.append(error.item())

        loss_each_epoch = np.mean(losses)
        error_each_epoch = np.mean(errors)
        print("Epoche:", i+starting_epoch, " loss:", loss_each_epoch, " L2-error:", error_each_epoch)
        losses_each_epoch.append(loss_each_epoch)
        errors_each_epoch.append(error_each_epoch)
        scheduler.step()
    
    if save==False:
        pass
    else:
        save_model(model, save, epochs, losses_each_epoch, errors_each_epoch, scheduler.get_last_lr(), starting_epoch)

    if return_errors_losses:
        return losses_each_epoch, errors_each_epoch


def train_L_shape_semi_weak(epochs, 
                       res_size,
                       num,
                       batch_size,
                       offset=0,
                       epochs_per_milestone=100,
                       lam=1,
                       save=False,
                       checkpoint=False,
                       return_errors_losses=False,
                       lr=False,
                       anchor=False):
    
    if checkpoint == False:
        model = FNN2d(modes1=[20, 20, 20, 20],
                      modes2=[20, 20, 20, 20],
                      fc_dim=128,
                      layers=[64, 64, 64, 64, 64],
                      activation='gelu',
                      out_dim=1).cuda()
        if lr==False:
            lr = 0.005
        starting_epoch = 0
    else:
        model = torch.load(checkpoint + '/net.pt', weights_only=False).cuda()
        with open(checkpoint + '/losses_errors.npy', 'rb') as f:
            epochs_prev = np.load(f)
            starting_epoch = epochs_prev[-1]+1
            np.load(f)
            np.load(f)
            if lr==False:
                lr = np.load(f)[0] / 2

    model.train()

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[epochs_per_milestone*(i+1) for i in range(int(epochs/epochs_per_milestone) - 1)],
                                                     gamma=0.5)

    dx = 1/(res_size-1)

    x = torch.Tensor( [[(i-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()
    y = torch.Tensor( [[(j-1)*dx for j in range(res_size+2)] for i in range(res_size+2)] ).cuda()

    phi_1 = x
    phi_2 = y

    w_1 = phi_2 / (phi_1+phi_2)
    w_2 = phi_1 / (phi_1+phi_2)

    w_1[1,1] = 0.5
    w_2[1,1] = 0.5

    dataset = L_shape('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', size=res_size, num=num, offset=offset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    losses_each_epoch = []
    errors_each_epoch = []
    for i in range(epochs):
        losses = []
        errors = []
        for input, u, f, g1, g2, h3, h4, h5, h6 in dataloader:
            optimizer.zero_grad()
            output = model(input)

            pred = w_1*g1 + w_2*g2 + phi_1*phi_2*output[..., 0]

            loss_pde = pde_loss(pred[:, 1:-1, 1:-1], input[:,1:-1,1:-1,0], f[:,1:-1,1:-1], res_size)
            loss_semi_bc = semi_bc_loss(pred, h3[:, 1:-1, 1:-1], h4[:, 1:-1, 1:-1], h5[:, 1:-1, 1:-1], h6[:, 1:-1, 1:-1], res_size)
            
            error = L2_error(pred[:, 1:-1, 1:-1], u[:,1:-1,1:-1], res_size)

            if anchor==False:
                loss = loss_pde + lam* loss_semi_bc
            elif i==0:
                anchor_pred = pred
                loss = loss_pde + lam* loss_semi_bc
            else:
                loss = loss_pde + lam* loss_semi_bc + anchor * L2_error(pred, anchor_pred, res_size)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            errors.append(error.item())

        loss_each_epoch = np.mean(losses)
        error_each_epoch = np.mean(errors)
        print("Epoche:", i+starting_epoch, " loss:", loss_each_epoch, " L2-error:", error_each_epoch)
        losses_each_epoch.append(loss_each_epoch)
        errors_each_epoch.append(error_each_epoch)
        scheduler.step()

    if save==False:
        pass
    else:
        save_model(model, save, epochs, losses_each_epoch, errors_each_epoch, scheduler.get_last_lr(), starting_epoch)

    if return_errors_losses:
        return losses_each_epoch, errors_each_epoch