import torch

def ansatz_orthogonal_projections(output, g1, g2, h3, h4, h5, h6, w_1, w_2, w_3, w_4, w_5, w_6, phi, phi_1, phi_2, phi_3_Tilde, phi_4_Tilde, phi_5_Tilde, phi_6_Tilde, phi_B, phi_C, phi_D, phi_E, phi_F, res_size):
    Psi = output[..., 0]
    Psi_boundary = (phi_2 * g1) / (phi_1 + phi_2) + (phi_1 * g2) / (phi_1 + phi_2) + phi_1 * phi_2 * output[..., 1]

    ones_1n = torch.ones((1, res_size+2)).cuda()
    ones_n1 = torch.ones((res_size+2, 1)).cuda()

    Psi_3 = ones_n1 * Psi_boundary[:,-2,:].reshape(Psi_boundary.shape[0],1,res_size+2)
    Psi_4 = Psi_boundary[:,:,int((res_size+1)/2)].reshape((Psi_boundary.shape[0],res_size+2,1)) * ones_1n
    Psi_5 = ones_n1 * Psi_boundary[:,int((res_size+1)/2),:].reshape(Psi_boundary.shape[0],1,res_size+2)
    Psi_6 = Psi_boundary[:,:,-2].reshape((Psi_boundary.shape[0],res_size+2,1)) * ones_1n

    u_1 = g1[:,1:-1,1:-1]
    u_2 = g2[:,1:-1,1:-1]
    u_3 = Psi_3[:,1:-1,1:-1] - phi_3_Tilde * h3[:,1:-1,1:-1]
    u_4 = Psi_4[:,1:-1,1:-1] - phi_4_Tilde * h4[:,1:-1,1:-1]
    u_5 = Psi_5[:,1:-1,1:-1] * (1 + phi_5_Tilde) - phi_5_Tilde * h5[:,1:-1,1:-1]
    u_6 = Psi_6[:,1:-1,1:-1] * (1 + phi_6_Tilde) - phi_6_Tilde * h6[:,1:-1,1:-1]
    
    return w_1 * u_1 + w_2 * u_2 + w_3 * u_3 + w_4 * u_4 + w_5 * u_5 + w_6 * u_6 + phi * Psi[:,1:-1,1:-1]




def ansatz_continous_different_phis(output, g1, g2, h3, h4, h5, h6, w_1, w_2, w_3, w_4, w_5, w_6, phi, phi_1, phi_2, phi_3_Tilde, phi_4_Tilde, phi_5_Tilde, phi_6_Tilde, phi_B, phi_C, phi_D, phi_E, phi_F, res_size):
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
    
    return w_1 * u_1 + w_2 * u_2 + w_3 * u_3 + w_4 * u_4 + w_5 * u_5 + w_6 * u_6 + phi * Psi[:,1:-1,1:-1]