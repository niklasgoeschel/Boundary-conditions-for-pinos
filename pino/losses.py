import torch

def pde_residual(u,a,f,res_size):
    # u, a, f : (batch_size,res_size,res_size)
    dx = 1 / (res_size-1)

    u_x, u_y = (u[:, 2::, 1:-1] - u[:, 0:-2, 1:-1]) / dx / 2, (u[:, 1:-1, 2::] - u[:, 1:-1, 0:-2]) / dx / 2
    a_pred_x, a_pred_y = u_x * a[:,1:-1,1:-1], u_y * a[:,1:-1,1:-1]
    a_pred_xx, a_pred_yy = (a_pred_x[:, 2::, 1:-1] - a_pred_x[:, 0:-2, 1:-1]) / dx / 2, (a_pred_y[:, 1:-1, 2::] - a_pred_y[:, 1:-1, 0:-2]) / dx / 2
    return - ( a_pred_xx + a_pred_yy ) - f[:,2:-2,2:-2]

def pde_loss(u, a, f, res_size):
    # u, a, f : (batch_size,res_size,res_size)
   
    residual = pde_residual(u,a,f,res_size)

    L_geometry = torch.ones((res_size-4, res_size-4)).cuda()
    L_geometry[-int(res_size/2)::, -int(res_size/2)::] = 0

    return torch.mean( torch.norm(residual*L_geometry, 2, dim=[1,2]) / torch.norm(L_geometry, 2) )
    #return torch.nn.functional.mse_loss(residual*L_geometry, torch.zeros(residual.shape).cuda()) * 4 / 3


def bc_loss(u_, g1, g2, h3, h4, h5, h6, res_size):
    #u_ : (batch_size,res_size+2,res_size+2)
    #g1, g2, h3, h4, h5, h6 : (batch_size,res_size,res_size)
    dx = 1 / (res_size-1)

    u_x, u_y = (u_[:, 2::, 1:-1] - u_[:, 0:-2, 1:-1]) / dx / 2, (u_[:, 1:-1, 2::] - u_[:, 1:-1, 0:-2]) / dx / 2
    u = u_[:,1:-1,1:-1]
    residual_1 = u[:,0,:] - g1[:,0,:]
    residual_2 = u[:,:,0] - g2[:,:,0]
    residual_3 = u_x[:, -1, 0:int(res_size/2)+1] - h3[:, -1, 0:int(res_size/2)+1]
    residual_4 = u_y[:, -(int(res_size/2)+1)::, int(res_size/2)] - h4[:, -(int(res_size/2)+1)::, int(res_size/2)]
    residual_5 = u_x[:, int(res_size/2), -(int(res_size/2)+1)::] + u[:, int(res_size/2), -(int(res_size/2)+1)::] - h5[:, int(res_size/2), -(int(res_size/2)+1)::]
    residual_6 = u_y[:, 0:int(res_size/2)+1, -1] + u[:, 0:int(res_size/2)+1, -1] - h6[:, 0:int(res_size/2)+1, -1]
    
    residual_bc = torch.cat( (residual_1, residual_2, residual_3, residual_4, residual_5, residual_6), dim=1 )
    return torch.mean( torch.norm( residual_bc, 2, dim=1 ) / torch.norm( torch.ones( (residual_bc.shape[1]) ) ) )
    #return torch.nn.functional.mse_loss(residual_bc, torch.zeros(residual_bc.shape).cuda())


def semi_bc_loss(u_, h3, h4, h5, h6, res_size):
    #u_ : (batch_size,res_size+2,res_size+2)
    #h3, h4, h5, h6 : (batch_size,res_size,res_size)
    dx = 1 / (res_size-1)

    u_x, u_y = (u_[:, 2::, 1:-1] - u_[:, 0:-2, 1:-1]) / dx / 2, (u_[:, 1:-1, 2::] - u_[:, 1:-1, 0:-2]) / dx / 2
    u = u_[:,1:-1,1:-1]
    residual_3 = u_x[:, -1, 1:int(res_size/2)+1] - h3[:, -1, 1:int(res_size/2)+1]
    residual_4 = u_y[:, -(int(res_size/2)+1)::, int(res_size/2)] - h4[:, -(int(res_size/2)+1)::, int(res_size/2)]
    residual_5 = u_x[:, int(res_size/2), -(int(res_size/2)+1)::] + u[:, int(res_size/2), -(int(res_size/2)+1)::] - h5[:, int(res_size/2), -(int(res_size/2)+1)::]
    residual_6 = u_y[:, 1:int(res_size/2)+1, -1] + u[:, 1:int(res_size/2)+1, -1] - h6[:, 1:int(res_size/2)+1, -1]
    
    residual_bc = torch.cat( (residual_3, residual_4, residual_5, residual_6), dim=1 )
    return torch.mean( torch.norm( residual_bc, 2, dim=1 ) / torch.norm( torch.ones( (residual_bc.shape[1]) ) ) )



def L2_error(pred, u, res_size):
    diff = u - pred
    diff[:, -int(res_size/2)::, -int(res_size/2)::] = 0
    u[:, -int(res_size/2)::, -int(res_size/2)::] = 0
    return torch.mean( torch.norm(diff, 2, dim=[1,2]) / torch.norm(u, 2, dim=[1,2]) )