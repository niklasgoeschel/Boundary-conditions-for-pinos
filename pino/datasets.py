import torch
import numpy as np
from torch.utils.data import Dataset

    

class L_shape(Dataset):
    def __init__(self,
                 datapath_coeffs,
                 size,
                 num,
                 offset=0):
        with open(datapath_coeffs, 'rb') as f:
            alpha = np.load(f)
            beta = np.load(f)
        
        self.input = torch.zeros(num, size+2, size+2, 3, dtype=torch.float).cuda()
        self.u = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.f = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.g1 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.g2 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.h3 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.h4 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.h5 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()
        self.h6 = torch.zeros(num, size+2, size+2, dtype=torch.float).cuda()

        x = torch.Tensor([[(i-1)/(size-1) for j in range(size+2)] for i in range(size+2)] ).cuda()
        y = torch.Tensor([[(j-1)/(size-1) for j in range(size+2)] for i in range(size+2)] ).cuda()

        for i in range(num):
            alp = alpha[i+offset]
            bet = beta[i+offset]
            
            self.input[i, :, :, 0] = torch.sin( alp*x ) * torch.sin( bet*y )
            self.input[i, :, :, 1] = x
            self.input[i, :, :, 2] = y
            self.u[i,...] = torch.sin( alp*x ) * torch.cos( bet*y )
            self.g1[i,...] = x * 0
            self.g2[i,...] = torch.sin( alp*x ) 
            self.h3[i,...] = alp * np.cos( alp ) * torch.cos( bet*y )
            self.h4[i,...] = -bet * torch.sin( alp*x ) * np.sin( bet*0.5 )
            self.h5[i,...] = ( alp * np.cos( alp*0.5 ) + np.sin( alp*0.5 ) ) * torch.cos( bet*y )
            self.h6[i,...] = ( np.cos(bet) - bet*np.sin(bet) ) * torch.sin( alp*x ) 
            self.f[i,...] = - 0.5 * torch.sin( 2*bet*y ) * ( alp**2 * torch.cos(2*alp*x) + bet**2 * torch.cos(2*alp*x) - bet**2 )

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, item):
        return self.input[item], self.u[item], self.f[item], self.g1[item], self.g2[item], self.h3[item], self.h4[item], self.h5[item], self.h6[item]

