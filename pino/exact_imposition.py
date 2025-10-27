import torch

def phi_lineseg(x,y,x_1,y_1,x_2,y_2):
    x_1 = x_1 * torch.ones(x.size()).cuda()
    y_1 = y_1 * torch.ones(y.size()).cuda()
    x_2 = x_2 * torch.ones(x.size()).cuda()
    y_2 = y_2 * torch.ones(y.size()).cuda()

    L = torch.sqrt( (x_2-x_1)**2 + (y_2-y_1)**2 )
    x_c = (x_1+x_2)/2
    y_c = (y_1+y_2)/2
    f = ( (x-x_1)*(y_2-y_1) - (y-y_1)*(x_2-x_1) ) / L
    t = ( (L/2)**2 - (x_c-x)**2 - (y_c-y)**2 ) / L
    varphi = torch.sqrt( t**2 + f**4 )
    return torch.sqrt( f**2 + ((varphi-t)/2)**2 )


def phi_quarter_circle(x,y,R,x_c,y_c,n_x,n_y,p_x,p_y):
    f = (R**2 - (x-x_c)**2 - (y-y_c)**2) / R / 2
    t = n_x * x + n_y * y - n_x * p_x - n_y * p_y
    varphi = torch.sqrt( t**2 + f**4 )
    return torch.sqrt( f**2 + ((varphi-t)/2)**2 )

