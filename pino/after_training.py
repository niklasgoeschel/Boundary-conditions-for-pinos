import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def save_model(model, path, epochs, losses_each_epoch, errors_each_epoch, lr, starting_epoch):
    torch.save(model, path + '/net.pt')

    with open(path + '/losses_errors.npy', 'wb') as f:
        np.save(f, np.array([starting_epoch+i for i in range(epochs)]))
        np.save(f, losses_each_epoch)
        np.save(f, errors_each_epoch)
        np.save(f, lr)

