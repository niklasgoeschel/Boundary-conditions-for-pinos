import numpy as np

nu = 0.0005*np.ones(100) + (0.1-0.0005)*np.random.rand(100)

with open('boundary-conditions-for-pinos-code/Navier_stokes_coeffs_nu.npy', 'wb') as f:
    np.save(f, nu)