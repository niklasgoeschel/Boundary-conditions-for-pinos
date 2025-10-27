import numpy as np

alp = np.ones(500) + 9*np.random.rand(500)
bet = np.ones(500) + 9*np.random.rand(500)

with open('boundary-conditions-for-pinos-code/Darcy_flow_coeffs_alp_bet.npy', 'wb') as f:
    np.save(f, alp)
    np.save(f, bet)