import numpy as np
from luc.hyperparameters import eps_coeff

def get_eps(episode_n):
    return eps_coeff/np.sqrt(episode_n+1 + 900)
