import numpy as np
from softmax.hyperparameters import eps_coeff

def get_eps(episode_n):
    return eps_coeff/(episode_n+1)**(.7)
