import numpy as np

from base.model import BaseModel
from std_q.hyperparameters import gamma, action_selection_coeff, alpha


class Model(BaseModel):
    def __init__(self, env, name, input_size, action_set):
        BaseModel.__init__(self, env, name, input_size, action_set, alpha=alpha, gamma=gamma)

    def get_action_selection_parameter(cur_episode, total_episodes):
        return 5/np.sqrt(cur_episode+25)-0.1