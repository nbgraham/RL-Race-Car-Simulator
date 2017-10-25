import numpy as np

from base.model import BaseModel
from std_q.hyperparameters import gamma, action_selection_coeff, alpha


class Model(BaseModel):
    def get_action(self, state, eps, reward):
        argmax_qvals, qvals = self.sample_action(state, eps)

        action = Model.convert_argmax_qval_to_env_action(argmax_qvals)
        change = 0
        
        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma*np.max(qvals)
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] = (1-alpha)*y[self.prev_argmax] + alpha*change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = qvals
        self.prev_argmax = argmax_qvals

        loss = change**2
        return action, loss

    def get_action_selection_parameter(cur_episode, total_episodes):
        return 5/np.sqrt(cur_episode+25)-0.1