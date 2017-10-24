import math
import numpy as np
import random


def softmax_select(qval, temp):
    prob = [math.exp(q/temp) for q in qval]
    prob = prob / np.sum(prob)

    softmax_selection_index = np.random.choice(range(len(qval)), p=prob)
    return softmax_selection_index


def eps_select(qval, eps):
    eps_selection_index = random.randint(0,10) if np.random.random() < eps else np.argmax(qval)
    return eps_selection_index