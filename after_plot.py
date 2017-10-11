import numpy as np
import json
import myplot


def plot_luc():
    f = open('luc/rewards.npy', 'rb')
    #open('rewards/rewards')
    rewards = np.load(f) #json.load(f)

    myplot.plotRewards("Luc Prieur's Deep-Q Learning",rewards,100)

def plot_k_means(k):
    f = open('rewards/reward_' + str(k) + '.json' )
    rewards = json.load(f)

    myplot.plotRewards("K=" + str(k) + " means Q Learning",rewards,100)

if __name__ == "__main__":
    plot_k_means(20)
