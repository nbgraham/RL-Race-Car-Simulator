import numpy as np
import json
import myplot

f = open('luc/rewards.npy', 'rb') #open('rewards/rewards')
rewards = np.load(f) #json.load(f)

myplot.plotRewards("Luc Prieur's Deep-Q Learning",rewards,100)
