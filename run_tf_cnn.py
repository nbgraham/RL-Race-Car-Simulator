import gym
from deep_qnet import *
from gym import wrappers

env = gym.make('CarRacing-v0')
#repeats an action for 4 frames with env.step(action)
every_four_frames = wrappers.SkipWrapper(4)
env = every_four_frames(env)




