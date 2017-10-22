import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plotRewards(agent, rewards, radius):
    averagedRewards = calculateAveragedRewards(rewards, radius)
    averagePrevRewards = calculateAveragedRewardsPrevious(rewards, radius)

    ax = plt.figure().gca()

    plt.plot(rewards, label="Reward per episode")
    plt.plot(averagedRewards,label="Averaged Reward (r={})".format(radius))
    plt.plot(averagePrevRewards, label="Averaged Reward last {} episodes".format(radius) )

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for {} agent'.format(agent))
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

def plotLoss(agent, rewards, radius):
    averagedRewards = calculateAveragedRewards(rewards, radius)
    averagePrevRewards = calculateAveragedRewardsPrevious(rewards, radius)

    ax = plt.figure().gca()

    plt.plot(rewards, label="Loss per episode")
    plt.plot(averagedRewards,label="Averaged Loss (r={})".format(radius))
    plt.plot(averagePrevRewards, label="Averaged Loss last {} episodes".format(radius) )

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode for {} agent'.format(agent))
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()



def calculateAveragedRewards(rewards, radius):
    averagedRewards = [np.mean(rewards[max(i-radius,0):min(i+radius,len(rewards))]) for i in range(len(rewards))]
    return averagedRewards

def calculateAveragedRewardsPrevious(rewards, radius):
    averagedRewards = [np.mean(rewards[max(i-radius,0):i]) for i in range(len(rewards))]
    return averagedRewards


def show_rgb(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()

# randomAgentRewards = [-37.7,-31.6,-34.9,-22.4,-35.6,-31.9,-45.1,-23.1,-35.1,-23.4,-29.9,-30.6,-33.1,-30.1,-37.5,-38.1,-33.8,-33.9,-28.8,-35.8,-31.7]
# plotRewards("Random", randomAgentRewards, 2)
# gasAgentRewards = [-52.01578947368444, -44.30465116279035, -56.26357615894072, -53.91541218638024, -62.06363636363665, -56.06357615894072, -52.239926739926965, -53.33505154639204, 41.67874564459976, 31.970648464164384, -54.51355932203423, 53.24301075268869, 39.018120805369676, -59.493399339934314, -57.41818181818214, -50.8669039145909, 54.194117647059386, -59.71052631578976, 28.80952380952445, -58.392307692307995, -47.14831460674142]
# plotRewards("Always Gas", randomAgentRewards, 2)
