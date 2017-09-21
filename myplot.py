import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plotRewards(agent, rewards, radius):
    averagedRewards = calculateAveragedRewards(rewards, radius)

    ax = plt.figure().gca()

    plt.plot(range(len(rewards)), rewards, label="Reward")
    plt.plot(range(len(averagedRewards)), averagedRewards,label="Averaged Reward (r={})".format(radius))

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for {} agent'.format(agent))
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

def calculateAveragedRewards(rewards, radius):
    averagedRewards = []
    for i in range(len(rewards)):
        left = i - radius
        if left < 0: left = 0
        right = i + radius
        if right > len(rewards): right = len(rewards)
        selection = rewards[left:right+1]
        average = sum(selection)/len(selection)
        averagedRewards.append(average)
    return averagedRewards

# randomAgentRewards = [-37.7,-31.6,-34.9,-22.4,-35.6,-31.9,-45.1,-23.1,-35.1,-23.4,-29.9,-30.6,-33.1,-30.1,-37.5,-38.1,-33.8,-33.9,-28.8,-35.8,-31.7]
