import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plotRewards(agent, rewards):
    ax = plt.figure().gca()

    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for {} agent'.format(agent))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

def plotAveragedRewards(agent, rewards, radius):
    averagedRewards = calculateAveragedRewards(rewards, radius)

    ax = plt.figure().gca()

    plt.plot(range(len(averagedRewards)), averagedRewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward Averaged over radius of {}'.format(radius))
    plt.title('Reward per Episode for {} agent'.format(agent))

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
