import numpy as np
import json
import myplot
import sys

def main():
    name = "std_q"
    if len(sys.argv) > 1:
        name = sys.argv[1]

    with open('rewards/rewards_' + name + '.npy', 'rb') as reward_file:
        rewards = np.load(reward_file)

    with open('losses/loss_' + name + '.npy', 'rb') as loss_file:
        losses = np.load(loss_file)

    radius = int(len(rewards)/10)
    if len(sys.argv) > 2:
        r = int(sys.argv[2])
        if r > 0:
            radius = r

    display_name = name
    if len(sys.argv) > 3:
        display_name = " ".join(sys.argv[3:])

    myplot.plotRewards(display_name + " Learning",rewards,radius)

    myplot.plotLoss(display_name + " Learning", losses, radius)


if __name__ == "__main__":
    main()
