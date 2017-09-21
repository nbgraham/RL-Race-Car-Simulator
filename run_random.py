import gym
import myplot

env = gym.make('CarRacing-v0')

rewards=[]
for i_episode in range(21):
    observation = env.reset()
    sum_reward = 0
    for t in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        sum_reward += reward
        if(t%100 == 0): print(t)
        if done or t==999:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)

myplot.plotRewards("Random", rewards)
