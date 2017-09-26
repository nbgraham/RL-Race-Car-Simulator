import gym
import myplot

render = False
n_episodes=1
env = gym.make('CarRacing-v0')

rewards=[]
for i_episode in range(n_episodes):
    observation = env.reset()
    sum_reward = 0
    for t in range(1000):
        if render: env.render()
        action = env.action_space.sample() # [steering, gas, brake]
        observation, reward, done, info = env.step(action) # observation is 96x96x3
        print(len(observation))
        print(len(observation[0]))
        print(len(observation[0][0]))
        print(observation)
        break
        sum_reward += reward
        if(t%100 == 0): print(t)
        if done or t==999:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break;

#print (rewards)
myplot.plotRewards("Random", rewards, 1)
