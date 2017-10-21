import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
from deep_qnet import *
from gym import wrappers
import tensorflow as tf

action_set = np.array([
#nothing
[0,0,0],
#steering (left,right)
[-1.0,0,0],
[1.0,0,0],
#gas
[0,1.0,0],
#brake
[0,0,0.5]
])

num_episodes = 1000
min_epsilon = 0.05
update_time_steps = 100
max_time_steps = 1500

gamma = 0.99

dqn = DQN(action_set)

env = gym.make('CarRacing-v0')
#repeats an action for 4 frames with env.step(action)
every_four_frames = wrappers.SkipWrapper(4)
env = every_four_frames(env)

with tf.Session() as sess:
    sess.run(dqn.currentQNet.init)
    sess.run(dqn.targetQNet.init)

    totalrewards = np.empty(num_episodes)
    totallosses = np.empty(num_episodes)
    for episode in range(num_episodes):
        # decrease epsilon every episode
        if (dqn.epsilon > min_epsilon):
            dqn.epsilon -= dqn.epsilon/np.sqrt(episode+1 + 900)
            if dqn.epsilon < min_epsilon:
                dqn.epsilon = min_epsilon

        # reset for each episode
        observation = env.reset()
        state = pre.above_bar_no_resize(observation)
        state = np.stack((state,state,state,state),axis=2).reshape(84,84,4)

        done = False
        totalreward = 0
        timesteps = 0

        # run episode until done (or until max time steps)
        while not done:
            env.render()

            action = dqn.selectAction(state)

            observation, reward, done, info = env.step(action)

            next_state = pre.above_bar_no_resize(observation)
            next_state = np.append(next_state,state,axis=2)[:,:,1:]

            dqn.storeExperience(state,action,reward,next_state,done)

            minibatch = dqn.sampleExperience()
            state_batch = [experience[0] for experience in minibatch]
            action_batch = [experience[1] for experience in minibatch]
            reward_batch = [experience[2] for experience in minibatch]
            nextstate_batch = [experience[3] for experience in minibatch]
            terminal_batch = [experience[4] for experience in minibatch]


            y_batch = []
            Q_batch = sess.run(dqn.targetQNet.q_value, feed_dict={dqn.targetQNet.stateInput: nextstate_batch})
            for i in range(len(minibatch)):
                terminal = terminal_batch[i]
                if terminal:#terminal states only use reward
                    y_batch.append(reward_batch[i])
                else:#else q-learning
                    y_batch.append(reward_batch[i]+gamma*np.max(Q_batch[i]))

            currentQ_batch = sess.run(dqn.currentQNet.q_value, feed_dict={dqn.currentQNet.stateInput: state_batch})

            sess.run(dqn.trainStep, feed_dict= {dqn.yInput: y_batch, dqn.actionInput: action_batch, dqn.currentQNet.stateInput: state_batch})

            state = next_state

            if timesteps % update_time_steps == 0:
                sess.run(dqn.copyCurrentToTargetNet())
            if timesteps % max_time_steps == 0:
                break

            state = pre.above_bar_no_resize(observation)

            totalreward += reward
            timesteps += 1

        totalrewards[episode] = totalreward
        if episode % 1 == 0:
            print("episode:", episode, "timesteps:", timesteps, "total reward:", totalreward, "eps:", dqn.epsilon,
                  "avg reward (last 100):", totalrewards[max(0, episode - 100):(episode + 1)].mean())
            # if episode % batch_size == 0:
            # save_path = saver.save(sess, model_path)
            # print("model saved in file: ", save_path,"\n")
            # can load later with saver.restore(sess, model_path)
        print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("simple qnet", totalrewards, int(num_episodes / 10))
    mp.show()

