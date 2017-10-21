import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
from simple_qnet import *
from gym import wrappers

env = gym.make('CarRacing-v0')
env = wrappers.Monitor(env, 'monitor-folder', force=True)
#repeats an action for 4 frames with env.step(action)
every_four_frames = wrappers.SkipWrapper(4)
env = every_four_frames(env)


num_episodes = 1000
max_time_steps = 1500
default_time_steps = 10
# save_episodes = 25

#learning parameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1
min_epsilon = 0.1
update_steps = 100

init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# model_path = "./model/car.ckpt"

with tf.Session() as sess:
    sqn = SQN()
    sess.run(init)


    totalrewards = np.empty(num_episodes)
    # totallosses = np.empty(num_episodes)

    for episode in range(num_episodes):
        #decrease epsilon every 10 episodes
        if (episode % 10 == 0 and epsilon >= min_epsilon):
            epsilon -= 0.01

        #reset for each episode
        observation = env.reset()
        done = False
        totalreward = 0
        timesteps = 0
        # totalloss = 0

        #run episode until done (or until max time steps)
        while not done:
            env.render()

            state = pre.compute_state(observation)

            action = sqn.selectAction(state,epsilon)

            observation, reward, done, info = env.step(action)
            prev_state = state
            state = pre.compute_state(observation)

            #store experience in replay memory
            sqn.storeExperience(prev_state,action,reward,state,done)

            #sample experiences from replay memory
            minibatch = sqn.sampleExperiences()
            state_batch = [experience[0] for experience in minibatch]
            action_batch = [experience[1] for experience in minibatch]
            reward_batch = [experience[2] for experience in minibatch]
            nextstate_batch = [experience[3] for experience in minibatch]
            terminal_batch = [experience[4] for experience in minibatch]

            y_batch = []
            q_batch = sess.run(sqn.targetQNet.qvals, feed_dict={sqn.targetQNet.stateInput: nextstate_batch})
            for i in range(len(minibatch)):
                terminal = terminal_batch[i]
                if terminal:
                    y_batch.append(reward_batch(i))
                else:
                    y_batch.append(reward_batch[i] + gamma * np.max(q_batch[i]))


            currentQ_batch = sess.run(sqn.currentQNet.qvals,feed_dict={sqn.currentQNet.stateInput: state_batch})

            sess.run(sqn.trainStep,feed_dict={sqn.yInput: y_batch, sqn.actionInput: action_batch, sqn.currentQNet.stateInput: state_batch})

            #copy current qnet to target net
            if timesteps % update_steps == 0:
                sess.run(sqn.copyCurrentToTargetNet())



            # totalloss += cost
            totalreward += reward
            timesteps += 1

            if timesteps > max_time_steps:
                print("too many time steps, breaking")
                break

        # totallosses[episode] = totalloss
        totalrewards[episode] = totalreward
        if episode % 1 == 0:
            print("episode:", episode, "timesteps:", timesteps, "total reward:", totalreward, "eps:", epsilon, "avg reward (last 100):",totalrewards[max(0,episode-100):(episode+1)].mean())
        # if episode % save_episodes == 0:
            # save_path = saver.save(sess, model_path)
            # print("model saved in file: ", save_path,"\n")
            #can load later with saver.restore(sess, model_path)
        print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("simple qnet w/ experience replay", totalrewards, int(num_episodes/10))
    mp.show()
