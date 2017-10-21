import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
import tensorflow as tf
from gym import wrappers


''' 
third experiment:
simple q-learning agent

same as second experiment but repeats actions for 3 frames
'''

env = gym.make('CarRacing-v0')
env = wrappers.Monitor(env, 'monitor-folder', force=True)
#repeats an action for 3 frames with env.step(action)
every_three_frames = wrappers.SkipWrapper(3)
env = every_three_frames(env)

num_episodes = 1000
max_time_steps = 1500
batch_size = 10
action_time_steps = 1
default_time_steps = 10

#learning parameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1 #starting at 1 so random all of the time (lowering as episodes increase)
# epsilon = 0.1 #after 900 episodes
min_epsilon = 0.1

action_set = np.array([
#steering (left,right)
[-1.0,0,0],
[-0.75,0,0],
[-0.5,0,0],
[-0.25,0,0],
[0,0,0],
[0.25,0,0],
[0.5,0,0],
[0.75,0,0],
[1.0,0,0],
#gas
[0,0.3,0],
[0,0.5,0],
#brake
[0,0,0.8]
])

#network parameters
num_input = 10*10+7+4 # size of list returned from preprocessing
num_hidden = 512
num_output = len(action_set)

#tf graph input
x = tf.placeholder("float",[None,num_input])

#layers and weight bias
weights = {
    'hidden':tf.Variable(tf.random_normal([num_input,num_hidden])),
    'output': tf.Variable(tf.random_normal([num_hidden, num_output]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden])),
    'output': tf.Variable(tf.random_normal([num_output]))
}

#establish forward feed part of network to choose actions
hidden = tf.nn.relu(tf.add(tf.matmul(x,weights['hidden']),biases['hidden']))
qvals = tf.add(tf.matmul(hidden,weights['output']),biases['output'])

#update model based on loss
next_qvals = tf.placeholder("float",shape=[1,num_output])
loss = tf.reduce_mean(tf.squared_difference(next_qvals,qvals))#mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_model = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
model_path = "./model/car.ckpt"

def get_env_action(nn_output, eps):
    if np.random.random() < eps:
        action_index = random.randint(0, len(action_set)-1)
    else:
        action_index = np.argmax(nn_output)
    return action_set[action_index]

with tf.Session() as sess:
    sess.run(init)

    totalrewards = np.empty(num_episodes)
    totallosses = np.empty(num_episodes)
    for episode in range(num_episodes):
        #decrease epsilon every 10 episodes
        if (episode % 10 == 0 and epsilon >= min_epsilon):
            epsilon -= 0.01

        #reset for each episode
        observation = env.reset()
        # prev_state = None
        done = False
        totalreward = 0
        timesteps = 0
        totalloss = 0

        #run episode until done (or until max time steps)
        while not done:
            env.render()

            # cur_state = pre.compute_state(observation)
            # state = cur_state - prev_state if prev_state is not None else np.zeros(num_input)
            state = pre.compute_state(observation)

            q_values = sess.run(qvals, feed_dict={x: state.reshape(-1,num_input)})[0]

            action = get_env_action(q_values,epsilon)

            observation, reward, done, info = env.step(action)
            prev_state = state
            state = pre.compute_state(observation)

            q_prime = sess.run(qvals,feed_dict={x:state.reshape(-1,num_input)})[0]
            if not done:
                q_target = reward + gamma * np.max(q_prime)
            else:#set terminal states to just the reward
                q_target = reward
            target = q_values[:]
            target[np.argmax(q_values)] = q_target

            #using cost bc didn't want to overwrite loss function
            _,cost = sess.run([update_model,loss],feed_dict={x:prev_state.reshape(-1,num_input),next_qvals:target.reshape(-1,num_output)})


            totalloss += cost
            totalreward += reward
            timesteps += 1

            if timesteps > max_time_steps:
                print("too many time steps, breaking")
                break

        totallosses[episode] = totalloss
        totalrewards[episode] = totalreward
        if episode % 1 == 0:
            print("episode:", episode, "timesteps:", timesteps, "total reward:", totalreward, "eps:", epsilon, "avg reward (last 100):",totalrewards[max(0,episode-100):(episode+1)].mean())
        if episode % batch_size == 0:
            save_path = saver.save(sess, model_path)
            print("model saved in file: ", save_path,"\n")
            #can load later with saver.restore(sess, model_path)
        print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("simple qnet", totalrewards, int(num_episodes/10))
    mp.show()
