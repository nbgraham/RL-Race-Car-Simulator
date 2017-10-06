import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
import tensorflow as tf

env = gym.make('CarRacing-v0')

num_episodes = 500
max_time_steps = 1500
batch_size = 10
action_time_steps = 1

#learning parameters
learning_rate = 0.01
gamma = 0.99
#epsilon = 1 #starting at 1 so random all of the time (lowering as episodes increase)
                #i.e. after 100 episodes 0.9, 200 0.8, etc...
epsilon = 0.3 

action_set = np.array([
#nothing
#[0,0,0],
#steering (left,right)
[-1.0,0,0],
[1.0,0,0],
#gas
[0,0.3,0],
#brake
[0,0,0.5]
])

#network parameters
num_input = 10*10+7+4 # size of list returned from preprocessing
num_hidden = 256 #arbitrarily chosen
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
hidden = tf.nn.tanh(tf.add(tf.matmul(x,weights['hidden']),biases['hidden']))
qvals = tf.add(tf.matmul(hidden,weights['output']),biases['output'])

#update model based on loss
next_qvals = tf.placeholder("float",shape=[num_input,num_output])
loss = tf.reduce_sum(tf.square(next_qvals-qvals))#mean squared error/sum of squares? I think
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_model = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
model_path = "./model/"

def get_env_action(nn_output, eps):
    if np.random.random() < eps:
        # print("using eps")
        action_index = random.randint(0, len(action_set)-1)
    else:
        # print("using selected")
        action_index = np.argmax(nn_output[0])
        # action_index = np.argmax(nn_output)
    return action_set[action_index]

with tf.Session() as sess:
    sess.run(init)

    totalrewards = np.empty(num_episodes)
    for episode in range(num_episodes):
        #decrease epsilon every 100 episodes
        if (episode % 100 == 0 and epsilon >= 0.1):
            eps = epsilon - 0.1

        #reset for each episode
        observation = env.reset()
        done = False
        totalreward = 0
        timesteps = 0

        #run episode until done (or until max time steps)
        while not done:
            env.render()
            state = pre.compute_state(observation)

            q_values = sess.run(qvals, feed_dict={x: np.identity(num_input)*state})
            # print("q_values\n",q_values)
            action = get_env_action(q_values,eps)
            # print("action\n",action)

            observation, reward, done, info = env.step(action)
            prev_state = state
            state = pre.compute_state(observation)

            q_prime = sess.run(qvals,feed_dict={x:np.identity(num_input)*state})
            # print("q_prime\n",q_prime)
            # print("npmax q_prime\n",np.max(q_prime))
            # print("np argmax q_prime\n",np.argmax(q_prime))
            # print("reward\n",reward)
            q_target = reward + gamma * np.max(q_prime)
            # print("q_target\n",q_target)
            target = q_values[:]
            # print("target (should be q_values)\n",target)
            # target[0][action_index] = q_target
            target[np.argmax(q_values[0])] = q_target
            # print("target updated action index\n",target)
            # print("len target",len(target))

            #unsure
            _,lol = sess.run([update_model,weights['output']],feed_dict={x:np.identity(num_input)*state,next_qvals:target})

            totalreward += reward
            timesteps += 1

            if timesteps > max_time_steps:
                print("too many time steps, breaking")
                break

        totalrewards[episode] = totalreward
        if episode % 1 == 0:
            print("episode:", episode, "timesteps:", timesteps, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):",totalrewards[max(0,episode-100):(episode+1)].mean())
        if episode % batch_size == 0:
            save_path = saver.save(sess, model_path)
            print("model saved in file: ", save_path,"\n")
            #can load later with saver.restore(sess, model_path)
        print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("simple tf", totalrewards, int(num_episodes / 10))
    mp.show()