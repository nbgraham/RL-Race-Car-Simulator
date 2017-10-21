from pyglet.window import key
import gym
import numpy as np
import preprocessing as pre


record_video = False
plays = 10



a = np.array( [0.0, 0.0, 0.0] )

def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0

env = gym.make('CarRacing-v0')
env.render()
if record_video:
    env.monitor.start('/tmp/video-test', force=True)
env.env.viewer.window.on_key_press = key_press
env.env.viewer.window.on_key_release = key_release

for i in range(plays):
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    s = None
    while True:
        if steps == 0:
            state = pre.compute_state(observation)
        print("state\n",state)
        print("action\n", a)
        s, r, done, info = env.step(a)
        state = pre.compute_state(s)
        print("reward\n",r)
        print("next state\n", state)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.savefig("test.jpeg")
        steps += 1
        if not record_video: # Faster, but you can as well call env.render() every time to play full window.
            env.render()
        if done or restart: break
env.close()
