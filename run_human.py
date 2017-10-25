import cv2
import numpy as np
from time import sleep
from pyglet.window import key
from matplotlib import pyplot as plt

from old_models.preprocessing import myprep, car_field
from human_env import CarRacing

ACTION = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  ACTION[0] = -1.0
    if k == key.RIGHT: ACTION[0] = +1.0
    if k == key.UP:    ACTION[1] = +1.0
    if k == key.DOWN:  ACTION[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and ACTION[0] == -1.0: ACTION[0] = 0
    if k == key.RIGHT and ACTION[0] == +1.0: ACTION[0] = 0
    if k == key.UP:    ACTION[1] = 0
    if k == key.DOWN:  ACTION[2] = 0


def main():
    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    matrices = []

    for i in range(10):
        play_one(env, matrices, record_video)

    f = open('roads.npy', 'wb')
    np.save(f, np.vstack(matrices))
    f.close()

    env.close()


def play_one(env, matrices, record_video):
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False

    while True:
        # if steps == 1:
        #     sleep(5)
        #     break

        s, r, done, info = env.step(ACTION)

        road_matrix = myprep(s)
        car_matrix = car_field(s)

        # if steps==50:
        #     plt.imshow(road_matrix)
        #     plt.show()

        cv2.imshow('road', road_matrix)
        cv2.imshow('car', car_matrix)
        cv2.waitKey(1)

        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in ACTION]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.savefig("test.jpeg")
        steps += 1
        if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
            env.render()
        if done or restart: break


if __name__=="__main__":
    main()