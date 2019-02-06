import gym
import numpy as np
LEFT = 0
RIGHT = 1
class LineEnv(gym.Env):
    def __init__(self, length=10):
        self.goal = length
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array(0), high=np.array(length-1), dtype=np.int32)
        self.observation_space.n = length
    def reset(self):
        self.state = np.array(0)
        return self.state
    def step(self, action):
        if action == LEFT:
            if self.state != 0:
                self.state -= 1
        elif action == RIGHT:
            if self.state != self.goal - 1:
                self.state += 1
        if self.state == self.goal - 1:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0
        return self.state, reward, done, {}
    def render(self):
        for i in range(self.goal-1):
            if i == self.state:
                print("S", end="")
            else:
                print("_", end="")
        print("G")