import numpy as np

class Q:
    #discrete only
    def __init__(self, ob_space, act_space, lr=0.5, gamma=0.9):
        self.q = np.random.randn(ob_space, act_space) * 0.1
        self.lr = lr
        self.gamma = gamma
    def __call__(self, state):
        return self.q[state]
    def update(self, states, rewards, actions, dones):
        for t in reversed(range(len(rewards))):
            td = rewards[t] + self.gamma * self.q[states[t+1]].max() * (1-dones[t]) - self.q[states[t],actions[t]]
            self.q[states[t],actions[t]] += self.lr * td
    def param(self):
        return self.q
    def save(self, path=""):
        np.save(path, self.q)
    def load(self, path=""):
        self.q = np.load(path)