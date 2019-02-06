import numpy as np

class Q_discrete:
    def __init__(self, ob_space, act_space, lr=0.5, gamma=0.9):
        self.W = np.random.randn(ob_space, act_space) * 0.1
        self.lr = lr
        self.gamma = gamma
    def __call__(self, state):
        st = np.eye(self.W.shape[0])[state].T
        return st @ self.W
    def update(self, states, rewards, actions, dones):
        states = np.eye(self.W.shape[0])[states]
        for t in reversed(range(len(rewards))):
            td = rewards[t] + self.gamma * (states[t+1] @ self.W).max() * (1-dones[t]) - (states[t].T @ self.W)[actions[t]]
            self.W[:,actions[t]] += self.lr * td * states[t].reshape(-1)
    def param(self):
        return self.W
    def save(self, path=""):
        np.save(path, self.W)
    def load(self, path=""):
        self.W = np.load(path)