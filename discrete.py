from envs.LineEnv import LineEnv
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="store_true")
    #parser.add_argument("--linear", action="store_true")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-1)
    parser.add_argument("-e", "--max_epochs", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--line_length", type=int, default=10)
    args = parser.parse_args()
    if args.table:
        from func.table import Q as Q_f
    else:
        from func.linear import Q_discrete as Q_f
    T = args.T
    lr = args.lr
    max_epochs = args.max_epochs
    gamma = args.gamma
    epsilon = args.epsilon
    length = args.line_length
    env = LineEnv(length)
    ob_s = env.observation_space.n
    ac_s = env.action_space.n
    Q = Q_f(ob_s, ac_s, lr, gamma)
    print(0, Q.param())
    for i in range(max_epochs):
        states = np.zeros([T+1], dtype=np.int32)
        actions = np.zeros([T], dtype=np.int32)
        rewards = np.zeros([T], dtype=np.float32)
        dones = np.zeros([T], dtype=np.int32)
        states[0] = env.reset()
        last = T
        for t in range(T):
            if np.random.multinomial(1,[epsilon, 1-epsilon])[1]:
                actions[t] = Q(states[t]).argmax()
            else:
                actions[t] = np.random.randint(ac_s)
            states[t+1], rewards[t], dones[t], _ = env.step(actions[t])
            if dones[t]:
                last = t + 1
                break
        Q.update(states[:last+1], rewards[:last], actions[:last], dones[:last])
        print(i+1, last)
        print(Q.param())