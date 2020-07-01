import numpy as np


class TDAgent:

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float,
                 n_step: int):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.epsilon = epsilon
        self.n_step = n_step

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self._policy_q = np.zeros(shape=(self.num_states, self.num_actions))

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self._policy_q[state, :].argmax()
        return action

    def update(self, episode):
        states, actions, rewards = episode
        ep_len = len(states)

        states += [0] * (self.n_step + 1)  # append dummy states
        rewards += [0] * (self.n_step + 1)  # append dummy rewards
        dones = [0] * ep_len + [1] * (self.n_step + 1)

        kernel = np.array([self.gamma ** i for i in range(self.n_step)])
        for i in range(ep_len):
            s = states[i]
            ns = states[i + self.n_step]
            done = dones[i]

            # compute n-step TD target
            # Refer to the lecture note <Part02 Chapter03 L03 TD evaluation> page 11
            g = "Fill this line"
            self.v[s] += self.lr * (g - self.v[s])

    def sample_update(self, state, action, reward, next_state, done):
        # Implement sample update version of 1-step TD target
        # Refer to the lecture note <Part02 Chapter03 L03 TD evaluation> page 5
        # Don't forget to utilize 'done' to ignore the last state values.
        td_target = "Fill this line"
        self.v[state] += self.lr * (td_target - self.v[state])

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class SARSA(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(SARSA, self).__init__(gamma=gamma,
                                    num_states=num_states,
                                    num_actions=num_actions,
                                    epsilon=epsilon,
                                    lr=lr,
                                    n_step=1)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self.q[state, :].argmax()
        return action

    def update_sample(self, state, action, reward, next_state, next_action, done):
        s, a, r, ns, na = state, action, reward, next_state, next_action

        # SARSA target
        # Refer to the lecture note <Part02 Chapter04 L04 TD control:SARSA> page 4
        td_target = "Fill this line"
        self.q[s, a] += self.lr * (td_target - self.q[s, a])


class QLearner(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(QLearner, self).__init__(gamma=gamma,
                                       num_states=num_states,
                                       num_actions=num_actions,
                                       epsilon=epsilon,
                                       lr=lr,
                                       n_step=1)

    def get_action(self, state, mode='train'):
        if mode == 'train':
            prob = np.random.uniform(0.0, 1.0, 1)
            # e-greedy policy over Q
            if prob <= self.epsilon:  # random
                action = np.random.choice(range(self.num_actions))
            else:  # greedy
                action = self.q[state, :].argmax()
        else:
            action = self.q[state, :].argmax()
        return action

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        # Q-Learning target
        # Refer to the lecture note <Part02 Chapter05 L02 Off-policy TD control and Q-Learning> page 7
        td_target = "Fill this line"
        self.q[s, a] += self.lr * (td_target - self.q[s, a])


def run_episode(env, agent):
    env.reset()
    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update_sample(state, action, reward, next_state, done)

        if done:
            break
