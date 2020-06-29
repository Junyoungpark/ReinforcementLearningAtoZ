class Trajectory:

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.next_states = list()
        self.dones = list()

        self.length = 0
        self.returns = None
        self._discounted = False

    def push(self, state, action, reward, next_state, done):
        if done and self._discounted:
            raise RuntimeError("done is given at least two times!")

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.length += 1

        if done and not self._discounted:
            # compute returns
            self.compute_return()

    def compute_return(self):
        rewards = self.rewards
        returns = list()

        g = 0
        # iterating returns in inverse order
        for r in rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)
        self.returns = returns
        self._discounted = True

    def get_samples(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones, self.returns
