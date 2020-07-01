import numpy as np
from queue import PriorityQueue


class AsyncDP:

    def __init__(self,
                 gamma=1.0,
                 error_tol=1e-8):
        self.gamma = gamma
        self.error_tol = error_tol

        # Following attributes will be set after call "set_env()"

        self.env = None  # environment
        self.policy = None  # policy
        self.ns = None  # Num. states
        self.na = None  # Num. actions
        self.P = None  # Transition tensor
        self.R = None  # Reward tensor

    def set_env(self, env, policy=None):
        self.env = env
        if policy is None:
            self.policy = np.ones([env.nS, env.nA]) / env.nA

        self.ns = env.nS
        self.na = env.nA
        self.P = env.P_tensor  # Rank 3 tensor [num. actions x num. states x num. states]
        self.R = env.R_tensor  # Rank 2 tensor [num. actions x num. states]

        print("Asynchronous DP agent initialized")
        print("Environment spec:  Num. state = {} | Num. actions = {} ".format(env.nS, env.nA))

    def compute_q_from_v(self, value):
        return self.R.T + self.gamma * self.P.dot(value)  # [num. actions x num. states]

    def construct_policy_from_v(self, value):
        qs = self.compute_q_from_v(value)  # [num. actions x num. states]

        # construct greedy policy from Qs.
        pi = np.zeros_like(self.policy)
        pi[np.arange(qs.shape[1]), qs.argmax(axis=0)] = 1
        return pi

    def in_place_vi(self, v_init=None):
        """
        :param v_init: (np.array) initial value 'guesstimation' (optional)
        :return:
        """

        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()
        info['converge'] = False
        info['step'] = None

        steps = 0
        # loop over value iteration:
        # perform the loop until the value estimates are converge
        while True:
            # loop over states:
            # we perform 'sweeping' over states in "any arbitrary" order.
            # without-loss-of-generality, we can perform as the order of states.
            # in this example, we consider the "full sweeping" of values.

            # perform in-place VI
            delta_v = 0
            for s in range(self.ns):
                # Bellman expectation backup of current state in in-place fashion
                # get Q values of given state 's'

                # output of 'self.compute_q_from_v(value)' is a tensor
                # with the shape of [num. actions X num. states].

                # Implement in-place iteration !!!
                # 1. compute q values of given state
                # 2. find the maximum of q values and set as value of the given state
                # hint: Use "compute_q_from_v()"
                # Refer to the lecture note <Part02 Chapter 02 L03 Efficient DP> page 8

                qs = "Fill this line"
                v = "Fill this line"  # get max value along the actions

                # accumulate the deviation from the current state s
                # the deviation = |v_new - v|
                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v

            info['v'].append(value.copy())
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi)
            info['gap'].append(delta_v)

            if delta_v < self.error_tol:
                if info['converge']:
                    info['step'] = steps
                    break
                else:
                    info['converge'] = True
            else:
                steps += 1

        return info

    def prioritized_sweeping_vi(self, v_init=None):
        """
        :param v_init: (np.array) initial value 'guesstimation' (optional)
        :return:
        """

        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()
        info['converge'] = False
        info['step'] = None

        steps = 0
        while True:
            # compute the Bellman errors
            # bellman_errors shape : [num.states]

            # compute the bellman errors
            # Refer to the lecture note <Part02 Chapter 02 L03 Efficient DP> page 8
            bellman_errors = "Fill this line"
            bellman_errors = np.abs(bellman_errors)
            state_indices = range(self.ns)

            # put the (bellman error, state index) into the priority queue
            priority_queue = PriorityQueue()
            for bellman_error, s_idx in zip(bellman_errors, state_indices):
                priority_queue.put((-bellman_error, s_idx))

            delta_v = 0

            while not priority_queue.empty():
                be, s = priority_queue.get()
                qs = self.compute_q_from_v(value)[:, s]
                v = qs.max(axis=0)  # get max value along the actions

                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v

            info['gap'].append(delta_v)
            info['v'].append(value.copy())
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi.copy())

            if delta_v < self.error_tol:
                if info['converge']:
                    info['step'] = steps
                    break
                else:
                    info['converge'] = True
            else:
                steps += 1
        return info

    def in_place_vi_partial_update(self,
                                   v_init=None,
                                   update_prob=0.5,
                                   vi_iters: int = 100):
        """
        :param v_init: (np.array) initial value 'guesstimation' (optional)
        :return:
        """

        if v_init is not None:
            value = v_init
        else:
            value = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['gap'] = list()

        # loop over value iteration:
        # perform the loop until the value estimates are converge
        for steps in range(vi_iters):
            # loop over states:
            # we perform 'sweeping' over states in "any arbitrary" order.
            # without-loss-of-generality, we can perform as the order of states.
            # in this example, we consider the "full sweeping" of values.

            # perform in-place VI
            delta_v = 0

            for s in range(self.ns):
                perform_update = np.random.binomial(size=1, n=1, p=update_prob)
                if not perform_update:
                    continue

                # Bellman expectation backup of current state in in-place fashion
                # get Q values of given state 's'
                qs = self.compute_q_from_v(value)[:, s]
                v = qs.max(axis=0)  # get max value along the actions

                # accumulate the deviation from the current state s
                # the deviation = |v_new - v|
                delta_v += np.linalg.norm(value[s] - v)
                value[s] = v
            info['gap'].append(delta_v)
            info['v'].append(value.copy())
            pi = self.construct_policy_from_v(value)
            info['pi'].append(pi)

        return info
