import numpy as np


class TensorDP:

    def __init__(self,
                 gamma=1.0,
                 error_tol=1e-5):
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

        print("Tensor DP agent initialized")
        print("Environment spec:  Num. state = {} | Num. actions = {} ".format(env.nS, env.nA))

    def reset_policy(self):
        self.policy = np.ones([self.ns, self.na]) / self.na

    def set_policy(self, policy):
        assert self.policy.shape == policy.shape
        self.policy = policy

    def get_r_pi(self, policy):
        """
        Compute R_pi. The expected output shape is [num. states x 1]
        Refer to the lecture note <Part02 Chapter 01 L02 MDP> page 6
        """
        r_pi = "Fill this line!"
        return r_pi

    def get_p_pi(self, policy):
        """
        Compute P_pi. The expected output shape is [num. states x num. states]
        Refer to the lecture note <Part02 Chapter 01 L02 MDP> page 6
        """
        p_pi = "Fill this line!"
        return p_pi

    def policy_evaluation(self, policy=None, v_init=None):
        """
        :param policy: policy to evaluate (optional)
        :param v_init: initial value 'guesstimation' (optional)
        :param steps: steps of bellman expectation backup (optional)
        if none, repeat the backup until converge.

        :return: v_pi: value function of the input policy
        """
        if policy is None:
            policy = self.policy

        r_pi = self.get_r_pi(policy)  # [num. states x 1]
        p_pi = self.get_p_pi(policy)  # [num. states x num. states]

        if v_init is None:
            v_old = np.zeros(self.ns)
        else:
            v_old = v_init

        while True:
            # perform bellman expectation back
            # Refer to the lecture note <Part02 Chapter 02 L02 DP> page 8
            v_new = "Fill this line!"

            # check convergence
            bellman_error = np.linalg.norm(v_new - v_old)
            if bellman_error <= self.error_tol:
                break
            else:
                v_old = v_new

        return v_new

    def policy_improvement(self, policy=None, v_pi=None):
        if policy is None:
            policy = self.policy

        if v_pi is None:
            v_pi = self.policy_evaluation(policy)

        # Compute Q_pi(s,a) from V_pi(s)
        # Refer to the lecture note <Part02 Chapter 02 L02 DP> page 9
        q_pi = "Fill this line!"  # q_pi = [num.action x num states]

        # Greedy improvement
        # Refer to the lecture note <Part02 Chapter 02 L02 DP> page 9
        policy_improved = np.zeros_like(policy)
        """you need to greedily improve the given policy on here!"""
        return policy_improved

    def policy_iteration(self, policy=None):
        if policy is None:
            pi_old = self.policy
        else:
            pi_old = policy

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['converge'] = None

        steps = 0
        converged = False
        while True:
            v_old = self.policy_evaluation(pi_old)
            pi_improved = self.policy_improvement(pi_old, v_old)
            steps += 1

            info['v'].append(v_old)
            info['pi'].append(pi_old)

            # check convergence
            policy_gap = np.linalg.norm(pi_improved - pi_old)

            if policy_gap <= self.error_tol:
                if not converged:  # record the first moment of within error tolerance.
                    info['converge'] = steps
                break
            else:
                pi_old = pi_improved
        return info

    def value_iteration(self, v_init=None, compute_pi=False):
        """
        :param v_init: (np.array) initial value 'guesstimation' (optional)
        :param compute_pi: (bool) compute policy during VI
        :return: v_opt: the optimal value function
        """

        if v_init is not None:
            v_old = v_init
        else:
            v_old = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['converge'] = None

        steps = 0
        converged = False

        while True:
            # Bellman optimality backup
            # Refer to the lecture note <Part02 Chapter 02 L02 DP> page 9
            v_improved = "Fill this line!"
            info['v'].append(v_improved)

            if compute_pi:
                # compute policy from v
                # 1) Compute v -> q
                q_pi = (self.R.T + self.gamma * self.P.dot(v_improved))

                # 2) Construct greedy policy
                pi = np.zeros_like(self.policy)
                pi[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1
                info['pi'].append(pi)

            steps += 1

            # check convergence
            policy_gap = np.linalg.norm(v_improved - v_old)

            if policy_gap <= self.error_tol:
                if not converged:  # record the first moment of within error tolerance.
                    info['converge'] = steps
                break
            else:
                v_old = v_improved
        return info
