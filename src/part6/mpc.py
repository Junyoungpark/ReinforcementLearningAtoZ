from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class MPC:

    def __init__(self,
                 model: nn.Module,
                 state_dim: int,
                 action_dim: int,
                 H: int,
                 state_ref: Union[np.array, torch.tensor] = None,
                 action_min: Union[float, List[float]] = None,
                 action_max: Union[float, List[float]] = None,
                 Q: Union[np.array, torch.tensor] = None,
                 R: Union[np.array, torch.tensor] = None,
                 r: Union[np.array, torch.tensor] = None):
        """
        :param model: an instance of pytorch nn.module.
        the input of model expected to be [1 x state_dim] and [1 x action_dim]
        the output of model expected to be [1 x state_dim]
        :param state_dim: dimension of state
        :param action_dim: dimension of action
        :param H: receding horizon
        :param state_ref: trajectory of goal state, torch.tensor with dimension [H x state_dim]
        :param action_min: minimum value of action
        :param action_max: maximum value of action
        :param Q: weighting matrix for (state-x_ref)^2, torch.tensor with dimension [state_dim x state_dim]
        :param R: weighting matrix for (action)^2, torch.tensor with dimension [action_dim x action_dim]
        :param r: weighting matrix for (del_action)^2, torch.tensor with dimension [action_dim x action_dim]
        """
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.H = H

        if action_min is None or action_max is None:  # assuming the actions are not constrained
            self._constraint = False
        else:
            self._constraint = True

        # inferring the action constraints
        if isinstance(action_min, float):
            self.action_min = [action_min] * self.action_dim * self.H
        else:
            self.action_min = action_min

        if isinstance(action_max, float):
            self.action_max = [action_max] * self.action_dim * self.H
        else:
            self.action_max = action_max

        self.action_bnds = []
        for a_min, a_max in zip(self.action_min, self.action_max):
            assert a_min < a_max, "Action min is larger or equal to the action max"
            self.action_bnds.append((a_min, a_max))
        self.action_bnds = tuple(self.action_bnds)

        if state_ref is None:  # infer the ground state as reference
            state_ref = torch.zeros(H, state_dim)

        self.x0 = None
        self.x_ref = state_ref
        self.u_prev = None

        # state deviation penalty matrix
        if Q is None:
            Q = torch.eye(state_dim)
        if isinstance(Q, np.ndarray):
            Q = torch.tensor(Q).float()
        self.Q = Q

        # action exertion penalty matrix
        if R is None:
            R = torch.zeros(self.action_dim, self.action_dim)
        if isinstance(R, np.ndarray):
            R = torch.tensor(R).float()
        self.R = R

        # delta action penalty matrix
        if r is None:
            r = torch.zeros(self.action_dim, self.action_dim)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r).float()
        self.r = r

    def roll_out(self, x0, us):

        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [1 x state_dim]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [time stamps x  action_dim]
        :return: rolled out sequence of states
        """
        xs = []
        x = x0

        # us : [time stamps x action_dim]
        for u in us.split(1, dim=0):  # iterating over time stamps
            x = self.model(x, u)
            xs.append(x)
        return torch.cat(xs, dim=0)  # [time stamps x state_dim]

    @staticmethod
    def _compute_loss(deltas, weight_mat):
        """
        :param deltas:  # [num_steps x variable_dim]
        :param weight_mat: # [variable_dim x variable_dim]
        :return:
        """

        steps = deltas.shape[0]
        weight_mat = weight_mat.unsqueeze(dim=0)  # [1 x variable_dim x variable_dim]
        weight_mat = weight_mat.repeat_interleave(steps, dim=0)  # [num_steps x variable_dim x variable_dim]
        deltas_transposed = deltas.unsqueeze(dim=1)  # [num_steps x 1 x variable_dim]
        deltas = deltas.unsqueeze(dim=-1)  # [num_steps x variable_dim x 1]
        loss = deltas_transposed.bmm(weight_mat).bmm(deltas)  # [num_steps x 1 x 1]
        loss = loss.mean() #sum()
        return loss

    def compute_objective(self, x0, us, x_ref=None, u_prev=None):
        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [1 x state_dim]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :param x_ref: state targets
        """
        assert self.H == us.shape[0], \
            "The length of given action sequences doesn't match with receeding horizon length H."

        # Compute state deviation loss
        x_preds = self.roll_out(x0, us)  # [time stamps x state_dim]
        if x_ref is None:
            x_ref = self.x_ref  # [time stamps x state_dim]
        x_deltas = x_preds - x_ref  # [time stamps x state_dim]
        state_loss = self._compute_loss(x_deltas, self.Q)

        # Compute action exertion loss
        action_loss = self._compute_loss(us, self.R)

        # Compute delta action loss
        if u_prev is None:
            u_prev = torch.zeros(1, self.action_dim)
        us = torch.cat([u_prev, us], dim=0)
        delta_actions = us[:1, :] - us[:-1, :]
        delta_action_loss = self._compute_loss(delta_actions, self.r)

        # construct MPC loss
        loss = state_loss + action_loss + delta_action_loss
        return loss

    def set_mpc_params(self, x0, x_ref=None, u_prev=None):
        self.x0 = x0

        if x_ref is not None:
            self.x_ref = x_ref
        self.u_prev = u_prev

    def _obj(self, us: np.array):
        us = torch.from_numpy(us).float()
        us = us.view(self.H, self.action_dim)
        with torch.no_grad():
            obj = self.compute_objective(x0=self.x0, us=us, x_ref=self.x_ref).numpy()
        return obj

    def _obj_jac(self, us: np.array):
        us = torch.from_numpy(us).float()
        us = us.view(self.H, self.action_dim)

        def _jac(us): return self.compute_objective(x0=self.x0, us=us, x_ref=self.x_ref)

        jac = torch.autograd.functional.jacobian(_jac, us)
        return jac

    def solve(self, u0=None):
        """
        :param u0:  initial action sequences, only 1D.
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :return:
        """
        if u0 is None:
            u0 = np.stack([self.action_max, self.action_min]).mean(axis=0)

        opt_result = minimize(self._obj, u0, method='SLSQP', bounds=self.action_bnds, jac=self._obj_jac)

        opt_action = torch.tensor(opt_result.x).view(self.H,
                                                     self.action_dim).float().detach()  # optimized action sequences
        pred_states = self.roll_out(self.x0, opt_action)
        return opt_action, pred_states, opt_result