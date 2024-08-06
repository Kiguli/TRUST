import sympy as sp
import numpy as np


class Stability:
    def __init__(self, data: dict):
        if data['mode'] != 'Stability':
            raise ValueError(f"Invalid mode '{data["mode"]}' for Stability calculations.")

        self.model = data['model']
        self.timing = data['timing']
        self.dataset = data['dataset']
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']

    def calculate(self):
        if self.model == 'Linear' and self.timing == 'Discrete-Time':
            lyapunov, controller = self._solve_discrete_time_linear_system()
        else:
            lyapunov, controller = None, None

        return {
            'lyapunov': lyapunov,
            'controller': controller,
        }

    def _solve_discrete_time_linear_system(self) -> tuple:
        """
        We wish to find the matrix :math:`H \\in \\mathbb{R}^{T\\times n}` and symmetric positive definite matrix
        :math:`P \\in \\mathbb{R}^{n\\times n}`.

        The Lyapunov function to return is then :math:`V(x) = x^\\top P x` and the controller to return is
        :math:`u=\\mathcal{U}_{0,T}HP^{-1}x`.

        :return: (Lyapunov_function, controller)
        """

        lyapunov = {'expression': 'x^T @ P @ x', 'values': {'P': 1}}
        controller = {'expression': 'U_{0,T} @ H @ P^{-1} @ x', 'values': {'H': 2, 'P': 3}}

        return lyapunov, controller
