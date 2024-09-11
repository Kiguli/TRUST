import array
from typing import Self

import cvxpy as cp
import numpy as np


class Stability:
    def __init__(self):
        self.model = None
        self.timing = None
        self.X0 = None
        self.X1 = None
        self.U0 = None
        self.state_space = None
        self.initial_state = None
        self.unsafe_states = None

    def create(self, data: dict) -> Self:
        """
        Create a new instance of the Stability class with the given data.
        """
        if data['mode'] != 'Stability':
            raise ValueError(f"Invalid mode '{data["mode"]}' for Stability calculations.")

        self.model = data['model']
        self.timing = data['timing']
        self.X0 = data['X0']
        self.X1 = data['X1']
        self.U0 = data['U0']
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']

        return self

    def calculate(self):
        if self.model == 'Linear':
            lyapunov, controller = self._solve_linear_system()
        else:
            lyapunov, controller = None, None

        return {
            'lyapunov': lyapunov,
            'controller': controller,
        }

    # --- Builder Pattern ---

    def model(self, model: str) -> Self:
        self.model = model
        return self

    def timing(self, timing: str) -> Self:
        self.timing = timing
        return self

    def X0(self, X0: array) -> Self:
        self.X0 = X0
        return self

    def X1(self, X1: array) -> Self:
        self.X1 = X1
        return self

    def U0(self, U0: array) -> Self:
        self.U0 = U0
        return self

    def state_space(self, state_space: array) -> Self:
        self.state_space = state_space
        return self

    def initial_state(self, initial_state: array) -> Self:
        self.initial_state = initial_state
        return self

    def unsafe_states(self, unsafe_states: array) -> Self:
        self.unsafe_states = unsafe_states
        return self

    def _solve_linear_system(self) -> tuple:
        X0 = np.array([self.X0])
        X1 = np.array([self.X1])

        n = X0.shape[0]
        T = X0.shape[1]

        P = cp.Variable((n, n), symmetric=True)
        H = cp.Variable((T, n))

        if self.timing == 'Discrete-Time':
            constraints = self._discrete_constraints(X0, X1, P, H)
        elif self.timing == 'Continuous-Time':
            constraints = self._continuous_constraints(X0, X1, P, H)
        else:
            constraints = []

        objective = cp.Minimize(cp.trace(P))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("The problem is infeasible or unbounded.")

        # TODO: Z = P.inverse()
        P = Z

        H = np.array2string(np.array(H.value))
        P = np.array2string(np.array(P.value))

        lyapunov = {
            'expression': 'x^T @ P @ x',
            'values': {'P': P}
        }
        controller = {
            'expression': 'U_{0,T} @ H @ P^{-1} @ x',
            'values': {'H': H}
        }

        return lyapunov, controller

    @staticmethod
    def _discrete_constraints(X0, X1, Z, H) -> array:
        block_matrix = cp.bmat([
            [Z, X1 @ H],
            [H.T @ X1.T, Z]
        ])

        return [Z >> 0, Z == X0 @ H, block_matrix >> 0]

    @staticmethod
    def _continuous_constraints(X0, X1, Z, H) -> array:
        eqn = X1 @ H + H.T @ X1.T

        return [Z >> 0, Z == X0 @ H, eqn << 0]
