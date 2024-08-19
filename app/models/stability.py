import sympy as sp
import numpy as np
import cvxpy as cp
import array


class Stability:
    def __init__(self, data: dict):
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

    def calculate(self):
        if self.model == 'Linear':
            lyapunov, controller = self._solve_linear_system()
        else:
            lyapunov, controller = None, None

        return {
            'lyapunov': lyapunov,
            'controller': controller,
        }

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

        lyapunov = {'expression': 'x^T @ P @ x', 'values': {'P': P.value.tolist()}}
        controller = {'expression': 'U_{0,T} @ H @ P^{-1} @ x',
                      'values': {'H': H.value.tolist(), 'P': P.value.tolist()}}

        return lyapunov, controller

    @staticmethod
    def _discrete_constraints(X0, X1, P, H) -> array:
        block_matrix = cp.bmat([
            [P, X1 @ H],
            [H.T @ X1.T, P]
        ])

        return [P >> 0, P == X0 @ H, block_matrix >> 0]

    @staticmethod
    def _continuous_constraints(X0, X1, P, H) -> array:
        eqn = X1 @ H + H.T @ X1.T

        return [P >> 0, P == X0 @ H, eqn << 0]
