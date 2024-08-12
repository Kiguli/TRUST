import sympy as sp
import numpy as np
import cvxpy as cp


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
        if self.model == 'Linear' and self.timing == 'Discrete-Time':
            lyapunov, controller = self._solve_discrete_time_linear_system()
        elif self.model == 'Linear' and self.timing == 'Continuous-Time':
            lyapunov, controller = self._solve_continuous_time_linear_system()
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

        X0 = np.array([self.X0])
        X1 = np.array([self.X1])

        n = X0.shape[0]
        T = X0.shape[1]

        # Unknown terms
        P = cp.Variable((n, n), symmetric=True)
        H = cp.Variable((T, n))

        block_matrix = cp.bmat([
            [P, X1 @ H],
            [H.T @ X1.T, P]
        ])

        constraints = [P >> 0, P == X0 @ H, block_matrix >> 0]

        # Set up a simple objective just to calculate P and H
        objective = cp.Minimize(cp.trace(P))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("The problem is infeasible or unbounded.")

        lyapunov = {'expression': 'x^T @ P @ x', 'values': {'P': P.value.tolist()}}
        controller = {'expression': 'U_{0,T} @ H @ P^{-1} @ x',
                      'values': {'H': H.value.tolist(), 'P': P.value.tolist()}}

        return lyapunov, controller

    def _solve_continuous_time_linear_system(self) -> tuple:
        X0 = np.array([self.X0])
        X1 = np.array([self.X1])

        n = X0.shape[0]
        T = X0.shape[1]

        P = cp.Variable((n, n), symmetric=True)
        H = cp.Variable((T, n))

        eqn = X1 @ H + H.T @ X1.T

        constraints = [P >> 0, eqn << 0]

        objective = cp.Minimize(cp.trace(P))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError("The problem is infeasible or unbounded.")

        lyapunov = {'expression': 'x^T @ P @ x', 'values': {'P': P.value.tolist()}}
        controller = {'expression': 'U_{0,T} @ H @ P^{-1} @ x',
                      'values': {'H': H.value.tolist(), 'P': P.value.tolist()}}

        return lyapunov, controller
