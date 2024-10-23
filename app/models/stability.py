import array
from typing import Self

import cvxpy as cp
import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem
from picos import Constant, RealVariable, SymmetricVariable, I
from sympy import Matrix


class Stability:
    def __init__(self):
        self.model = None
        self.timing = None
        self.monomials = None
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
        if data["mode"] != "Stability":
            raise ValueError(
                f"Invalid mode '{data["mode"]}' for Stability calculations."
            )

        self.model = data["model"]
        self.timing = data["timing"]
        self.monomials = data.get("monomials", [])
        self.X0 = self.parse_dataset(data["X0"])
        self.X1 = self.parse_dataset(data["X1"])
        self.U0 = self.parse_dataset(data["U0"])
        self.state_space = data["stateSpace"]
        self.initial_state = data["initialState"]
        self.unsafe_states = data["unsafeStates"]

        return self

    def calculate(self) -> dict:
        results = {}

        if self.model == "Linear":
            results = self._solve_linear()
        elif self.model == "Non-Linear Polynomial":
            results = self._solve_polynomial()

        return results

    def _solve_linear(self) -> dict:
        # X0 = np.array([self.X0])
        # X1 = np.array([self.X1])

        n = self.X0.shape[0]
        T = self.X0.shape[1]

        # P = cp.Variable((n, n), symmetric=True)
        # H = cp.Variable((T, n))

        H, Z = None, None
        if self.timing == "Discrete-Time":
            H, Z = self._discrete_constraints()
        elif self.timing == "Continuous-Time":
            H, Z = self._continuous_constraints()
        # else:
        #     constraints = []

        # objective = cp.Minimize(cp.trace(P))

        # prob = cp.Problem(objective, constraints)
        # prob.solve()

        # if prob.status in ["infeasible", "unbounded"]:
        #     raise ValueError("The problem is infeasible or unbounded.")

        H = np.array2string(np.array(sp.Matrix(H)))
        P = np.array2string(np.array(sp.Matrix(Z).inv())) if n > 1 else str(1 / Z.value)

        return {
            "lyapunov": {"expression": "x^T @ P @ x", "values": {"P": P}},
            "controller": {
                "expression": "U_{0,T} @ H @ P^{-1} @ x",
                "values": {"H": H},
            },
        }

    def _solve_polynomial(self) -> dict:
        if self.timing == "Discrete-Time":
            return self.__discrete_polynomial()
        elif self.timing == "Continuous-Time":
            return self.__continuous_polynomial()

    def __continuous_polynomial(self) -> dict:
        """
        Calculate the Lyapunov function and controller for a continuous-time non-linear polynomial system.

        Solve for P and H_x
        (1) N0 @ H_x = P_inv
        (2) dMdx @ X1 @ H_x + H_x.T @ X1.T @ dMdX.T << 0
        """
        T = self.num_samples
        N = self.N

        # P is a symmetric positive definite (N x N) matrix, therefore so is P_inv
        P_inv = cp.Variable((N, N), symmetric=True)
        # H_x is (T x N) matrix
        H_x = cp.Variable((T, N))

        M_x = self._calculate_M_x()
        N0 = self._calculate_N0(M_x)
        dMdx = self.calculate_dMdx(M_x)

        schur = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T

        # Add the constraints
        constraints = [P_inv >> 0, N0 @ H_x == P_inv, schur << 0]

        # Solve for P_inv and H_x
        objective = cp.Minimize(cp.trace(P_inv))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        P = np.linalg.inv(P_inv.value)
        H_x = H_x.value

        return {
            "lyapunov": {"expression": "M(x)^T @ P @ M(x)", "values": {"P": P}},
            "controller": {
                "expression": "U0 @ H(x) @ P @ M(x)",
                "values": {"H(x)": H_x},
            },
        }

    def __discrete_polynomial(self) -> dict:
        N = self.N
        n = self.dimensionality
        T = self.num_samples
        X = self.state_space

        # Theta(x) = N0 @ Q(x)
        # Q(x) = H(x) @ P
        # So, Theta(x) = N0 @ H(x) @ P
        # M(x) = N0 @ H(x) @ P @ x

        M_x = self._calculate_M_x()

        # TODO: refactor to use the solver to calculate Theta_x given M_x and x.
        Theta_x = self._calculate_Theta_x(M_x)
        # Sub in the x1, x2, ..., xN values from the state space, X.
        Theta_x = np.array([
            [t.subs({x: X[f"{x}"][i] for i, x in enumerate(self.x)}) for t in row]
            for row in Theta_x
        ])

        N0 = self._calculate_N0(M_x)

        # P is a symmetric positive definite (n x n) matrix, therefore so is P_inv
        P_inv = cp.Variable((n, n), symmetric=True)
        H_x = cp.Variable((T, n))

        # Solve for P_inv and H(x) using the following two equations (with Mosek):
        # (1) N0 @ H(x) = Theta(x) @ P_inv
        # (2) [[P_inv, H(x).T @ X1.T], [X1 @ H(x), P_inv]] >> 0

        schur = cp.bmat([[P_inv, H_x.T @ self.X1.T], [self.X1 @ H_x, P_inv]])

        constraints = [P_inv >> 0, N0 @ H_x == Theta_x @ P_inv, schur >> 0]

        objective = cp.Minimize(cp.trace(P_inv))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return {
            "lyapunov": {"expression": "x^T @ P @ x", "values": {"P": "P"}},
            "controller": {
                "expression": "U0 @ H @ P^{-1} @ x",
                "values": {"H": "H"},
            },
        }

    def calculate_dMdx(self, M_x):
        dMdx = np.array([[m.diff(x) for x in self.x] for m in M_x])
        # x is a list of symbols for the state space, from x1 to xN
        # Substitute the initial conditions, X0, into the expression
        dMdx = np.array(
            [
                [
                    d.subs({x: self.X0.T[i][j] for j, x in enumerate(self.x)})
                    for d in row
                ]
                for i, row in enumerate(dMdx)
            ]
        )
        return dMdx

    def _calculate_M_x(self):
        M_x = [sp.sympify(m) for m in self.monomials]
        return M_x

    def _calculate_N0(self, M_x):
        """
        N0 is an (N x T) full row-rank matrix.
        N0 = [M(x(0)), M(x(1)), M(x(2)), ..., M(x(T − 1))]
        """
        N0 = []

        for k in range(self.num_samples):
            N0.append([m.subs({x: self.X0.T[k][i] for i, x in enumerate(self.x)}) for m in M_x])

        return np.array(N0).T

    def _calculate_Theta_x(self, M_x: list[sp.Expr]) -> list:
        """
        Theta_x is an (N x n) matrix.
        """
        Theta_x = {}
        for i, expr in enumerate(M_x):
            Theta_x[expr] = [expr.coeff(x) for x in self.x]

        return list(Theta_x.values())

    def _discrete_constraints(self) -> array:
        # block_matrix = cp.bmat([[Z, X1 @ H], [H.T @ X1.T, Z]])
        #
        # return [Z >> 0, Z == X0 @ H, block_matrix >> 0]
        X0 = self.X0
        X1 = self.X1

        n = self.X0.shape[0]
        T = self.X0.shape[1]

        problem = SOSProblem()

        X0 = Constant('X0', X0)
        X1 = Constant('X1', X1)

        H = RealVariable('H', (T, n))
        Z = SymmetricVariable('Z', (n, n))

        problem.add_constraint(Z == X0 * H)
        # Z must be positive definite
        problem.add_constraint(Z - 1.0e-6 * I(n) >> 0)

        schur = ((Z & H.T * X1.T) // (X1 * H & Z))
        # schur = ((Z & X1 * H) // (H.T * X1.T & Z))
        problem.add_constraint(schur >> 0)

        problem.solve(solver='mosek')

        return H, Z

    def _continuous_constraints(self) -> array:
        # eqn = X1 @ H + H.T @ X1.T
        #
        # return [Z >> 0, Z == X0 @ H, eqn << 0]
        X0 = self.X0
        X1 = self.X1

        n = self.X0.shape[0]
        T = self.X0.shape[1]

        problem = SOSProblem()

        # -- Solve for H and Z

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        H = RealVariable('H', (T, n))
        Z = SymmetricVariable('Z', (n, n))

        problem.add_constraint(H.T * X1.T + X1 * H << 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        problem.solve(solver='mosek')

        return H, Z

    # --- Properties ---

    @property
    def dimensionality(self):
        """
        Return the dimensionality in the state space, n
        """
        return len(self.state_space)

    @property
    def N(self) -> int:
        """
        Return the number of monomial terms, N
        """
        return len(self.monomials)

    @property
    def num_samples(self) -> int:
        """
        Return the number of samples, T
        """
        return self.X0.shape[1]

    @property
    def x(self) -> list[sp.Symbol]:
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """
        return sp.symbols(f"x1:{self.dimensionality + 1}")

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

    @staticmethod
    def parse_dataset(data: list) -> np.array:
        """
        Get the initial state of the system as a numpy array of floats
        """

        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = float(data[i][j])

        return np.array(data)

