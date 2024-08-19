from sympy.matrices.expressions import Identity, MatrixSymbol
from SumOfSquares import poly_variable, SOSProblem, SOSConstraint
from picos.modeling.problem import SolutionFailure
import numpy as np
import sympy as sp

from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    def __init__(self, data: dict):
        if data['mode'] != 'Safety':
            raise ValueError(f"Invalid mode '{data['mode']}' for Safety Barrier calculations.")

        super().__init__(data)

        self.problem = SOSProblem()

    def calculate(self):
        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')

        # 2.4 Continuous-time Linear System Barrier
        # eqn 9: I = X0 @ Q, where I is the identity matrix and X0 is given
        # eqn 12: −[ X1 @ Q + Q_T @ X1_T ] − L_T(x) @ g(x), where X1 is given, L is the Lagrangian, and g is known

        gamma_var, lambda_var = self._add_level_set_constraints(gamma, lambda_)
        barrier_constraint = self._add_lagrangian_constraints(gamma, lambda_)

        try:
            self.problem.solve(solver='mosek')
        except SolutionFailure as e:
            raise ValueError(f"Failed to solve problem: {e}")
        except Exception as e:
            raise ValueError(f"An unknown error occurred: {e}")

        # TODO: return the values
        P = sum(barrier_constraint.get_sos_decomp())
        U = None
        Q = None

        return super().result(P, U, Q, float(gamma_var), float(lambda_var))

    def _add_level_set_constraints(self, gamma, lambda_):
        gamma_var = self.problem.sym_to_var(gamma)
        self.problem.require(gamma_var > 0)

        lambda_var = self.problem.sym_to_var(lambda_)
        self.problem.require(lambda_var > 0)

        self.problem.require(lambda_var - gamma_var > 0)

        return gamma_var, lambda_var

    def _add_lagrangian_constraints(self, gamma, lambda_) -> SOSConstraint:
        x = self.x()

        # barrier = x^T @ P @ x
        barrier = poly_variable('barrier', x, self.degree)
        lie_derivative = np.array([sp.diff(barrier, xi) for xi in x])

        # --- Lagrangian's ---
        L = [poly_variable(f'L_{i + 1}', x, self.degree) for i in range(len(x))]
        L_init = [poly_variable(f'L_init_{i + 1}', x, self.degree) for i in range(len(x))]
        L_unsafe_list = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_list.append([poly_variable(f'L_unsafe_{j}_{i + 1}', x, self.degree) for j in range(len(x))])

        g = self.generate_polynomial(self.state_space.values())
        g_init = self.generate_polynomial(self.initial_state.values())
        g_unsafe_list = [self.generate_polynomial(unsafe_state.values()) for unsafe_state in self.unsafe_states]

        L_G = [L * g for L, g in zip(L, g)]
        L_init_G_init = [L * g for L, g in zip(L_init, g_init)]
        L_unsafe_G_unsafe = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_G_unsafe.append([L * g for L, g in zip(L_unsafe_list[i], g_unsafe_list[i])])

        [self.problem.require(i, x) for i in L]
        [self.problem.require(i, x) for i in L_init]
        for L_unsafe in L_unsafe_list:
            [self.problem.require(i, x) for i in L_unsafe]

        self.problem.require(-barrier - sum(L_init_G_init) + gamma, x)
        self.problem.require(barrier - sum(L_unsafe_G_unsafe) - lambda_, x)
        self.problem.require(-np.sum(lie_derivative * f) - sum(L_G), x)

        barrier_constraint = self.problem.require(barrier, x)

        return barrier_constraint
