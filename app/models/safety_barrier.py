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

    def _calculate_lagrangian(self) -> list:
        # TODO: what parameters need to be passed per lagrangian?
        return [poly_variable(f'L{i}', self.state_space, self.degree) for i in range(len(self.state_space))]

    def _add_level_set_constraints(self, gamma, lambda_):
        gamma_var = self.problem.sym_to_var(gamma)
        self.problem.require(gamma_var > 0)

        lambda_var = self.problem.sym_to_var(lambda_)
        self.problem.require(lambda_var > 0)

        self.problem.require(lambda_var - gamma_var > 0)

        return gamma_var, lambda_var

    def _add_lagrangian_constraints(self, gamma, lambda_) -> SOSConstraint:
        # barrier = x^T @ P @ x
        barrier = poly_variable('barrier', self.state_space, self.degree)
        lie_derivative = np.array([sp.diff(barrier, xi) for xi in self.state_space])

        # --- Lagrangian's ---
        # TODO: pass relevant params
        L = self._calculate_lagrangian()
        L_init = self._calculate_lagrangian()
        # TODO: support multiple unsafe regions
        L_unsafe = self._calculate_lagrangian()

        g = Barrier.generate_polynomial(self.state_space)
        g_init = Barrier.generate_polynomial(self.initial_state)
        g_unsafe = [Barrier.generate_polynomial(unsafe_state) for unsafe_state in self.unsafe_states]

        L_init_G_init = [L * g for L, g in zip(L_init, g_init)]
        # TODO: support multiple unsafe regions
        L_unsafe_G_unsafe = [L * g for L, g in zip(L_unsafe, g_unsafe)]
        L_G = [L * g for L, g in zip(L, g)]

        [self.problem.require(i, self.state_space) for i in L]
        # TODO: support multiple unsafe regions
        [self.problem.require(i, self.state_space) for i in L_init]
        [self.problem.require(i, self.state_space) for i in L_unsafe]

        self.problem.require(-barrier - sum(L_init_G_init) + gamma, self.state_space)
        self.problem.require(barrier - sum(L_unsafe_G_unsafe) - lambda_, self.state_space)
        self.problem.require(-np.sum(lie_derivative * f) - sum(L_G), self.state_space)

        barrier_constraint = self.problem.require(barrier, self.state_space)

        return barrier_constraint
