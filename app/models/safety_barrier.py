import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem, poly_variable, matrix_variable
from picos import Constant, I, RealVariable, SolutionFailure, SymmetricVariable
from picos.expressions.data import cvxopt_inverse
from sympy import Identity, Inverse, MatAdd, Matrix, eye, simplify

from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    def __init__(self, data: dict):
        if data['mode'] != 'Safety':
            raise ValueError(f"Invalid mode '{data['mode']}' for Safety Barrier calculations.")

        super().__init__(data)

        self.problem = SOSProblem()

    def calculate(self):
        results = None

        if self.timing == 'Discrete-Time':
            results = self._discrete_system()
        elif self.timing == 'Continuous-Time':
            results = self._continuous_system()
        else:
            raise ValueError(f"Invalid timing '{self.timing}' for Safety Barrier calculations.")

        return results

    def _discrete_system(self):
        if self.model == 'Linear':
            return self._discrete_linear_system()
        elif self.model == 'Non-Linear Polynomial':
            return self._discrete_nonlinear_system()
        else:
            raise ValueError(f"Invalid model '{self.model}' for Safety Barrier calculations.")

    def _continuous_system(self):
        if self.model == 'Linear':
            return self._continuous_linear_system()
        elif self.model == 'Non-Linear Polynomial':
            return self._continuous_nonlinear_system()
        else:
            raise ValueError(f"Invalid model '{self.model}' for Safety Barrier calculations.")

    def _discrete_linear_system(self):
        problem = SOSProblem()

        U = None

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X0', self.X1)

        # -- Solve for H and Z

        H = RealVariable('H', (self.X0.shape[1], self.dimensions))
        Z = SymmetricVariable('Z', (self.dimensions, self.dimensions))

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        schur = ((Z & H.T * X1.T) // (X1 * H & Z))
        problem.add_constraint(schur >> 0)

        problem.solve(solver='mosek')

        H = Matrix(H)
        Z = Matrix(Z)

        P = Z.inv()
        P_inv = Z

        # -- Level set constraints

        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)

        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        # -- Lagrangian polynomials

        L_init = matrix_variable('l_init', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
        g_init = self.generate_polynomial(self.initial_state.values())
        Lg_init = sum(L_init @ g_init)

        Lg_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe = matrix_variable(f'l_unsafe_{i}', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
            g_unsafe = self.generate_polynomial(self.unsafe_states[i].values())
            Lg_unsafe_set.append(sum(L_unsafe @ g_unsafe))

        L = matrix_variable('l', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
        g = self.generate_polynomial(self.state_space.values())
        Lg = sum(L @ g)

        # -- SOS constraints

        barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe + lambda_, x)

        schur = Matrix(schur)
        Lg_matrix = Matrix(np.full(schur.shape, Lg))
        self.problem.add_matrix_sos_constraint(schur - Lg_matrix, list(x))

        try:
            self.problem.solve()
        except SolutionFailure as e:
            return {
                'error': 'Failed to solve the problem.',
                'description': str(e)
            }
        except Exception as e:
            return {
                'error': 'An unknown error occurred.',
                'description': str(e)
            }

        # TODO: output the simplified version: sp.simplify(barrier[0])

        return {
            'barrier': {
                'expression': 'x^T @ P @ x', 'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ H @ P @ x',
                'values': {'H': H}
            },
            'gamma': gamma_var.value,
            'lambda': lambda_var.value
        }

    def _discrete_nonlinear_system(self):
        return {
            'error': 'Not implemented yet.'
        }

    def _continuous_linear_system(self):
        problem = SOSProblem()

        U = None

        # -- Solve for H and Z

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X0', self.X1)

        H = RealVariable('H', (self.X0.shape[1], self.dimensions))
        Z = SymmetricVariable('Z', (self.dimensions, self.dimensions))

        problem.add_constraint(H.T * X1.T + X1 * H << 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        problem.solve(solver='mosek')

        H = Matrix(H)
        Z = Matrix(Z)

        P = Z.inv()
        P_inv = Z

        # -- Solve for Q
        Q = H @ P

        # TODO: Assert I = X0 @ Q? (It is, up to 10^-6)

        # -- Level set constraints

        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)

        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        # -- Lagrangian polynomials

        L_init = matrix_variable('l_init', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
        g_init = self.generate_polynomial(self.initial_state.values())
        Lg_init = sum(L_init @ g_init)

        Lg_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe = matrix_variable(f'l_unsafe_{i}', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
            g_unsafe = self.generate_polynomial(self.unsafe_states[i].values())
            Lg_unsafe_set.append(sum(L_unsafe @ g_unsafe))

        L = matrix_variable('l', list(x), 0, dim=(self.X0.shape[1], self.dimensions), hom=False, sym=False)
        g = self.generate_polynomial(self.state_space.values())
        Lg = sum(L @ g)

        # -- SOS constraints

        barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe + lambda_, x)

        schur = self.X1 @ Q + Q.T @ self.X1.T
        Lg_matrix = Matrix(np.full(schur.shape, Lg))
        self.problem.add_matrix_sos_constraint(-schur - Lg_matrix, list(x))

        # -- Solve
        self.problem.solve(solver='mosek')

        return {
            'barrier': {
                'expression': 'x^T @ P @ x',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ Q @ x',
                'values': {'Q': Q},
            },
            'gamma': gamma_var.value,
            'lambda': lambda_var.value
        }
        # gamma = sp.symbols('gamma')
        # lambda_ = sp.symbols('lambda')
        #
        # # 2.4 Continuous-time Linear System Barrier
        # # eqn 9: I = X0 @ Q, where I is the identity matrix and X0 is given
        # # eqn 12: −[ X1 @ Q + Q_T @ X1_T ] − L_T(x) @ g(x), where X1 is given, L is the Lagrangian, and g is known
        #
        # gamma_var, lambda_var = self._add_level_set_constraints(gamma, lambda_)
        # barrier_constraint = self._add_lagrangian_constraints(gamma, lambda_)
        #
        # x = self.x
        #
        # Q = matrix_variable('q', list(x), 0, dim=(self.X1.cols, self.dimensions), hom=False, sym=False)
        #
        # # ct_lyapunov: sp.MutableDenseMatrix = self.X1 @ Q + Q.T @ self.X1.T
        # #
        # # ct_vals = np.array(ct_lyapunov.values())
        # #
        # # condition3_set = -ct_vals - sum(L_g)
        # # for condition3 in condition3_set:
        # #     self.problem.add_sos_constraint(condition3, x)
        #
        # self.problem.require(self.X0 @ Q == sp.eye(self.dimensions))
        #
        # try:
        #     self.problem.solve(solver='mosek')
        # except SolutionFailure as e:
        #     raise ValueError(f"Failed to solve problem: {e}")
        # except Exception as e:
        #     raise ValueError(f"An unknown error occurred: {e}")
        #
        # # TODO: return the values
        # P = sum(barrier_constraint.get_sos_decomp())
        # U = None
        # Q = None
        #
        # return {
        #     'barrier': {'expression': 'x^T @ P @ x', 'values': {'P': P}, },
        #     'controller': {'expression': 'U_{0,T} @ Q @ x', 'values': {'U': U, 'Q': Q}, }, 'gamma': gamma_var,
        #     'lambda': lambda_var
        # }

    def _add_level_set_constraints(self, gamma, lambda_):
        gamma_var = self.problem.sym_to_var(gamma)
        self.problem.require(gamma_var > 0)

        lambda_var = self.problem.sym_to_var(lambda_)
        self.problem.require(lambda_var > 0)

        self.problem.require(lambda_var - gamma_var > 0)

        return gamma_var, lambda_var

    def _add_matrix_sos_const(self, prob, name, mat, variables):
        p, vs = self._matrix_sos_poly(name, mat)
        return prob.add_sos_constraint(p, vs + variables)

    @staticmethod
    def _matrix_sos_poly(name, mat):
        """Returns a polynomial that must be sum of squares for MAT to be sum of
        squares. This polynomial is defined using auxiliary variables with NAME,
        which are also returned.
        """
        n, m = mat.shape
        assert n == m, 'Matrix must be square!'
        # TODO: check that matrix is symmetric
        aux_variables = sp.symbols(f'{name}_:{n}')
        x = sp.Matrix([aux_variables])

        return (x @ mat @ x.T)[0], list(aux_variables)
