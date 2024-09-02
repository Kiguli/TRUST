import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem, matrix_variable, poly_variable
from picos import Constant, I, RealVariable, SolutionFailure, SymmetricVariable
from sympy import Identity, MatAdd, Matrix

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

    def _discrete_linear_system(self):
        U = None

        x = self.x()
        X0 = Constant('X0', self.X0)
        X1 = Constant('X0', self.X1)

        # -- Solve for Z and H

        # Define H as a matrix of size T x n
        H = RealVariable('H', (self.X0.shape[1], self.dimensions))
        # Define Z as a matrix of size n x n
        Z = SymmetricVariable('Z', (self.dimensions, self.dimensions))

        self.problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        self.problem.add_constraint(Z == X0 * H)

        schur = ((Z & H.T * X1.T) // (X1 * H & Z))
        self.problem.add_constraint(schur >> 0)

        self.problem.solve()

        # -- Solve for P

        P_inv = Matrix(Z)
        P = Matrix(Z).inv()
        self.problem.add_matrix_sos_constraint(P - (10 ** -6) * sp.eye(self.dimensions), list(x))

        # Define gamma and lambda
        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)
        # Constrain gamma and lambda
        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        barrier: Matrix = sp.Matrix(x).T @ P @ sp.Matrix(x)

        # Condition 1: Initial state

        L_init = [poly_variable(f'L_init_{i + 1}', x, self.degree) for i in range(len(x))]
        g_init = self.generate_polynomial(self.initial_state.values())
        L_init_g_init = sum([L * g for L, g in zip(L_init, g_init)])

        gamma_vec = gamma * Identity(1)
        L_init_g_init_vec = L_init_g_init * Identity(1)
        condition1 = (-MatAdd(barrier, L_init_g_init_vec) + gamma_vec)[0]
        self.problem.add_sos_constraint(condition1, x)

        # Condition 2: Unsafe states

        L_unsafe_list = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_list.append([poly_variable(f'L_unsafe_{j}_{i + 1}', x, self.degree) for j in range(len(x))])
        # Generate the polynomials for the state spaces
        g_unsafe_list = [self.generate_polynomial(unsafe_state.values()) for unsafe_state in self.unsafe_states]
        L_unsafe_g_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_g_unsafe_set.append(sum([L * g for L, g in zip(L_unsafe_list[i], g_unsafe_list[i])]))

        for L_unsafe_g_unsafe in L_unsafe_g_unsafe_set:
            lambda_vec = lambda_ * Identity(1)
            L_unsafe_g_unsafe_vec = L_unsafe_g_unsafe * Identity(1)
            condition2 = (MatAdd(barrier, -L_unsafe_g_unsafe_vec) - lambda_vec)[0]
            self.problem.add_sos_constraint(condition2, x)

        # Condition 3: State space

        L = Matrix([poly_variable(f'L_{i + 1}', x, self.degree) for i in range(len(x))])
        g = Matrix(self.generate_polynomial(self.state_space.values()))
        L_g = (L.T @ g)[0]

        schur_arr = np.array(schur)
        condition3 = schur_arr - L_g
        self.problem.add_matrix_sos_constraint(condition3, list(x))

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
                'values': {'U': U, 'H': H}
            },
            'gamma': gamma,
            'lambda': lambda_
        }

    def _discrete_nonlinear_system(self):
        return {
            'error': 'Not implemented yet.'
        }

    def _continuous_system(self):
        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')

        # 2.4 Continuous-time Linear System Barrier
        # eqn 9: I = X0 @ Q, where I is the identity matrix and X0 is given
        # eqn 12: −[ X1 @ Q + Q_T @ X1_T ] − L_T(x) @ g(x), where X1 is given, L is the Lagrangian, and g is known

        gamma_var, lambda_var = self._add_level_set_constraints(gamma, lambda_)
        barrier_constraint = self._add_lagrangian_constraints(gamma, lambda_)

        x = self.x()

        Q = matrix_variable('q', list(x), 0, dim=(self.X1.cols, self.dimensions), hom=False, sym=False)

        # ct_lyapunov: sp.MutableDenseMatrix = self.X1 @ Q + Q.T @ self.X1.T
        #
        # ct_vals = np.array(ct_lyapunov.values())
        #
        # condition3_set = -ct_vals - sum(L_g)
        # for condition3 in condition3_set:
        #     self.problem.add_sos_constraint(condition3, x)

        self.problem.require(self.X0 @ Q == sp.eye(self.dimensions))

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

        return {
            'barrier': {'expression': 'x^T @ P @ x', 'values': {'P': P}, },
            'controller': {'expression': 'U_{0,T} @ Q @ x', 'values': {'U': U, 'Q': Q}, }, 'gamma': gamma_var,
            'lambda': lambda_var
        }

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
