from picos import RealVariable, SolutionFailure
from SumOfSquares import poly_variable, SOSProblem, SOSConstraint, matrix_variable
from sympy import Add, Mul, Sum, Identity, MatAdd, MatMul, MatrixSymbol, Matrix
from typing import List, Union
import cvxpy as cp
import numpy as np
import picos as pc
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
        P = None
        Z = None
        H = None
        U = None

        x = self.x()

        # 2.2 Discrete-time Linear System Barrier
        # We now introduce $L_I(x),L_U(x),L(x)$ as vectors of SOS polynomials
        # (probably use degree $2$ so also quadratic, could become a user chosen term later).
        # We can solve the following using the SumOfSquares toolbox where we also
        # add an additional constraint for the equality.
        # We consider matrix $H\in\reals^{T\times n}$ and symmetric positive definite matrix $P\in\reals^{n\times n}$
        # such that P_inverse = X0 @ H, and the following are sum-of-squares:
        # 1. -x.T @ P @ x - L_init_T(x) @ g_init(x) + gamma
        # 2. x.T @ P @ x - L_unsafe_T(x) @ g_unsafe(x) + lambda
        # 3. Matrix([[ P_inverse, H.T @ X1.T], [X1 @ H, P_inverse]]) - L_T(x) @ g(x)
        # We can define P_inverse as Z, i.e Z = X0 @ H, and
        # 4. Matrix([[ Z, H.T @ X1.T], [X1 @ H, Z]]) >> 0

        # initial_problem = pc.Problem()

        # Define H as a matrix of size T x n
        # H = RealVariable('H', (self.X0.cols, self.dimensions))
        # H = matrix_variable('H', list(x), deg=0, dim=(self.X0.cols, self.dimensions))
        # H = cp.Variable((self.X0.cols, self.dimensions))
        H = cp.Variable((self.X0.shape[1], self.dimensions))

        # Define Z as a matrix of size n x n
        # Z = RealVariable('Z', (self.dimensions, self.dimensions))
        Z = cp.Variable((self.dimensions, self.dimensions), symmetric=True)
        # Constrain matrix Z as positive definite (where all eigenvalues are greater than 10^-6)
        # initial_problem.add_constraint(Z >> 0)
        # Z = matrix_variable('Z', list(x), 0, self.dimensions)
        # Z_sym = MatrixSymbol('Z', self.dimensions, self.dimensions)
        # self.problem.add_matrix_sos_constraint(mat=Z - (10 ** -6) * sp.eye(self.dimensions), variables=list(x))
        # Constrain Z = X0 @ H
        # initial_problem.add_constraint(Z == self.X0 @ H)
        # self.problem.add_matrix_sos_constraint(mat=Z - self.X0 @ H, variables=list(x))
        # Constrain [[ Z, H.T @ X1.T], [X1 @ H, Z]] >> 0
        dtLS_z = cp.bmat([
            [Z, H.T @ self.X1.T],
            [self.X1 @ H, Z]
        ])
        constraints = [Z >> 0, Z == self.X0 @ H, dtLS_z >> 0]

        # initial_problem.add_list_of_constraints(dtLS_z >> 0)
        # matrix_Z_dt_LS = sp.Matrix([[Z, H.T @ self.X1.T], [self.X1 @ H, Z]])
        # Z_constraint = self.problem.add_matrix_sos_constraint(mat=matrix_Z_dt_LS, variables=list(x))

        # Solve for Z and H first, so we can then solve for P
        # solution = initial_problem.solve(solver='mosek')
        Z_problem = cp.Problem(cp.Minimize(cp.trace(Z)), constraints)
        Z_problem.solve()
        z_vals = Z.value

        # solution = self.problem.solve(solver='mosek')
        # decomp = Z_constraint.get_sos_decomp().values()
        # z_vals = solution.ap
        # print(z_vals)

        # Define P as Z_inverse
        # P = Z.inv()
        P = np.linalg.inv(z_vals)
        # Constrain matrix Z as positive definite (where all eigenvalues are greater than 10^-6)
        # self.problem.add_matrix_sos_constraint(mat=P - (10 ** -6) * sp.eye(self.dimensions), variables=list(x))

        # Define gamma and lambda
        gamma = sp.symbols('gamma')
        lambda_ = sp.symbols('lambda')
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)
        # Constrain gamma and lambda
        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        # Calculate the lagrangian's for initial, unsafe and total state space
        L_init = [poly_variable(f'L_init_{i + 1}', x, self.degree) for i in range(len(x))]
        L_unsafe_list = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_list.append([poly_variable(f'L_unsafe_{j}_{i + 1}', x, self.degree) for j in range(len(x))])
        L = [poly_variable(f'L_{i + 1}', x, self.degree) for i in range(len(x))]
        # Generate the polynomials for the state spaces
        g_init = self.generate_polynomial(self.initial_state.values())
        g_unsafe_list = [self.generate_polynomial(unsafe_state.values()) for unsafe_state in self.unsafe_states]
        g = self.generate_polynomial(self.state_space.values())
        # Compute the Lagrangian-polynomial products
        L_init_G_init = Matrix([L * g for L, g in zip(L_init, g_init)])
        L_unsafe_G_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_G_unsafe_set.append([L * g for L, g in zip(L_unsafe_list[i], g_unsafe_list[i])])
        L_unsafe_G_unsafe_set = Matrix(L_unsafe_G_unsafe_set)
        L_G = Matrix([L * g for L, g in zip(L, g)])

        # Add the lagrangian SOS constraints
        barrier = MatMul(MatMul(Matrix(x).T, P), Matrix(x))

        condition1a = MatAdd(barrier + L_init_G_init)
        # condition1 = -barrier - sum(L_init_G_init) + gamma
        # condition1 =

        self.problem.add_sos_constraint(condition1, list(x))
        for L_unsafe_G_unsafe in L_unsafe_G_unsafe_set:
            self.problem.add_sos_constraint(barrier - sum(L_unsafe_G_unsafe) + lambda_, x)
        matrix_dt_LS = matrix_Z_dt_LS - sum(L_G)
        # self.problem.add_matrix_sos_constraint()

        # Solve for P, gamma and lambda given the SOS barrier constraints.

        # --- Level Set Constraints ---

        # --- SOS Constraints ---
        self.problem.add_matrix_sos_constraint(mat=H, variables=list(x))
        self.problem.add_matrix_sos_constraint(mat=P - (10 ** -6) * sp.eye(self.dimensions), variables=list(x))

        barrier_constraint = self._add_lagrangian_constraints(gamma, lambda_)

        # Solve for Z and H first, so we can then solve for P

        return {
            'barrier': {
                'expression': 'x^T @ P @ x',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ H @ P @ x',
                'values': {'U': U, 'H': H},
            },
            'gamma': gamma,
            'lambda': lambda_,
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
        # condition3_set = -ct_vals - sum(L_G)
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
            'barrier': {
                'expression': 'x^T @ P @ x',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ Q @ x',
                'values': {'U': U, 'Q': Q},
            },
            'gamma': gamma_var,
            'lambda': lambda_var
        }

    def _add_level_set_constraints(self, gamma, lambda_):
        gamma_var = self.problem.sym_to_var(gamma)
        self.problem.require(gamma_var > 0)

        lambda_var = self.problem.sym_to_var(lambda_)
        self.problem.require(lambda_var > 0)

        self.problem.require(lambda_var - gamma_var > 0)

        return gamma_var, lambda_var

    def _add_lagrangian_constraints(self, gamma, lambda_) -> SOSConstraint:
        x = self.x()

        P = matrix_variable('P', list(x), 0, self.dimensions)
        self._add_matrix_sos_const(self.problem, 'P_const', P - (10 ** -6) * sp.eye(self.dimensions), list(x))

        barrier: sp.Matrix = sp.Matrix(x).T @ P @ sp.Matrix(x)
        vals = barrier.values()[0]
        # lie_derivative = np.array([sp.diff(barrier, xi) for xi in x])

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
        L_unsafe_G_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe_G_unsafe_set.append([L * g for L, g in zip(L_unsafe_list[i], g_unsafe_list[i])])

        [self.problem.add_sos_constraint(i, x) for i in L]
        [self.problem.add_sos_constraint(i, x) for i in L_init]
        for L_unsafe in L_unsafe_list:
            [self.problem.add_sos_constraint(i, x) for i in L_unsafe]

        barrier_vals = barrier.values()[0]
        sum_L_init_G_init = sum(L_init_G_init)
        condition1 = -barrier_vals - sum_L_init_G_init + gamma

        self.problem.add_sos_constraint(condition1, x)
        for L_unsafe_G_unsafe in L_unsafe_G_unsafe_set:
            condition2 = barrier.values()[0] - sum(L_unsafe_G_unsafe) - lambda_
            self.problem.add_sos_constraint(condition2, x)
        # self.problem.add_sos_constraint(-np.sum(lie_derivative * f) - sum(L_G), x)

        barrier_constraint = self.problem.add_sos_constraint(barrier.values()[0], x)

        return barrier_constraint

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
