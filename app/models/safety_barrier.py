import cvxpy as cp
import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem, matrix_variable
from picos import Constant, I, RealVariable, SolutionFailure, SymmetricVariable
from sympy import Matrix, simplify, symbols, sympify

from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    def __init__(self, data: dict):
        # TODO: migrate to builder pattern
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
            return self._discrete_linear()
        elif self.model == 'Non-Linear Polynomial':
            return self._discrete_nps()
        else:
            raise ValueError(f"Invalid model '{self.model}' for Safety Barrier calculations.")

    def _continuous_system(self):
        if self.model == 'Linear':
            return self._continuous_linear()
        elif self.model == 'Non-Linear Polynomial':
            return self._continuous_nps()
        else:
            raise ValueError(f"Invalid model '{self.model}' for Safety Barrier calculations.")

    def _discrete_linear(self):
        problem = SOSProblem()

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        # -- Solve for H and Z

        H = RealVariable('H', (self.X0.shape[1], self.dimensionality))
        Z = SymmetricVariable('Z', (self.dimensionality, self.dimensionality))

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        schur = ((Z & H.T * X1.T) // (X1 * H & Z))
        problem.add_constraint(schur >> 0)

        problem.solve(solver='mosek')

        H = Matrix(H)
        Z = Matrix(Z)

        P = Z.inv()

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])

        # -- SOS constraints

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

        # TODO: output the simplified version: sp.simplify(barrier[0])?

        P = np.array2string(np.array(P))
        H = np.array2string(np.array(H))

        return {
            'barrier': {
                'expression': 'x^T @ P @ x', 'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ H @ P @ x',
                'values': {'H': H}
            },
            'gamma': str(gamma_var.value),
            'lambda': str(lambda_var.value)
        }

    def _discrete_nps(self):
        problem = SOSProblem()

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        # Q(x) is a (T x n) matrix polynomial such that Theta(x) = N0 @ Q(x)
        # Theta(x) is an (N x n) matrix polynomial, M(x) = Theta(x) @ x
        # N0 is an (N x T) full row rank matrix, N0 = [M(x(0)), M(x(1)), ..., M(x(T-1))]

        # -- Solve for H and Z

        Hx = matrix_variable('Hx', list(x), self.degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)

        Z = SymmetricVariable('Z', self.dimensionality)

        # schur = (Z & Hx.T @ self.X1.T) // (self.X1 @ Hx & Z)
        schur = Matrix([
            [Z, Hx.T @ self.X1.T],
            [self.X1 @ Hx, Z]
        ])
        problem.require(schur >> 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(list(x)).shape[1]) >> 0)
        problem.require()

        problem.solve(solver='mosek')

        Z = Matrix(Z)

        P = Z.inv()
        P_inv = Z

        # Q(x) = H(x) @ P
        # Q(x).T @ X1.T @ P @ X1 @ Q(x) <= P

        # N0 @ H(x) = Theta(x) @ P_inv

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])

        # -- SOS constraints

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe + lambda_, x)

        Hx_var = H_var @ Matrix(x)
        schur_matrix = Matrix([
            [Z_var, Hx_var.T @ self.X1.T],
            [self.X1 @ Hx_var, Z_var]
        ])
        Lg_matrix = Matrix(np.full(schur_matrix.shape, Lg))
        self.problem.add_matrix_sos_constraint(schur_matrix - Lg_matrix, list(x))

        self.problem.solve()

        return {
            'barrier': {
                'expression': 'x^T @ P @ x',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U0 @ H(x) @ [N0 @ H(x)]^-1 @ x',
                'values': {
                    'H': H,
                    'N': N
                }
            },
        }

    def _continuous_linear(self):
        problem = SOSProblem()

        U = None

        # -- Solve for H and Z

        x = self.x
        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        H = RealVariable('H', (self.X0.shape[1], self.dimensionality))
        Z = SymmetricVariable('Z', (self.dimensionality, self.dimensionality))

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

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])

        # -- SOS constraints

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

    def _continuous_nps(self):
        """
        Solve for a continuous non-linear polynomial system.
        """

        # TODO: approximate X1 as the derivatives of the state at each sampling time, if not provided.

        # Monomials M(x) = ['x1', 'x2', 'x1*x2', 'x2-x1']
        # N0 = [M(x(0)), M(x(tau)), M(x(2*tau)), ..., M(x((T-1)*tau))]
        # N0 is a (N x T) matrix, where N is the number of monomial terms and T is the number of samples.

        N = self.N
        n = self.dimensionality
        T = self.num_samples
        mon_syms = self.x

        # Create symbolic expressions for each monomial and tau
        M_x = [sympify(m) for m in self.monomials]
        tau = symbols('tau')  # TODO: allow user to specify tau?

        N0 = []
        for state in self.X0.T.tolist():
            row = []
            for m in M_x:
                value = m.subs({x: state[i] for i, x in enumerate(mon_syms)})
                row.append(value)
            N0.append(row)

        N0 = np.array(N0).T

        # --- (1) First, solve P^-1 = N0 @ H(x) and -[dMdx @ X1 @ H(x) + H(x).T @ X1.T @ dMdx.T] >= 0 ---
        # Note: Z = P^-1

        Z = cp.Variable((N, N), symmetric=True)

        # Compute dMdx
        dMdx = np.array([[m.diff(x) for x in self.x] for m in M_x])
        # Sub in the values of X0 for x1 and x2
        dMdx = np.array([[
            d.subs({x: self.X0.T[i][j] for j, x in enumerate(mon_syms)}) for d in row
        ] for i, row in enumerate(dMdx)])

        H_x = cp.Variable((T, N))

        # Add the constraints
        constraint1 = N0 @ H_x == Z
        schur = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T
        constraint2 = schur << 0

        # Solve for the matrices Z and H_x
        objective = cp.Minimize(cp.trace(Z))
        constraints = [constraint1, constraint2, Z >> 0]
        problem = cp.Problem(objective, constraints)

        problem.solve()

        Z = Z.value
        H_x = H_x.value
        schur = schur.value

        # --- (2) Then, solve SOS conditions for gamma and lambda ---

        P = np.linalg.inv(Z)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = np.array(M_x).T @ P @ M_x

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)
        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe + lambda_, self.x)
        Lg_matrix = Matrix(np.full(schur.shape, Lg))
        self.problem.add_matrix_sos_constraint(-schur - Lg_matrix, list(self.x))

        try:
            self.problem.solve(solver='mosek')
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

        # Q(x) = H(x) @ P
        #controller = U0 @ Hx @ P @ self.M_x

        return {
            'barrier': {
                'expression': 'M(x)^T @ P @ M(x)',
                'values': {'P': P}
            },
            'controller': {
                'expression': 'U0 @ H(x) @ P @ M(x)',
                'values': {
                    'H(x)': H_x
                }
            },
            'gamma': gamma_var.value,
            'lambda': lambda_var.value
        }

    def __level_set_constraints(self):
        gamma, lambda_ = sp.symbols('gamma lambda')
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)

        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        return gamma, lambda_, gamma_var, lambda_var

    def __compute_lagrangians(self):
        x = self.x

        # TODO: remove (degree 0 for faster feedback, invalid maths)
        degree = self.degree
        degree = 0

        L_init = matrix_variable('l_init', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        g_init = self.generate_polynomial(self.initial_state.values())
        Lg_init = sum(L_init @ g_init)

        Lg_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            L_unsafe = matrix_variable(f'l_unsafe_{i}', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False,
                                       sym=False)
            g_unsafe = self.generate_polynomial(self.unsafe_states[i].values())
            Lg_unsafe_set.append(sum(L_unsafe @ g_unsafe))

        L = matrix_variable('l', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        g = self.generate_polynomial(self.state_space.values())
        Lg = sum(L @ g)

        return Lg_init, Lg_unsafe_set, Lg
