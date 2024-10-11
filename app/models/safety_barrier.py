import cvxpy as cp
import numpy as np
import picos as pc
import sympy as sp
from SumOfSquares import SOSProblem, matrix_variable, poly_variable
from cvxpy.expressions.cvxtypes import problem
from picos import Constant, I, Problem, RealVariable, SolutionFailure, SymmetricVariable
from sympy import Matrix, symbols, sympify

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
        T = self.num_samples
        n = self.dimensionality
        x = self.x

        # -- Part 1: Solve for H and Z

        problem = SOSProblem()

        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        H = RealVariable('H', (T, n))
        Z = SymmetricVariable('Z', (n, n))

        problem.add_constraint(Z == X0 * H)
        # Z must be positive definite
        problem.add_constraint(Z - 1.0e-6 * I(n) >> 0)

        schur = ((Z & H.T * X1.T) // (X1 * H & Z))
        # schur = ((Z & X1 * H) // (H.T * X1.T & Z))
        problem.add_constraint(schur >> 0)

        problem.solve(solver='mosek')

        # -- Part 2: SOS ---

        # TODO: Consider np.array
        H = Matrix(H)
        Z = Matrix(Z)
        P = Z.inv()

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        # barrier = simplify((Matrix(x).T @ P @ Matrix(x))[0])
        barrier = (Matrix(x).T * P * Matrix(x))[0]
        barrier_constraint = self.problem.add_sos_constraint(barrier, x)

        # -- SOS constraints

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, x))

        # schur = Matrix(schur)
        # Lg_matrix = Matrix(np.full(schur.shape, Lg))
        # condition3 = self.problem.add_matrix_sos_constraint(schur - Lg_matrix, list(x))

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

        # --- Validate the results ---

        barrier_decomp = barrier_constraint.get_sos_decomp()
        first_decomp = condition1.get_sos_decomp()
        second_decomps = [cond.get_sos_decomp() for cond in condition2]
        # third_decomp = condition3.get_sos_decomp()

        isAllPositiveSecondDecomps = all([len(decomp) > 0 for decomp in second_decomps])

        if len(barrier_decomp) <=0 or len(first_decomp) <= 0 or not isAllPositiveSecondDecomps:
            return {
                'error': 'Constraints are not sum-of-squares.'
            }

        if barrier_decomp.free_symbols == 0:
            return {
                'error': 'Barrier is scalar.'
            }

        # TODO: output the simplified barrier: sp.simplify(barrier)? – issue with simplify() not working
        barrier = np.array2string(np.array(simplify(barrier)), separator=', ')

        controller = self.U0 @ H @ P @ Matrix(x)
        controller = np.array2string(np.array(controller), separator=', ')

        P = np.array2string(np.array(P), separator=', ')
        H = np.array2string(np.array(H), separator=', ')


        return {
            'barrier': {
                'expression': barrier, 'values': {'P': P},
            },
            'controller': {
                'expression': controller,
                'values': {'H': H}
            },
            'gamma': str(gamma_var.value),
            'lambda': str(lambda_var.value)
        }

    def _discrete_nps(self):
        # Create the 1st problem.
        problem = Problem()

        # (We're given the monomials M(x) by the user, i.e. self.monomials)
        x = list(self.x)
        M_x = self.M_x
        N = self.N
        n = self.dimensionality

        # TODO: Solve for Theta(x), where M(x) = Theta(x) @ x (use a nested for)
        # TODO: assert M(x) = Theta(x)x

        Theta = RealVariable('Theta', (N, n))
        Theta_x = sum(Theta[i] * x[i] for i in range(N))

        eqn25 = np.array(Theta_x) @ np.array(x) - np.array(M_x).astype(object)

        problem.require(eqn25)

        # M(x) = Theta(x) @ x
        # i.e. Theta(x) = M(x) @ x^-1 (but we can use the solver to find Theta(x) directly)

        # Since we're given M(x) by the user, i.e. self.monomials,
        # we can then use the solver to find Theta(x).
        Theta_x = matrix_variable('Theta_x', list(x), self.degree, dim=(len(self.monomials['terms']), self.dimensionality), hom=False, sym=False)

        problem.require(np.array(self.monomials['terms']) == np.array(np.array(Theta_x) @ np.array(x)))

        # Theta(x) = N0 @ Q(x)
        # i.e. Q(x) = N0^-1 @ Theta(x)

        N0 = self.__compute_N0()

        X0 = Constant('X0', self.X0)
        X1 = Constant('X1', self.X1)

        # Q(x) is a (T x n) matrix polynomial such that Theta(x) = N0 @ Q(x)
        # Theta(x) is an (N x n) matrix polynomial, M(x) = Theta(x) @ x
        # N0 is an (N x T) full row rank matrix, N0 = [M(x(0)), M(x(1)), ..., M(x(T-1))]

        # -- Part 1: Solve for Theta(x) H and Z

        Q_x = matrix_variable('Q_x', list(x), self.degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        Hx = matrix_variable('Hx', list(x), self.degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        Z = SymmetricVariable('Z', (self.dimensionality, self.dimensionality))

        # Add the simultaneous constraints, schur and theta
        schur = (np.array(Z) & np.array(Hx.T) * np.array(self.X1.T)) // (np.array(self.X1 * Hx) & np.array(Z))
        # schur = Matrix([
        #     [Z, Hx.T @ self.X1.T],
        #     [self.X1 @ Hx, Z]
        # ])
        problem.require(schur >> 0)

        # TODO: lagrangian 3rd cond w/ schur

        problem.require(Theta_x @ Z == N0 @ Hx)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(list(x)).shape[1]) >> 0)

        problem.solve(solver='mosek')

        # --- Part 2: SOS ---

        Z = Matrix(Z)
        P = Z.inv()

        self.problem.add_constraint(Q_x == Hx @ P)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = (Matrix(x).T @ P @ Matrix(x))[0]

        # -- SOS constraints

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, x)

        schur_matrix = Matrix([
            [Z, Hx.T @ self.X1.T],
            [self.X1 @ Hx, Z]
        ])
        Lg_matrix = Matrix(np.full(schur_matrix.shape, Lg))
        self.problem.add_matrix_sos_constraint(schur_matrix - Lg_matrix, list(x))

        self.problem.solve()

        P = np.array2string(np.array(P), separator=', ')
        H = np.array2string(np.array(Hx), separator=', ')

        return {
            'barrier': {
                'expression': 'x<sup>T</sup>Px',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U<sub>0</sub>H(x)[N<sub>0</sub>H(x)]<sup>-1</sup>x',
                'values': {
                    'H': H,
                }
            },
        }

    def _continuous_linear(self):
        problem = SOSProblem()

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

        # -- Solve for Q
        Q = H @ P

        # TODO: Assert I = X0 @ Q? (It is, up to 10^-6)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = (Matrix(x).T @ P @ Matrix(x))[0]

        # -- SOS constraints

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, x)

        # schur = self.X1 @ Q + Q.T @ self.X1.T
        # Lg_matrix = Matrix(np.full(schur.shape, Lg))
        # self.problem.add_matrix_sos_constraint(-schur - Lg_matrix, list(x))

        # -- Solve
        self.problem.solve(solver='mosek')

        barrier = np.array2string(np.array(barrier), separator=', ')

        controller = self.U0 @ H @ P @ Matrix(x)
        controller = np.array2string(np.array(controller), separator=', ')

        P = np.array2string(np.array(P), separator=', ')
        H = np.array2string(np.array(H), separator=', ')
        Q = np.array2string(np.array(Q), separator=', ')

        return {
            'barrier': {
                'expression': barrier,
                'values': {'P': P},
            },
            'controller': {
                'expression': controller,
                'values': {'H': H},
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
        x = self.x

        problem = SOSProblem()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        # --- (1) First, solve P^-1 = N0 @ H(x) and -[dMdx @ X1 @ H(x) + H(x).T @ X1.T @ dMdx.T] >= 0 ---

        # Create symbolic expressions for each monomial
        M_x = [sympify(m) for m in self.monomials['terms']]

        N0 = self.__compute_N0()

        H_x = matrix_variable('H_x', list(x), self.degree, dim=(T, N), hom=False, sym=False)
        Z = SymmetricVariable('Z', (N, N))
        # # Change Z from a PICOS expression to a SymPy matrix
        # z_symbols = sp.symbols(f'z:{N}:{N}')
        # Z = sp.Matrix(N, N, z_symbols)

        # Compute dMdx
        dMdx = np.array([[m.diff(x) for x in self.x] for m in M_x])

        # Add the constraints
        constraint1 = problem.add_constraint(N0 @ H_x == Z)
        lie_derivative = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T
        constraint2 = problem.add_matrix_sos_constraint(lie_derivative - sp.Mul(Lg, Matrix(I(N))), list(x))

        problem.solve()

        Z = Matrix(Z)
        H_x = Matrix(H_x)
        lie_derivative = Matrix(lie_derivative)

        # --- (2) Then, solve SOS conditions for gamma and lambda ---

        P = Z.inv()

        # TODO: assert Q_x == H_x @ P (to 10^-6)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = np.array(M_x).T @ P @ M_x

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)
        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)

        try:
            self.problem.solve(solver='mosek')
        except SolutionFailure as e:
            # TODO: include info on what wasn't feasible
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

        # TODO: add SOS decomp (to all)
        # TODO: save out barrier and controller (to all)

        return {
            'barrier': {
                'expression': 'M(x)<sup>T</sup>PM(x)',
                'values': {'P': P}
            },
            'controller': {
                'expression': 'U<sub>0</sub>H(x)PM(x)',
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

        degree = self.degree

        L_init = [poly_variable('Li' + str(i + 1), x, degree) for i in range(len(x))]

        # L_init = matrix_variable('l_init', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        g_init = self.generate_polynomial(self.initial_state)
        Lg_init = sum([L_init * g_init for L_init, g_init in zip(L_init, g_init)])

        Lg_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            # L_unsafe = matrix_variable(f'l_unsafe_{i}', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
            L_unsafe = [poly_variable(f'Lu{i}{j}', x, degree) for j in range(len(x))]
            g_unsafe = self.generate_polynomial(self.unsafe_states[i])
            Lg_unsafe_set.append(sum([L * g for L, g in zip(L_unsafe, g_unsafe)]))

        # L = matrix_variable('l', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        L = [poly_variable('L' + str(i + 1), x, degree) for i in range(len(x))]
        g = self.generate_polynomial(self.state_space)
        Lg = sum([L * g for L, g in zip(L, g)])

        return Lg_init, Lg_unsafe_set, Lg

    def __compute_N0(self) -> list:
        """
        Compute the N0 matrix by evaluating the monomials at each time step.
        """

        T = self.num_samples

        # Initialise the N0 matrix
        N0 = np.zeros((self.N, T))

        for t in range(self.num_samples):
            # Get the x values at time t
            x_t = self.X0[:, t]

            for i in range(self.N):
                expr = sympify(self.monomials['terms'][i])
                N0[i, t] = float(expr.subs({k: val for k, val in zip(self.x, x_t)}))

        return N0
