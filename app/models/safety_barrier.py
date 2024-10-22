from typing import Union, List

import numpy as np
import sympy as sp
from SumOfSquares import (
    Basis,
    SOSConstraint,
    SOSProblem,
    matrix_variable,
    poly_variable,
)
from picos import Constant, I, Problem, RealVariable, SolutionFailure, SymmetricVariable
from picos.constraints import Constraint
from sympy import Matrix, sympify

from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    def __init__(self, data: dict):
        # TODO: migrate to builder pattern?
        if data["mode"] != "Safety":
            raise ValueError(
                f"Invalid mode '{data['mode']}' for Safety Barrier calculations."
            )

        super().__init__(data)

        self.problem = SOSProblem()

    def calculate(self):
        results = None

        if self.timing == "Discrete-Time":
            results = self._discrete_system()
        elif self.timing == "Continuous-Time":
            results = self._continuous_system()
        else:
            raise ValueError(
                f"Invalid timing '{self.timing}' for Safety Barrier calculations."
            )

        return results

    def _discrete_system(self):
        if self.model == "Linear":
            return self._discrete_linear()
        elif self.model == "Non-Linear Polynomial":
            return self._discrete_nps()
        else:
            raise ValueError(
                f"Invalid model '{self.model}' for Safety Barrier calculations."
            )

    def _continuous_system(self):
        if self.model == "Linear":
            return self._continuous_linear()
        elif self.model == "Non-Linear Polynomial":
            return self._continuous_nps()
        else:
            raise ValueError(
                f"Invalid model '{self.model}' for Safety Barrier calculations."
            )

    def _discrete_linear(self):

        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        H = RealVariable("H", (self.num_samples, self.dimensionality))
        Z = SymmetricVariable("Z", (self.dimensionality, self.dimensionality))

        HZ_problem = SOSProblem()

        HZ_problem.add_constraint(Z == X0 * H)
        # Z must be positive definite
        HZ_problem.add_constraint(Z - 1.0e-6 * I(self.dimensionality) >> 0)

        schur = (Z & H.T * X1.T) // (X1 * H & Z)
        HZ_problem.add_constraint(schur >> 0)

        HZ_problem.solve(solver="mosek")

        H = Matrix(H)
        Z = Matrix(Z)
        P = Z.inv()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()
        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = (Matrix(self.x).T * P * Matrix(self.x))[0]
        barrier_constraint = self.problem.add_sos_constraint(barrier, self.x)

        # -- SOS constraints

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(
                self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)
            )

        # schur = Matrix(schur)
        # Lg_matrix = Matrix(np.full(schur.shape, Lg))
        # condition3 = self.problem.add_matrix_sos_constraint(schur - Lg_matrix, list(x))

        self.__solve()

        validation = self.__validate_solution(
            barrier_constraint, condition1, condition2
        )
        if validation != True and "error" in validation:
            return validation

        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H @ P @ Matrix(self.x)
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H = self.__matrix_to_string(H)

        return {
            "barrier": {
                "expression": barrier,
                "values": {"P": P},
            },
            "controller": {"expression": controller, "values": {"H": H}},
            "gamma": str(gamma_var.value),
            "lambda": str(lambda_var.value),
        }

    def _discrete_nps(self):
        x = list(self.x)
        M_x = self.M_x
        N = self.N
        n = self.dimensionality

        HZ_problem = SOSProblem()

        Theta_x = matrix_variable("Theta_x", x, self.degree, dim=(N, n))

        self.__add_matrix_constraint(HZ_problem, M_x - Theta_x @ Matrix(x), x)

        N0 = self.__compute_N0()

        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        # Q(x) is a (T x n) matrix polynomial such that Theta(x) = N0 @ Q(x)
        # Theta(x) is an (N x n) matrix polynomial, M(x) = Theta(x) @ x
        # N0 is an (N x T) full row rank matrix, N0 = [M(x(0)), M(x(1)), ..., M(x(T-1))]

        # -- Part 1: Solve for Theta(x) H and Z

        Q_x = matrix_variable(
            "Q_x",
            list(x),
            self.degree,
            dim=(self.num_samples, n),
            hom=False,
            sym=False,
        )
        H_x = matrix_variable(
            "H_x",
            list(x),
            self.degree,
            dim=(self.num_samples, n),
            hom=False,
            sym=False,
        )
        Z = matrix_variable("Z", list(x), 0, dim=(n, n), hom=False, sym=False)
        # Z = SymmetricVariable('Z', (n, self.dimensionality))

        # Add the simultaneous constraints, schur and theta

        schur = (np.array(Z) & np.array(H_x.T) * np.array(self.X1.T)) // (
            np.array(self.X1 * H_x) & np.array(Z)
        )
        # schur = Matrix([
        #     [Z, H_x.T @ self.X1.T],
        #     [self.X1 @ H_x, Z]
        # ])
        HZ_problem.require(schur >> 0)

        # TODO: lagrangian 3rd cond w/ schur

        HZ_problem.require(Theta_x @ Z == N0 @ H_x)

        HZ_problem.add_constraint(Z - 1.0e-6 * I(Matrix(list(x)).shape[1]) >> 0)

        HZ_problem.solve(solver="mosek")

        # TODO: assert M(x) = Theta(x)x

        # --- Part 2: SOS ---

        Z = Matrix(Z)
        P = Z.inv()

        self.problem.add_constraint(Q_x == H_x @ P)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        barrier = (Matrix(x).T @ P @ Matrix(x))[0]

        # -- SOS constraints

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, x)

        schur_matrix = Matrix([[Z, H_x.T @ self.X1.T], [self.X1 @ H_x, Z]])
        Lg_matrix = Matrix(np.full(schur_matrix.shape, Lg))
        self.problem.add_matrix_sos_constraint(schur_matrix - Lg_matrix, list(x))

        self.__solve()

        P = np.array2string(np.array(P), separator=", ")
        H = np.array2string(np.array(H_x), separator=", ")

        return {
            "barrier": {
                "expression": "x<sup>T</sup>Px",
                "values": {"P": P},
            },
            "controller": {
                "expression": "U<sub>0</sub>H(x)[N<sub>0</sub>H(x)]<sup>-1</sup>x",
                "values": {
                    "H": H,
                },
            },
        }

    def _continuous_linear(self):
        problem = SOSProblem()

        # -- Solve for H and Z

        x = self.x
        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        H = RealVariable("H", (self.X0.shape[1], self.dimensionality))
        Z = SymmetricVariable("Z", (self.dimensionality, self.dimensionality))

        problem.add_constraint(H.T * X1.T + X1 * H << 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        problem.solve(solver="mosek")

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
        self.problem.solve(solver="mosek")

        barrier = np.array2string(np.array(barrier), separator=", ")

        controller = self.U0 @ H @ P @ Matrix(x)
        controller = np.array2string(np.array(controller), separator=", ")

        P = np.array2string(np.array(P), separator=", ")
        H = np.array2string(np.array(H), separator=", ")
        Q = np.array2string(np.array(Q), separator=", ")

        return {
            "barrier": {
                "expression": barrier,
                "values": {"P": P},
            },
            "controller": {
                "expression": controller,
                "values": {"H": H},
            },
            "gamma": gamma_var.value,
            "lambda": lambda_var.value,
        }

    def _continuous_nps(self):
        """
        Solve for a continuous non-linear polynomial system.
        """

        # TODO: approximate X1 as the derivatives of the state at each sampling time, if not provided.

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()
        N0 = self.__compute_N0()

        H_x = matrix_variable(
            "H_x",
            list(self.x),
            self.degree,
            dim=(self.num_samples, self.N),
            hom=False,
            sym=False,
        )
        Z = matrix_variable(
            "Z", list(self.x), 0, dim=(self.N, self.N), hom=False, sym=True
        )

        HZ_problem = SOSProblem()

        self.__add_matrix_constraint(HZ_problem, N0 @ H_x - Z, list(self.x))

        dMdx = np.array([[m.diff(x) for x in self.x] for m in self.M_x])
        lie_derivative = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T

        if self.N == 1:
            lie_derivative = lie_derivative[0]
            HZ_problem.add_sos_constraint(-lie_derivative - Lg, list(self.x))
        else:
            Lg = sp.Mul(Lg, Matrix(I(self.N)))
            HZ_problem.add_matrix_sos_constraint(-lie_derivative - Lg, list(self.x))

        HZ_problem.solve()

        H_x, Z = self.__substitute_for_values(HZ_problem.variables.values(), H_x, Z)

        P = Z.inv()

        # --- (2) Then, solve remaining SOS conditions for gamma and lambda ---

        # TODO: assert Q_x == H_x @ P (to 10^-6)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = (Matrix(self.M_x).T @ P @ Matrix(self.M_x))[0]

        self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)
        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)

        try:
            self.problem.solve(solver="mosek")
        except SolutionFailure as e:
            # TODO: include info on what wasn't feasible
            return {"error": "Failed to solve the problem.", "description": str(e)}
        except Exception as e:
            return {"error": "An unknown error occurred.", "description": str(e)}

        # Q(x) = H(x) @ P
        # controller = U0 @ Hx @ P @ self.M_x

        # TODO: add SOS decomp (to all)
        # TODO: save out barrier and controller (to all)

        return {
            "barrier": {"expression": "M(x)<sup>T</sup>PM(x)", "values": {"P": P}},
            "controller": {
                "expression": "U<sub>0</sub>H(x)PM(x)",
                "values": {"H(x)": H_x},
            },
            "gamma": gamma_var.value,
            "lambda": lambda_var.value,
        }

    @staticmethod
    def __validate_solution(
        barrier_constraint, condition1, condition2
    ) -> Union[bool, dict]:
        """
        Validate the solution of the SOS problem.
        """

        try:
            barrier_decomp = barrier_constraint.get_sos_decomp()
            first_decomp = condition1.get_sos_decomp()
            second_decomps = [cond.get_sos_decomp() for cond in condition2]
        except Exception as e:
            return {"error": "No SOS decomposition found.", "description": str(e)}
        # third_decomp = condition3.get_sos_decomp()

        isAllPositiveSecondDecomps = all([len(decomp) > 0 for decomp in second_decomps])

        if (
            len(barrier_decomp) <= 0
            or len(first_decomp) <= 0
            or not isAllPositiveSecondDecomps
        ):
            return {"error": "Constraints are not sum-of-squares."}

        if barrier_decomp.free_symbols == 0:
            return {"error": "Barrier is scalar."}

        return True

    def __level_set_constraints(self):
        gamma, lambda_ = sp.symbols("gamma lambda")
        gamma_var = self.problem.sym_to_var(gamma)
        lambda_var = self.problem.sym_to_var(lambda_)

        self.problem.require(gamma_var > 0)
        self.problem.require(lambda_var > 0)
        self.problem.require(lambda_var > gamma_var)

        return gamma, lambda_, gamma_var, lambda_var

    def __compute_lagrangians(self):
        x = self.x

        degree = self.degree

        L_init = [poly_variable("Li" + str(i + 1), x, degree) for i in range(len(x))]

        # L_init = matrix_variable('l_init', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        g_init = self.generate_polynomial(self.initial_state)
        Lg_init = sum([L_init * g_init for L_init, g_init in zip(L_init, g_init)])

        Lg_unsafe_set = []
        for i in range(len(self.unsafe_states)):
            # L_unsafe = matrix_variable(f'l_unsafe_{i}', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
            L_unsafe = [poly_variable(f"Lu{i}{j}", x, degree) for j in range(len(x))]
            g_unsafe = self.generate_polynomial(self.unsafe_states[i])
            Lg_unsafe_set.append(sum([L * g for L, g in zip(L_unsafe, g_unsafe)]))

        # L = matrix_variable('l', list(x), degree, dim=(self.X0.shape[1], self.dimensionality), hom=False, sym=False)
        L = [poly_variable("L" + str(i + 1), x, degree) for i in range(len(x))]
        g = self.generate_polynomial(self.state_space)
        Lg = sum([L * g for L, g in zip(L, g)])

        return Lg_init, Lg_unsafe_set, Lg

    def __compute_N0(self) -> list:
        """
        Compute the N0 matrix by evaluating the monomials at each time step.
        """

        # Initialise the N0 matrix
        N0 = np.zeros((self.N, self.num_samples))

        for t in range(self.num_samples):
            # Get the x values at time t
            x_t = self.X0[:, t]

            for i in range(self.N):
                expr = sympify(self.monomials["terms"][i])
                N0[i, t] = float(expr.subs({k: val for k, val in zip(self.x, x_t)}))

        return N0

    def __solve(self):
        try:
            self.problem.solve()
        except SolutionFailure as e:
            return {"error": "Failed to solve the problem.", "description": str(e)}
        except Exception as e:
            return {"error": "An unknown error occurred.", "description": str(e)}

        return True

    @staticmethod
    def __add_matrix_constraint(
        problem: SOSProblem, mat: sp.Matrix, variables: List[sp.Symbol]
    ) -> List[Constraint]:
        """
        Add a matrix constraint to the problem.
        """

        variables = sorted(variables, key=str)  # To lex order

        constraints = []

        n, m = mat.shape
        # TODO: parallelize this loop
        for i in range(n):
            for j in range(m):
                expr = mat[i, j]

                poly = sp.poly(expr, variables)
                mono_to_coeffs = dict(
                    zip(poly.monoms(), map(problem.sp_to_picos, poly.coeffs()))
                )
                basis = Basis.from_poly_lex(poly, sparse=True)

                Q = RealVariable(f"Q_{i}_{j}", (len(basis), len(basis)))
                for mono, pairs in basis.sos_sym_entries.items():
                    coeff = mono_to_coeffs.get(mono, 0)
                    coeff_constraint = problem.add_constraint(
                        sum(Q[k, l] for k, l in pairs) == coeff
                    )
                    constraints.append(coeff_constraint)

                problem.add_constraint(Q == 0)

        return constraints

    @staticmethod
    def __substitute_for_values(variables, H_x, Z):
        # TODO: refactor for efficiency
        H_x_dict = {}
        Z_dict = {}
        for item in variables:
            if str(item.name).startswith("H_x"):
                H_x_dict[item.name] = item.value
            elif str(item.name).startswith("Z"):
                Z_dict[item.name] = item.value

        H_x = H_x.subs({key: value for key, value in H_x_dict.items()})
        Z = Z.subs({key: value for key, value in Z_dict.items()})

        # assert values are correct

        return H_x, Z

    @staticmethod
    def __matrix_to_string(matrix):
        """
        Convert a matrix to its comma-separated string notation.
        """
        return np.array2string(np.array(matrix), separator=", ")
