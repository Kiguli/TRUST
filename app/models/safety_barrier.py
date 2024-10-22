import io
import time
from contextlib import redirect_stdout
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
                "expression": {'x<sup>T</sup>Px': barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>HPx": controller},
                "values": {"H": H},
            },
            "gamma": str(gamma_var.value),
            "lambda": str(lambda_var.value),
        }

    def _discrete_nps(self):
        Theta_x = matrix_variable(
            "Theta_x", self.x, self.degree, dim=(self.N, self.dimensionality)
        )
        Q_x = matrix_variable(
            "Q_x", self.x, self.degree, dim=(self.num_samples, self.dimensionality)
        )
        N0 = self.__compute_N0()

        # -- Part 1

        design_theta = SOSProblem()

        # 17. Theta(x) = N0 @ Q(x)
        self.__add_matrix_constraint(design_theta, Theta_x - N0 @ Q_x, self.x)

        # 18. M(x) = Theta(x) @ x
        self.__add_matrix_constraint(
            design_theta, self.M_x - Theta_x @ Matrix(self.x), self.x
        )

        design_theta.solve(solver="mosek")

        # TODO: sub real values

        # -- Part 2

        H_x = matrix_variable(
            "H_x", self.x, self.degree, dim=(self.num_samples, self.N)
        )
        Z = matrix_variable("Z", self.x, 0, dim=(self.N, self.N), sym=True)

        design_HZ = SOSProblem()

        # 21a. N0 @ H(x) = Theta(x) @ Z and Z is positive definite
        self.__add_matrix_constraint(design_HZ, N0 @ H_x - Theta_x @ Z, self.x)
        design_HZ.add_constraint(Z - 1.0e-6 * I(self.N) >> 0)

        # 21d. Schur's complement
        schur = Matrix([[Z, self.X1 @ H_x], [H_x.T @ self.X1.T, Z]])
        self.__add_matrix_inequality_constraint(design_HZ, schur, self.x)

        design_HZ.solve(solver="mosek")

        H_x, Z = self.__substitute_for_values(design_HZ.variables.values(), H_x, Z)
        P = Z.inv()

        # -- Part 3

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()
        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        # 9a. SOS gamma
        self.problem.add_sos_constraint(
            -Matrix(self.x).T @ P @ Matrix(self.x) - Lg_init + gamma, self.x
        )

        # 9b. SOS lambda
        for Lg_unsafe in Lg_unsafe_set:
            self.problem.add_sos_constraint(
                Matrix(self.x).T @ P @ Matrix(self.x) - Lg_unsafe - lambda_, self.x
            )

        # Note: Redefine schur with the now-valued matrices
        schur = Matrix([[Z, self.X1 @ H_x], [H_x.T @ self.X1.T, Z]])

        # 9c. SOS state space
        self.problem.add_matrix_sos_constraint(schur - Lg, self.x)

        self.__solve()

        # TODO: validate

        barrier = (Matrix(self.x).T * P * Matrix(self.x))[0]
        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H_x @ P @ Matrix(self.x)
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "barrier": {
                "expression": {'x<sup>T</sup>Px': barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {'U<sub>0</sub>H(x)Px': controller},
                "values": {"H(x)": H_x},
            },
            "gamma": str(gamma_var.value),
            "lambda": str(lambda_var.value),
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
                "expression": {'x<sup>T</sup>Px': barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {'U<sub>0</sub>HPx': controller},
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

        # assert np.allclose(N0 @ H_x @ P, np.eye(self.N), atol=1e-6)

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

        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H_x @ P @ Matrix(self.M_x)
        controller = self.__matrix_to_string(controller)

        # sp.pretty_print(P, use_unicode=True, mat_symbol_style="bold")

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "barrier": {
                "expression": {'M(x)<sup>T</sup>PM(x)': barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>H(x)PM(x)": controller},
                "values": {"H(x)": H_x},
            },
            "gamma": str(gamma_var.value),
            "lambda": str(lambda_var.value),
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

        # TODO: parallelize this loop
        start_time = time.time()

        n, m = mat.shape
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

        print(f"Time taken: {time.time() - start_time}")

        return constraints

    @staticmethod
    def __add_matrix_inequality_constraint(
        problem: SOSProblem, mat: sp.Matrix, variables: List[sp.Symbol]
    ) -> List[Constraint]:
        """
        Add a matrix constraint to the problem.

        TODO: refactor to combine with __add_matrix_constraint
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

                problem.add_constraint(Q >= 0)

        return constraints

    @staticmethod
    def __substitute_for_values(variables, H_x: Matrix, Z: Matrix) -> tuple:
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
