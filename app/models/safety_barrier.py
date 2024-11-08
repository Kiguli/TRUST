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
from soupsieve.util import deprecated
from sympy import Matrix, simplify, sympify

from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    is_stability = False

    def __init__(self, data: dict):
        # TODO: migrate to builder pattern?
        if data["mode"] == "Stability":
            self.is_stability = True

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
        # Rank condition:
        assert self.num_samples > self.dimensionality, "The number of samples, T, must be greater than the number of states, n."

        rank = np.linalg.matrix_rank(self.X0)
        assert rank == self.dimensionality, "The X0 data is not full row-rank."

        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        H = RealVariable("H", (self.num_samples, self.dimensionality))
        Z = SymmetricVariable("Z", (self.dimensionality, self.dimensionality))

        HZ_problem = SOSProblem()

        HZ_problem.add_constraint(Z == X0 * H)
        # Z must be positive definite
        HZ_problem.add_constraint(Z - 1.0e-6 * I(self.dimensionality) >> 0)

        schur = (Z & X1 * H) // (H.T * X1.T & Z)
        HZ_problem.add_constraint(schur >> 0)

        HZ_problem.solve(solver="mosek")

        H = Matrix(H)
        Z = Matrix(Z)
        P = Z.inv()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()
        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = (Matrix(self.x).T * P * Matrix(self.x))[0]

        # -- SOS constraints

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(
                self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)
            )

        self.__solve()

        validation = self.__validate_solution(
            condition1, condition2
        )
        if validation != True and "error" in validation:
            return validation

        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H @ P @ Matrix(self.x)
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H = self.__matrix_to_string(H)

        return {
            "function": {
                "expression": {"x<sup>T</sup>Px": barrier},
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
        N0 = self.__compute_N0()

        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        Theta_x = Matrix(self.Theta_x)

        # -- Part 2

        Hx_degree = max([sp.poly(term).total_degree() for term in self.M_x])

        H_x = matrix_variable("H_x", self.x, Hx_degree, dim=(self.num_samples, self.dimensionality))
        Z = matrix_variable("Z", self.x, 0, dim=(self.dimensionality, self.dimensionality), sym=True)

        design_HZ = SOSProblem()

        # 21a. N0 @ H(x) = Theta(x) @ Z and Z is positive definite
        self.__add_matrix_constraint(design_HZ, (N0 @ H_x) - (Theta_x @ Z), self.x)
        self.__add_positive_matrix_constraint(design_HZ, Z - 1.0e-6 * np.eye(self.dimensionality), self.x)

        # 21d. Schur's complement
        schur = Matrix([[Z, self.X1 @ H_x], [H_x.T @ self.X1.T, Z]])

        if self.N == 1:
            schur = schur[0]
            design_HZ.add_matrix_sos_constraint(schur - Lg, list(self.x))
        else:
            Lg = sp.Mul(Lg, Matrix(I(2 * self.dimensionality)))
            design_HZ.add_matrix_sos_constraint(schur - Lg, list(self.x))

        design_HZ.solve(solver="mosek")

        H_x, Z = self.__substitute_for_values(design_HZ.variables.values(), H_x, Z)
        P = Z.inv()

        # -- Part 3

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = (Matrix(self.x).T @ P @ Matrix(self.x))[0]

        # -- SOS constraints

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(
                self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)
            )

        schur = Matrix([[Z, self.X1 @ H_x], [H_x.T @ self.X1.T, Z]])
        condition3 = self.problem.add_matrix_sos_constraint(schur - Lg, list(self.x))

        self.__solve()

        validation = self.__validate_solution(
            condition1, condition2, condition3
        )
        if validation != True and "error" in validation:
            return validation

        barrier = (Matrix(self.x).T * P * Matrix(self.x))[0]
        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H_x @ P @ Matrix(self.x)
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "function": {
                "expression": {"x<sup>T</sup>Px": barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>H(x)Px": controller},
                "values": {"H(x)": H_x},
            },
            "gamma": str(gamma_var.value),
            "lambda": str(lambda_var.value),
        }

    def _continuous_linear(self):
        # Rank condition:
        assert self.num_samples > self.dimensionality, "The number of samples, T, must be greater than the number of states, n."

        rank = np.linalg.matrix_rank(self.X0)
        assert rank == self.dimensionality, "The X0 data is not full row-rank."

        problem = SOSProblem()

        # -- Solve for H and Z

        x = self.x
        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        H = RealVariable("H", (self.num_samples, self.dimensionality))
        Z = SymmetricVariable("Z", (self.dimensionality, self.dimensionality))

        problem.add_constraint(H.T * X1.T + X1 * H << 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        problem.solve(solver="mosek")

        H = np.array(H)
        Z = np.array(Z)
        P = np.linalg.inv(Z)

        barrier = np.array(x).T @ P @ np.array(x)

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()
        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()

        # -- SOS constraints

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, x))

        # -- Solve
        self.__solve()

        validation = self.__validate_solution(
            condition1, condition2
        )
        if validation != True and "error" in validation:
            return validation

        barrier = np.array2string(np.array(barrier), separator=", ")

        controller = self.U0 @ H @ P @ Matrix(x)
        controller = np.array2string(np.array(controller), separator=", ")

        P = np.array2string(np.array(P), separator=", ")
        H = np.array2string(np.array(H), separator=", ")

        return {
            "function": {
                "expression": {"x<sup>T</sup>Px": barrier},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>HPx": controller},
                "values": {"H": H},
            },
            "gamma": gamma_var.value,
            "lambda": lambda_var.value,
        }

    def _continuous_nps(self):
        """
        Solve for a continuous non-linear polynomial system.
        """

        # TODO: Future work: approximate X1 as the derivatives of the state at each sampling time, if not provided.

        # # TODO: Get the highest degree term in M_x
        # highest_degree = max([term.total_degree() for term in self.M_x])
        # self.degree = highest_degree * 2
        # Test for Jet Engine setting degree to 9 (3^2)
        # Else, we can try highest degree (or highest degree - 1)


        Lg_init, Lg_unsafe_set, Lg = self.__compute_lagrangians()
        N0 = self.__compute_N0()

        Hx_degree = max([sp.poly(term).total_degree() for term in self.M_x])

        H_x = matrix_variable("H_x", list(self.x), Hx_degree, dim=(self.num_samples, self.N))

        Z = matrix_variable(
            "Z", list(self.x), 0, dim=(self.N, self.N), hom=False, sym=True
        )

        HZ_problem = SOSProblem()

        self.__add_matrix_constraint(HZ_problem, N0 @ H_x - Z, list(self.x))
        self.__add_positive_matrix_constraint(HZ_problem, Z - 1.0e-6 * np.eye(self.N), list(self.x))

        dMdx = Matrix(self.M_x).jacobian(self.x)

        lie_derivative = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T

        if self.N == 1:
            lie_derivative = lie_derivative[0]
            HZ_problem.add_sos_constraint(-lie_derivative - Lg, list(self.x))
        else:
            Lg = sp.Mul(Lg, Matrix(I(self.N)))
            HZ_problem.add_matrix_sos_constraint(-lie_derivative - Lg, list(self.x))

        HZ_problem.solve(solver="mosek")

        H_x, Z = self.__substitute_for_values(HZ_problem.variables.values(), H_x, Z)

        P = Z.inv()

        # --- (2) Then, solve remaining SOS conditions for gamma and lambda ---

        gamma, lambda_, gamma_var, lambda_var = self.__level_set_constraints()

        barrier = (Matrix(self.M_x).T @ P @ Matrix(self.M_x))[0]

        condition1 = self.problem.add_sos_constraint(-barrier - Lg_init + gamma, self.x)

        condition2 = []
        for Lg_unsafe in Lg_unsafe_set:
            condition2.append(
                self.problem.add_sos_constraint(barrier - Lg_unsafe - lambda_, self.x)
            )

        self.__solve()

        validation = self.__validate_solution(
            condition1, condition2
        )
        if validation != True and "error" in validation:
            return validation

        barrier = self.__matrix_to_string(barrier)

        controller = self.U0 @ H_x @ P @ Matrix(self.M_x)
        controller = self.__matrix_to_string(controller)

        # sp.pretty_print(P, use_unicode=True, mat_symbol_style="bold")

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "function": {
                "expression": {"M(x)<sup>T</sup>PM(x)": barrier},
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
        condition1, condition2, condition3 = None
    ) -> Union[bool, dict]:
        """
        Validate the solution of the SOS problem.
        """

        try:
            first_decomp = condition1.get_sos_decomp()
            second_decomps = [cond.get_sos_decomp() for cond in condition2]
            if condition3 is not None:
                third_decomp = condition3.get_sos_decomp()
        except Exception as e:
            return {"error": "No SOS decomposition found.", "description": str(e)}

        isAllPositiveSecondDecomps = all([len(decomp) > 0 for decomp in second_decomps])

        if (
            len(first_decomp) <= 0
            or not isAllPositiveSecondDecomps
            or (condition3 is not None and len(third_decomp) <= 0)
        ):
            return {"error": "Constraints are not sum-of-squares."}

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

        # Rank conditions
        assert self.num_samples > self.N, "The number of samples, T, must be greater than the number of monomial terms, N."
        rank = np.linalg.matrix_rank(N0)
        assert rank == self.N, "The N0 data is not full row-rank."

        return N0

    def __solve(self):
        try:
            self.problem.solve(solver="mosek")
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

        return constraints

    @staticmethod
    def __add_positive_matrix_constraint(
        problem: SOSProblem, mat: sp.Matrix, variables: List[sp.Symbol]
    ) -> List[Constraint]:
        """
        Add a matrix constraint to the problem.
        """

        variables = sorted(variables, key=str)  # To lex order

        constraints = []

        # TODO: parallelize this loop
        n, m = mat.shape
        for i in range(n):
            for j in range(m):
                expr = mat[i, j]

                poly = sp.poly(expr, variables)
                mono_to_coeffs = dict(
                    zip(poly.monoms(), map(problem.sp_to_picos, poly.coeffs()))
                )
                basis = Basis.from_poly_lex(poly, sparse=True)

                R = RealVariable(f"R_{i}_{j}", (len(basis), len(basis)))
                for mono, pairs in basis.sos_sym_entries.items():
                    coeff = mono_to_coeffs.get(mono, 0)
                    coeff_constraint = problem.add_constraint(
                        sum(R[k, l] for k, l in pairs) == coeff
                    )
                    constraints.append(coeff_constraint)

                problem.add_constraint(R >> 0)

        return constraints

    # @staticmethod
    # def __add_positive_matrix_constraint(
    #         problem: SOSProblem, mat: sp.Matrix, variables: List[sp.Symbol]
    # ) -> List[Constraint]:
    #     """
    #     Add a matrix constraint to the problem.
    #     """
    #
    #     variables = sorted(variables, key=str)  # To lex order
    #
    #     constraints = []
    #
    #     # TODO: parallelize this loop
    #     n, m = mat.shape
    #     for i in range(n):
    #         for j in range(m):
    #             expr = mat[i, j]
    #
    #             poly = sp.poly(expr, variables)
    #             mono_to_coeffs = dict(
    #                 zip(poly.monoms(), map(problem.sp_to_picos, poly.coeffs()))
    #             )
    #             basis = Basis.from_poly_lex(poly, sparse=True)
    #
    #             R = SymmetricVariable(f"R_{i}_{j}", len(basis))
    #             for mono, pairs in basis.sos_sym_entries.items():
    #                 coeff = mono_to_coeffs.get(mono, 0)
    #                 coeff_constraint = problem.add_constraint(
    #                     sum(R[k, l] for k, l in pairs) == coeff
    #                 )
    #                 constraints.append(coeff_constraint)
    #
    #             problem.add_constraint(R >> 0)
    #
    #     return constraints

    # @staticmethod
    # def __add_matrix_sos_constraint(
    #         problem: SOSProblem, mat: sp.Matrix, variables: List[sp.Symbol]
    # ) -> List[SOSConstraint]:
    #     """
    #     Add a matrix SOS constraint to the problem.
    #     """
    #
    #     n, m = mat.shape
    #     assert n == m, 'Matrix must be square!'
    #
    #     name = hash(mat)
    #     aux_var_name = f'_y{name}'
    #     aux_vars = list(sp.symbols(f'{aux_var_name}_:{n}'))
    #
    #     # p is sos iff all matrix elements are sos?
    #
    #     variables = sorted(variables, key=str) + aux_vars
    #     sos_constraints = []
    #
    #     # TODO: parallelize this loop
    #     for i in range(n):
    #         for j in range(m):
    #             expr = mat[i, j]
    #
    #             poly = sp.poly(expr, variables)
    #
    #             deg = poly.total_degree() # Fails for some matrix polys
    #             assert deg % 2 == 0, 'Polynomial degree must be even!'
    #
    #             mono_to_coeffs = dict(
    #                 zip(poly.monoms(), map(problem.sp_to_picos, poly.coeffs()))
    #             )
    #             basis = Basis.from_poly_lex(poly, sparse=True)
    #
    #             Q = SymmetricVariable(f"Q_sos_{i}_{j}", len(basis))
    #             for mono, pairs in basis.sos_sym_entries.items():
    #                 coeff = mono_to_coeffs.get(mono, 0)
    #                 problem.add_constraint(sum(Q[k, l] for k, l in pairs) == coeff)
    #
    #             constraint = problem.add_constraint (Q >> 0)
    #
    #             sos_constraints.append(SOSConstraint(constraint, Q, basis, variables, deg))
    #
    #     return sos_constraints

    @staticmethod
    def __substitute_for_values(variables, H_x: Matrix, Z: Matrix) -> tuple:
        # TODO: refactor for efficiency?
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
