import array
import json
from typing import List, Optional, Self, Union

import numpy as np
import sympy as sp
from SumOfSquares import (
    Basis,
    SOSProblem,
    matrix_variable,
    poly_variable,
)
from picos import Constant, I, RealVariable, SolutionFailure, SymmetricVariable
from picos.constraints import Constraint
from sympy import Matrix, sympify
from werkzeug.utils import cached_property


class Stability:
    def __init__(self, data: dict):
        self._data: dict = data

        if self._data.get("mode") != "Stability":
            raise ValueError(
                f"Invalid mode '{self._data.get("mode")}' for Stability calculations."
            )

        self._model = self._data.get("model")
        self._timing = self._data.get("timing")
        self._X0 = self.parse_dataset(self._data.get("X0"))
        self._X1 = self.parse_dataset(self._data.get("X1"))
        self._U0 = self.parse_dataset(self._data.get("U0"))

    def calculate(self) -> dict:
        results = {}

        if self.model == "Linear":
            results = self._solve_linear()
        elif self.model == "Non-Linear Polynomial":
            results = self._solve_polynomial()

        return results

    def _solve_linear(self) -> dict:
        assert (
            self.num_samples > self.dimensionality
        ), "The number of samples, T, must be greater than the number of states, n."

        rank = np.linalg.matrix_rank(self.X0)
        assert rank == self.dimensionality, "The X0 data is not full row-rank."

        H, Z = None, None
        if self.timing == "Discrete-Time":
            H, Z = self._discrete_constraints()
        elif self.timing == "Continuous-Time":
            H, Z = self._continuous_constraints()

        P = Matrix(Z).inv() if self.dimensionality > 1 else 1 / Z.value

        lyapunov = Matrix(self.x).T @ P @ Matrix(self.x)
        lyapunov = np.array2string(np.array(lyapunov), separator=", ")

        controller = self.U0 @ H @ P @ Matrix(self.x)
        controller = np.array2string(np.array(controller), separator=", ")

        H = np.array2string(np.array(Matrix(H)))
        P = np.array2string(np.array(P))

        return {
            "function": {
                "expression": {"x<sup>T</sup>Px": lyapunov},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>HPx": controller},
                "values": {"H": H},
            },
        }

    def _solve_polynomial(self) -> dict:
        if self.timing == "Discrete-Time":
            return self.__discrete_polynomial()
        elif self.timing == "Continuous-Time":
            return self.__continuous_polynomial()

    def __continuous_polynomial(self) -> dict:
        """
        Calculate the Lyapunov function and controller for a continuous-time non-linear polynomial system.

        Solve for P and H_x
        (1) N0 @ H_x = P_inv
        (2) dMdx @ X1 @ H_x + H_x.T @ X1.T @ dMdX.T << 0
        """

        # Rank condition
        N0 = self.__compute_N0()
        assert (
            self.num_samples > self.N
        ), "The number of samples, T, must be greater than the number of monomial terms, N."

        rank = np.linalg.matrix_rank(N0)
        assert rank == self.N, "The data must be full row rank."

        Hx_degree = max([sp.poly(term).total_degree() for term in self.M_x])
        H_x = matrix_variable(
            "H_x",
            list(self.x),
            Hx_degree,
            dim=(self.num_samples, self.N),
            hom=False,
            sym=False,
        )
        Z = matrix_variable(
            "Z", list(self.x), 0, dim=(self.N, self.N), hom=False, sym=True
        )

        HZ_problem = SOSProblem()

        self.__add_matrix_constraint(HZ_problem, N0 @ H_x - Z, list(self.x))
        self.__add_positive_matrix_constraint(
            HZ_problem, Z - 1.0e-6 * np.eye(self.N), list(self.x)
        )

        dMdx = Matrix(self.M_x).jacobian(self.x)

        lie_derivative = dMdx @ self.X1 @ H_x + H_x.T @ self.X1.T @ dMdx.T

        # Add a virtually infinite constraint
        L = [
            poly_variable("L" + str(i + 1), self.x, self.degree)
            for i in range(len(self.x))
        ]
        g = self.generate_polynomial(
            [[-1.0e-308, 1.0e-308] for _ in range(self.dimensionality)]
        )
        Lg = sum([L * g for L, g in zip(L, g)])

        if self.N == 1:
            lie_derivative = lie_derivative[0]
            HZ_problem.add_sos_constraint(-lie_derivative - Lg, list(self.x))
        else:
            Lg = sp.Mul(Lg, Matrix(I(self.N)))
            HZ_problem.add_matrix_sos_constraint(-lie_derivative - Lg, list(self.x))

        HZ_problem.solve(solver="mosek")

        H_x, Z = self.__substitute_for_values(HZ_problem.variables.values(), H_x, Z)

        P = Z.inv()

        lyapunov = (Matrix(self.M_x).T @ P @ Matrix(self.M_x))[0]
        lyapunov = self.__matrix_to_string(lyapunov)

        controller = self.U0 @ H_x @ P @ Matrix(self.M_x)
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "function": {
                "expression": {"M(x)^T @ P @ M(x)": lyapunov},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U0 @ H(x) @ P @ M(x)": controller},
                "values": {"H(x)": H_x},
            },
        }

    def __discrete_polynomial(self) -> dict:
        # Rank condition:
        # N0 is an (n, T) full row-rank matrix.
        N0 = self.__compute_N0()

        assert (
            self.num_samples > self.N
        ), "The number of samples, T, must be greater than the number of monomial terms, N."

        rank = np.linalg.matrix_rank(N0)
        assert rank == self.N, "The data must be full row rank."

        L = [
            poly_variable("L" + str(i + 1), self.x, self.degree)
            for i in range(len(self.x))
        ]
        g = self.generate_polynomial(
            [[-1.0e-6, 1.0e-6] for _ in range(self.dimensionality)]
        )
        Lg = sum([L * g for L, g in zip(L, g)])

        Theta_x = Matrix(self.Theta_x)

        # -- Part 2

        Hx_degree = max([sp.poly(term).total_degree() for term in self.M_x])
        H_x = matrix_variable("H_x", self.x, Hx_degree, dim=(self.num_samples, self.dimensionality))
        Z = matrix_variable("Z", self.x, 0, dim=(self.dimensionality, self.dimensionality), sym=True)

        design_HZ = SOSProblem()

        # 21a. N0 @ H(x) = Theta(x) @ Z and Z is positive definite
        self.__add_matrix_constraint(design_HZ, (N0 @ H_x) - (Theta_x @ Z), self.x)
        self.__add_positive_matrix_constraint(
            design_HZ, Z - 1.0e-6 * np.eye(self.dimensionality), self.x
        )
        # design_HZ.add_constraint(Z - 1.0e-6 * I(self.dimensionality) >> 0)
        # design_HZ.add_constraint(Z - 1.0e-6 * np.eye(self.dimensionality) >> 0)

        # 21d. Schur's complement
        schur = Matrix([[Z, self.X1 @ H_x], [H_x.T @ self.X1.T, Z]])

        if self.N == 1:
            schur = schur[0]
            schur_constraint = design_HZ.add_matrix_sos_constraint(
                schur - Lg - 1.0e-6 * np.eye(2 * self.dimensionality), list(self.x)
            )
        else:
            Lg = sp.Mul(Lg, Matrix(I(2 * self.dimensionality)))
            schur_constraint = design_HZ.add_matrix_sos_constraint(
                schur - Lg - 1.0e-6 * np.eye(2 * self.dimensionality), list(self.x)
            )

        # 9c. SOS state space

        # Note: Redefine schur with the now-valued matrices

        try:
            design_HZ.solve(solver="mosek")
        except SolutionFailure as e:
            # TODO: include info on what wasn't feasible
            return {"error": "Failed to solve the problem.", "description": str(e)}
        except Exception as e:
            return {"error": "An unknown error occurred.", "description": str(e)}

        H_x, Z = self.__substitute_for_values(design_HZ.variables.values(), H_x, Z)
        P = Z.inv()

        # -- Part 3

        lyapunov = (Matrix(self.x).T @ P @ Matrix(self.x))[0]
        lyapunov = self.__matrix_to_string(lyapunov)

        validation = self.__validate_solution(schur_constraint)
        if validation != True and "error" in validation:
            return validation

        # U0 @ H(x) (left_pseudoinverse(Theta(x)) @ N0 @ H(x))
        # controller = self.U0 @ H_x @ P @ Matrix(self.x)
        controller = self.U0 @ H_x @ P @ Matrix(list(self.x))
        controller = self.__matrix_to_string(controller)

        P = self.__matrix_to_string(P)
        H_x = self.__matrix_to_string(H_x)

        return {
            "function": {
                "expression": {"x<sup>T</sup>Px": lyapunov},
                "values": {"P": P},
            },
            "controller": {
                "expression": {"U<sub>0</sub>H(x)Px": controller},
                "values": {"H(x)": H_x},
            },
        }

    def _discrete_constraints(self) -> array:
        # block_matrix = cp.bmat([[Z, X1 @ H], [H.T @ X1.T, Z]])
        #
        # return [Z >> 0, Z == X0 @ H, block_matrix >> 0]
        X0 = self.X0
        X1 = self.X1

        n = self.X0.shape[0]
        T = self.X0.shape[1]

        problem = SOSProblem()

        X0 = Constant("X0", X0)
        X1 = Constant("X1", X1)

        H = RealVariable("H", (T, n))
        Z = SymmetricVariable("Z", (n, n))

        problem.add_constraint(Z == X0 * H)
        # Z must be positive definite
        problem.add_constraint(Z - 1.0e-6 * I(n) >> 0)

        schur = (Z & H.T * X1.T) // (X1 * H & Z)
        # schur = ((Z & X1 * H) // (H.T * X1.T & Z))
        problem.add_constraint(schur >> 0)

        problem.solve(solver="mosek")

        return Matrix(H), Matrix(Z)

    def _continuous_constraints(self) -> array:
        # eqn = X1 @ H + H.T @ X1.T
        #
        # return [Z >> 0, Z == X0 @ H, eqn << 0]
        n = self.X0.shape[0]
        T = self.X0.shape[1]

        problem = SOSProblem()

        # -- Solve for H and Z

        x = self.x
        X0 = Constant("X0", self.X0)
        X1 = Constant("X1", self.X1)

        H = RealVariable("H", (T, n))
        Z = SymmetricVariable("Z", (n, n))

        problem.add_constraint(H.T * X1.T + X1 * H << 0)

        problem.add_constraint(Z - 1.0e-6 * I(Matrix(x).shape[1]) >> 0)
        problem.add_constraint(Z == X0 * H)

        problem.solve(solver="mosek")

        return Matrix(H), Matrix(Z)

    # --- Properties ---

    @cached_property
    def degree(self):
        if self.timing == 'Continuous-Time' and self.model == 'Non-Linear Polynomial':
            return max([sp.poly(term).total_degree() for term in self.M_x]) * 2

        return 2

    @cached_property
    def dimensionality(self):
        """
        Return the dimensionality in the state space, n
        """
        return len(self.X0)

    @cached_property
    def N(self) -> int:
        """
        Return the number of monomial terms, N
        """
        return len(self.M_x)

    @cached_property
    def num_samples(self) -> int:
        """
        Return the number of samples, T
        """
        return self.X0.shape[1]

    @cached_property
    def state_space(self) -> Union[array, None]:
        return self._get_value('stateSpace')

    @cached_property
    def initial_state(self) -> array:
        return self._get_value('initialState')

    @cached_property
    def unsafe_states(self) -> array:
        return self._get_value('unsafeStates')

    @cached_property
    def x(self) -> list[sp.Symbol]:
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """
        return sp.symbols(f"x1:{self.dimensionality + 1}")

    @cached_property
    def model(self) -> str:
        return self._model

    @cached_property
    def timing(self) -> str:
        return self._timing

    @cached_property
    def X0(self) -> np.array:
        return self._X0

    @cached_property
    def X1(self) -> np.array:
        return self._X1

    @cached_property
    def U0(self) -> np.array:
        return self._U0

    # --- Helper Methods ---

    def _get_value(self, key: str) -> Union[Optional[list], Optional[dict], None]:
        value = self._data.get(key)
        if value is None:
            return

        return json.loads(value)

    @staticmethod
    def parse_dataset(data: list) -> np.array:
        """
        Get the initial state of the system as a numpy array of floats
        """

        if isinstance(data, str):
            data = json.loads(data)

        return np.array(data, dtype=float)

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
                expr = self.M_x[i]
                N0[i, t] = float(expr.subs({k: val for k, val in zip(self.x, x_t)}))

        # Rank conditions
        assert self.num_samples > self.N, "The number of samples, T, must be greater than the number of monomial terms, N."
        rank = np.linalg.matrix_rank(N0)
        assert rank == self.N, "The N0 data is not full row-rank."

        return N0

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

    @property
    def M_x(self) -> Union[list, None]:
        """
        Return the monomial terms.
        """
        monomials = self._get_value('monomials')
        if monomials is None:
            return None

        return [sympify(term) for term in monomials['terms']]

    @property
    def Theta_x(self) -> list:
        """
        Return the theta terms.
        """

        theta_x = self._get_value('theta_x')

        if np.array(theta_x).shape != (self.N, self.dimensionality):
            raise ValueError(f"Theta_x should be of shape ({self.N}, {self.dimensionality}), not {np.array(theta_x).shape}")

        return [[sympify(term) for term in row] for row in theta_x]

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

    def generate_polynomial(self, space) -> list:
        """Generate the polynomial for the given space"""

        lower_bounds = []
        upper_bounds = []

        for dimension in space:
            if dimension[0] is None or dimension[1] is None:
                raise ValueError(
                    f"{space} is not a valid state space. Please provide valid lower and upper bounds."
                )

            lower_bounds.append(float(dimension[0]))
            upper_bounds.append(float(dimension[1]))

        lower_bounds = [float(dimension[0]) for dimension in space]
        upper_bounds = [float(dimension[1]) for dimension in space]

        return [
            (var - lower) * (upper - var)
            for var, lower, upper in zip(self.x, lower_bounds, upper_bounds)
        ]

    @staticmethod
    def __matrix_to_string(matrix):
        """
        Convert a matrix to its comma-separated string notation.
        """
        return np.array2string(np.array(matrix), separator=", ")

    @staticmethod
    def __validate_solution(schur_constraint) -> Union[bool, dict]:
        """
        Validate the solution of the SOS problem.
        """

        try:
            schur_decomp = schur_constraint.get_sos_decomp()
        except Exception as e:
            return {"error": "No SOS decomposition found.", "description": str(e)}
        # third_decomp = condition3.get_sos_decomp()

        if schur_decomp.free_symbols == 0 or len(schur_decomp) <= 0:
            return {"error": "Constraints are not sum-of-squares."}

        return True
