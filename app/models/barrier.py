import numpy as np
import sympy as sp
from sympy import Matrix


class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.timing = data['timing']
        self.monomials: dict = data.get('monomials', {})
        self.X0 = self.parse_dataset(data['X0'])
        self.X1 = self.parse_dataset(data['X1'])
        self.U0 = self.parse_dataset(data['U0'])
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']

    def calculate(self):
        """Calculate the components of the Barrier Certificate"""
        raise NotImplementedError

    def generate_polynomial(self, space: list) -> Matrix:
        """Generate the polynomial for the given space"""

        lower_bounds = [float(dimension[0]) for dimension in space]
        upper_bounds = [float(dimension[1]) for dimension in space]

        return Matrix([(var - lower) * (upper - var) for var, lower, upper in zip(self.x, lower_bounds, upper_bounds)])

    @property
    def x(self) -> list[sp.Symbol]:
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """
        return sp.symbols(f'x1:{self.dimensionality + 1}')

    @property
    def degree(self):
        """Default the degree to the dimensionality"""
        # TODO: allow a custom degree
        return self.dimensionality

    @property
    def dimensionality(self):
        """
        Return the dimensionality in the state space, n
        """
        return len(self.state_space)

    @property
    def num_samples(self):
        """
        Return the number of samples, T
        """
        return self.X0.shape[1]

    @property
    def M_x(self):
        """
        Return the monomial terms.
        """
        return self.monomials['terms']

    @property
    def N(self):
        """
        Return the number of monomial terms, N
        """
        return len(self.M_x)

    @staticmethod
    def parse_dataset(data: list) -> np.array:
        """
        Get the initial state of the system as a numpy array of floats
        """

        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = float(data[i][j])

        return np.array(data)
