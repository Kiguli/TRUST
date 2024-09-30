import numpy as np
import sympy as sp
from sympy import Matrix


# TODO: [Use Pydantic](https://thatgardnerone.atlassian.net/browse/PHD-125)
class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.timing = data['timing']
        self.monomials = data.get('monomials', [])
        self.X0 = self.__X0(data['X0'])
        self.X1 = np.array(data['X1'])
        self.U0 = np.array(data['U0'])
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']

    def calculate(self):
        """Calculate the components of the Barrier Certificate"""
        raise NotImplementedError

    def generate_polynomial(self, space: list) -> Matrix:
        """Generate the polynomial for the given space"""

        lower_bounds = [dimension[0] for dimension in space]
        upper_bounds = [dimension[1] for dimension in space]

        return Matrix([(var - lower) * (upper - var) for var, lower, upper in zip(self.x, lower_bounds, upper_bounds)])

    @property
    def x(self) -> list[sp.Symbol]:
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """

        dimensions = len(self.state_space)

        return sp.symbols(f'x1:{dimensions + 1}')

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
    def N(self):
        """
        Return the number of monomial terms, N
        """
        return len(self.monomials)

    @staticmethod
    def __X0(dataX0: list) -> np.array:
        """
        Get the initial state of the system as a numpy array of floats
        """

        for i in range(len(dataX0)):
            for j in range(len(dataX0[i])):
                dataX0[i][j] = float(dataX0[i][j])

        return np.array(dataX0)
