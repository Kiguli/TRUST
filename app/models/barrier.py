import numpy as np
import sympy as sp
from sympy import Matrix


# TODO: [Use Pydantic](https://thatgardnerone.atlassian.net/browse/PHD-125)
class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.timing = data['timing']
        self.X0 = np.transpose(np.array(data['X0']))
        self.X1 = np.transpose(np.array(data['X1']))
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
    def x(self):
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """

        dimensions = len(self.state_space)

        return sp.symbols(f'x1:{dimensions + 1}')

    @property
    def degree(self):
        return self.dimensions

    @property
    def dimensions(self):
        """
        Return the number of dimensions in the state space
        """
        return len(self.state_space)

    @property
    def num_samples(self):
        """
        Return the number of samples
        """
        return self.X0.shape[1]
