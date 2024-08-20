import numpy as np
import sympy as sp


class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.timing = data['timing']
        self.X0: sp.MutableDenseMatrix = sp.MutableDenseMatrix(data['X0']).T
        self.X1: sp.MutableDenseMatrix = sp.MutableDenseMatrix(data['X1']).T
        self.U0: sp.MutableDenseMatrix = sp.MutableDenseMatrix(data['U0'])
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']
        # TODO: ask user for custom degree
        self.degree = 2
        self.dimensions = len(self.state_space)

    def calculate(self):
        """Calculate the components of the Barrier Certificate"""
        raise NotImplementedError

    def generate_polynomial(self, space: list) -> list:
        """Generate the polynomial for the given space"""

        lower_bounds = [dimension[0] for dimension in space]
        upper_bounds = [dimension[1] for dimension in space]

        return [(var - lower) * (upper - var) for var, lower, upper in zip(self.x(), lower_bounds, upper_bounds)]

    def x(self):
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """

        dimensions = len(self.state_space)

        return sp.symbols(f'x1:{dimensions + 1}')
