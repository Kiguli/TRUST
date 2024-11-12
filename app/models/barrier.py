import array
import json
from typing import Optional, Union

import numpy as np
import sympy as sp
from sympy import sympify


class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self._data = data

        self.model = data['model']
        self.timing = data['timing']
        self.X0 = self.parse_dataset(data['X0'])
        self.X1 = self.parse_dataset(data['X1'])
        self.U0 = self.parse_dataset(data['U0'])

    def calculate(self):
        """Calculate the components of the Barrier Certificate"""
        raise NotImplementedError

    def generate_polynomial(self, space: list) -> list:
        """Generate the polynomial for the given space"""

        lower_bounds = []
        upper_bounds = []

        for dimension in space:
            if dimension[0] is None or dimension[1] is None:
                raise ValueError(f"{space} is not a valid state space. Please provide valid lower and upper bounds.")

            lower_bounds.append(float(dimension[0]))
            upper_bounds.append(float(dimension[1]))

        lower_bounds = [float(dimension[0]) for dimension in space]
        upper_bounds = [float(dimension[1]) for dimension in space]

        return [(var - lower) * (upper - var) for var, lower, upper in zip(self.x, lower_bounds, upper_bounds)]

    @property
    def state_space(self) -> Union[array, None]:
        return self._get_value('stateSpace')

    @property
    def initial_state(self) -> array:
        return self._get_value('initialState')

    @property
    def unsafe_states(self) -> array:
        return self._get_value('unsafeStates')

    @property
    def x(self) -> list[sp.Symbol]:
        """
        Return a range of symbols for the state space, from x1 to xN, where N is the number of dimensions
        """
        return sp.symbols(f'x1:{self.dimensionality + 1}')

    @property
    def degree(self):
        """Default the degree to the dimensionality"""
        # TODO: allow a custom degree?
        # For now, if dimensionality is even, use it, else degree - 1
        # return self.dimensionality if self.dimensionality % 2 == 0 else self.dimensionality - 1
        return 2

    @degree.setter
    def degree(self, value):
        self.degree = value

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
    def M_x(self) -> list:
        """
        Return the monomial terms.
        """
        monomials = self._get_value('monomials')
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

        if isinstance(data, str):
            data = json.loads(data)

        return np.array(data, dtype=float)

    def _get_value(self, key: str) -> Union[Optional[list], Optional[dict], None]:
        value = self._data.get(key)
        if value is None:
            return

        return json.loads(value)

