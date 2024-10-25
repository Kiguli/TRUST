from abc import ABC, abstractmethod
from numpy import array


class System(ABC):
    """
    Base class for all systems
    """

    def __init__(self):
        self.X0: array = None
        self.U0: array = None
        self.X1: array = None

    def trajectories(self, X0: array, U0: array, X1: array):
        self.X0 = X0
        self.U0 = U0
        self.X1 = X1

        return self

    def _validate_rank(self):
        """
        Validate the data is full rank.
        """
        pass


class Polynomial(System):
    @abstractmethod
    def monomials(self, M_x: str):
        pass


class Linear(System):
    pass


class ContinuousPolynomial(Polynomial):
    pass


class ContinuousLinear(Linear):
    pass


class DiscretePolynomial(Polynomial):
    pass


class DiscreteLinear(Linear):
    pass


class Stability:
    pass


class Barrier:
    pass


class SafetyBarrier:
    pass


def system(class_: str, model: str) -> System:
    """
    Factory function to create a system object.
    """
    classmap = {
        "ContinuousPolynomial": ContinuousPolynomial,
        "ContinuousLinear": ContinuousLinear,
        "DiscretePolynomial": DiscretePolynomial,
        "DiscreteLinear": DiscreteLinear,
    }

    return classmap[f"{class_}{model}"]()


if __name__ == "__main__":
    # -- User Input --
    X0 = [
        [
            2,
            1.76597758600002,
            1.57552318001073,
            1.41091550185672,
            1.26025065553552,
            1.12036451989824,
            0.985314239227562,
            0.853030699022831,
            0.724349070294856,
            0.599801824386643,
            0.479987290646065,
            0.363697446826679,
        ],
        [
            3,
            3.1371526399729,
            2.90611230139269,
            3.53581753966377,
            3.79970538645042,
            4.24739284586515,
            4.75763862865177,
            5.15250524184748,
            5.35359872248094,
            5.49071285873162,
            5.45492126365955,
            5.56032226506329,
        ],
    ]

    U0 = [
        -4.97867971784556,
        13.2194696065295,
        -29.9931375109593,
        -11.8600456420896,
        -21.1946465509732,
        -24.4596843138721,
        -18.8243873173397,
        -9.26635637741714,
        -6.1939515461598,
        2.32900404020142,
        -4.84832913580231,
        11.1131700238056,
    ]

    X1 = [
        [
            -13,
            -10.5689245849901,
            -8.58496179186933,
            -7.92618380665269,
            -7.18283798846243,
            -6.83336793592972,
            -6.69219813623366,
            -6.55435604709436,
            -6.33064739169323,
            -6.13824922204905,
            -5.85579457019575,
            -5.78279020564391,
        ],
        [
            6.97867971784556,
            -11.4534920205295,
            31.56866069097,
            13.2709611439463,
            22.4548972065087,
            25.5800488337704,
            19.8097015565673,
            10.11938707644,
            6.91830061645466,
            -1.72920221581478,
            5.32831642644838,
            -10.7494725769789,
        ],
    ]

    M_x = "x1*x2; x2; x1**2; x1**3"

    state_space = [[-1, 1], [-1, 1]]
    initial_state = [[0.1, 0.5], [0.1, 0.5]]
    unsafe_state = [[0.7, 1], [0.7, 1]]

    system_class = "Continuous"
    system_model = "Polynomial"
    system_type = "Safety"

    # -- Backend --

    # Build the system

    # Instantiate the system
    system = system(system_class, system_model)

    # Add the trajectories
    system = system.trajectories(X0=X0, U0=U0, X1=X1)

    # Add the spaces (only for barrier systems)
    if hasattr(system, "spaces"):
        system = system.spaces(
            state_space=state_space,
            initial_state=initial_state,
            unsafe_state=unsafe_state,
        )

    # Add monomials (only for polynomial systems)
    if hasattr(system, "monomials"):
        system = system.monomials(M_x=M_x)

    print(system)
