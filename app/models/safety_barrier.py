from app.models.barrier import Barrier


class SafetyBarrier(Barrier):
    """Safety Barrier Certificate"""

    def __init__(self, data: dict):
        if data['mode'] != 'Safety':
            raise ValueError(f"Invalid mode '{data['mode']}' for Safety Barrier calculations.")

        super().__init__(data)

    def calculate(self):
        P = None
        U = None
        Q = None
        gamma = 1
        _lambda = 2

        return super().result(P, U, Q, gamma, _lambda)
