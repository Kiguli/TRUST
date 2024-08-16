class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        pass

    def calculate(self):
        """Calculate the components of the Barrier Certificate"""
        raise NotImplementedError

    def result(self, P, U, Q, gamma, _lambda):
        """Return the result of the Barrier Certificate calculation"""
        return {
            'barrier': {
                'expression': 'x^T @ P @ x',
                'values': {'P': P},
            },
            'controller': {
                'expression': 'U_{0,T} @ Q @ x',
                'values': {'U': U, 'Q': Q},
            },
            'gamma': gamma,
            'lambda': _lambda,
        }
