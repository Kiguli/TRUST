class Barrier:
    """Barrier Interface"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.timing = data['timing']
        self.X0 = data['X0']
        self.X1 = data['X1']
        self.U0 = data['U0']
        self.state_space = data['stateSpace']
        self.initial_state = data['initialState']
        self.unsafe_states = data['unsafeStates']
        # TODO: ask user for custom degree
        self.degree = len(self.state_space)

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
