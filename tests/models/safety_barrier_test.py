from app.models.safety_barrier import SafetyBarrier
from tests import sample_data


def test_it_returns_correctly_for_discrete_time_safety_barrier_certificates(sample_data):
    """
    The Lyapunov function to return is then
    $\\mathcal{V}(x) = x^\\top Px$ and the controller to return is $u=\\mathcal{U}_{0,T}HP^{-1}x$.
    """
    sample_data['mode'] = 'Safety'
    sample_data['model'] = 'Linear'
    sample_data['timing'] = 'Discrete-Time'
    safety = SafetyBarrier(data=sample_data)

    actual = safety.calculate()

    assert actual['barrier']['expression'] == 'x^T @ P @ x'
    assert 'P' in actual['barrier']['values']

    assert actual['controller']['expression'] == 'U_{0,T} @ Q @ x'
    assert 'U' in actual['controller']['values']
    assert 'Q' in actual['controller']['values']

    assert actual['gamma'] is not None
    assert actual['lambda'] is not None
    assert actual['gamma'] < actual['lambda']
