from app.models.stability import Stability
from tests import sample_data


def test_it_calculates_the_lyapunov_function_and_controller(sample_data):
    stability = Stability().create(sample_data)

    actual = stability.calculate()

    assert isinstance(actual, dict)
    assert 'lyapunov' in actual
    assert 'controller' in actual


def test_it_solves_discrete_time_linear_systems(sample_data):
    """
    The Lyapunov function to return is then
    $\\mathcal{V}(x) = x^\\top Px$ and the controller to return is $u=\\mathcal{U}_{0,T}HP^{-1}x$.
    """
    sample_data['mode'] = 'Stability'
    sample_data['model'] = 'Linear'
    sample_data['timing'] = 'Discrete-Time'
    stability = Stability().create(sample_data)

    actual = stability.calculate()

    # discrete-time linear Lyapunov function is V(x) = x^T @ P @ x
    assert actual['lyapunov']['expression'] == 'x^T @ P @ x'
    assert 'P' in actual['lyapunov']['values']

    # discrete-time linear controller expression is u = U_{0,T} @ H @ P^{-1} @ x
    assert actual['controller']['expression'] == 'U_{0,T} @ H @ P^{-1} @ x'
    assert 'H' in actual['controller']['values']
    assert 'P' in actual['controller']['values']


def test_it_solves_continuous_time_linear_systems(sample_data):
    """
    The Lyapunov function to return is then
    $\\mathcal{V}(x) = x^\\top Px$ and the controller to return is $u=\\mathcal{U}_{0,T}HP^{-1}x$.
    """
    sample_data['mode'] = 'Stability'
    sample_data['model'] = 'Linear'
    sample_data['timing'] = 'Continuous-Time'
    stability = Stability().create(sample_data)

    actual = stability.calculate()

    # discrete-time linear Lyapunov function is V(x) = x^T @ P @ x
    assert actual['lyapunov']['expression'] == 'x^T @ P @ x'
    assert 'P' in actual['lyapunov']['values']

    # discrete-time linear controller expression is u = U_{0,T} @ H @ P^{-1} @ x
    assert actual['controller']['expression'] == 'U_{0,T} @ H @ P^{-1} @ x'
    assert 'H' in actual['controller']['values']
    assert 'P' in actual['controller']['values']
