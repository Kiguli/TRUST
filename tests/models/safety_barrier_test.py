from app.models.safety_barrier import SafetyBarrier
from tests import sample_data


# --- Discrete-Time Linear System Barrier ---

def test_it_returns_the_correct_barrier_expression(sample_data):
    actual = _discrete_setup(sample_data)

    assert actual['barrier']['expression'] == 'x^T @ P @ x'


def test_it_contains_the_barrier_value(sample_data):
    actual = _discrete_setup(sample_data)

    assert 'P' in actual['barrier']['values']


def test_it_contains_a_valid_barrier_value(sample_data):
    actual = _discrete_setup(sample_data)

    assert actual['barrier']['values']['P'] is not None


def test_it_returns_the_correct_controller_expression(sample_data):
    actual = _discrete_setup(sample_data)

    assert actual['controller']['expression'] == 'U_{0,T} @ H @ P @ x'


def test_it_returns_the_correct_controller_values(sample_data):
    actual = _discrete_setup(sample_data)

    assert 'U' in actual['controller']['values']
    assert actual['controller']['values']['U'] is not None
    assert 'H' in actual['controller']['values']
    assert actual['controller']['values']['H'] is not None


def test_it_calculates_the_level_sets(sample_data):
    actual = _discrete_setup(sample_data)

    assert actual['gamma'] is not None
    assert actual['lambda'] is not None


def test_it_returns_valid_level_sets(sample_data):
    actual = _discrete_setup(sample_data)

    assert actual['gamma'] < actual['lambda']


# --- Continuous-Time Linear System Barrier ---

def test_it_returns_correctly_for_continuous_time_linear_safety_barrier_certificates(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['model'] = 'Linear'
    sample_data['timing'] = 'Continuous-Time'
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


def _discrete_linear_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Discrete-Time'
    sample_data['model'] = 'Linear'
    safety = SafetyBarrier(data=sample_data)

    return safety.calculate()


def _continuous_linear_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Continuous-Time'
    sample_data['model'] = 'Linear'
    safety = SafetyBarrier(data=sample_data)

    return safety.calculate()


def _discrete_polynomial_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Discrete-Time'
    sample_data['model'] = 'Polynomial'
    safety = SafetyBarrier(data=sample_data)

    return safety.calculate()


def _continuous_polynomial_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Continuous-Time'
    sample_data['model'] = 'Polynomial'
    safety = SafetyBarrier(data=sample_data)

    return safety.calculate()
