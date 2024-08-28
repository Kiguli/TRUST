from app.models.safety_barrier import SafetyBarrier
from tests import sample_data


# Always P and Z is >> 0

def test_it_must_be_in_safety_mode_for_safety_barriers(sample_data):
    sample_data['mode'] = 'Invalid'

    try:
        SafetyBarrier(data=sample_data)
        assert False
    except ValueError as e:
        assert str(e) == f"Invalid mode '{sample_data['mode']}' for Safety Barrier calculations."


def test_it_requires_a_valid_timing_for_safety_barriers(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Invalid'
    safety = SafetyBarrier(data=sample_data)

    try:
        safety.calculate()
        assert False
    except ValueError as e:
        assert str(e) == f"Invalid timing '{sample_data['timing']}' for Safety Barrier calculations."


# --- Discrete-Time Linear System Barrier ---

def test_it_calculates_a_barrier_for_discrete_time_linear_system(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(data=sample_data).calculate()

    assert actual is not None


def test_it_returns_the_correct_barrier_expression(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert actual['barrier']['expression'] == 'x^T @ P @ x'


def test_it_contains_the_barrier_value(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert 'P' in actual['barrier']['values']


def test_it_contains_a_valid_barrier_value(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert actual['barrier']['values']['P'] is not None


def test_it_returns_the_correct_controller_expression(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert actual['controller']['expression'] == 'U_{0,T} @ H @ P @ x'


def test_it_returns_the_correct_controller_values(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert 'U' in actual['controller']['values']
    assert actual['controller']['values']['U'] is not None
    assert 'H' in actual['controller']['values']
    assert actual['controller']['values']['H'] is not None


def test_it_calculates_the_level_sets(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert actual['gamma'] is not None
    assert actual['lambda'] is not None


def test_it_returns_valid_level_sets(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    assert actual['gamma'] < actual['lambda']


def test_it_returns_a_valid_barrier_for_the_level_sets(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    # Assert that B(x) <= gamma for all x in X_initial aka initial_state
    # Assert that B(x) >= lambda for all x in X_unsafe aka unsafe_states


def test_the_next_step_is_less_than_the_current_step(sample_data):
    sample_data = _discrete_linear_setup(sample_data)

    actual = SafetyBarrier(sample_data).calculate()

    # Assert that B(x+) <= B(x)


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

    return sample_data


def _continuous_linear_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Continuous-Time'
    sample_data['model'] = 'Linear'

    return sample_data


def _discrete_polynomial_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Discrete-Time'
    sample_data['model'] = 'Polynomial'

    return sample_data


def _continuous_polynomial_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Continuous-Time'
    sample_data['model'] = 'Polynomial'

    return sample_data
