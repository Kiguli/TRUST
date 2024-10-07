import pytest

from app.models.safety_barrier import SafetyBarrier
from tests import sample_data


class TestSafetyBarrier:

    def test_it_must_be_in_safety_mode_for_safety_barriers(self, sample_data):
        sample_data['mode'] = 'Invalid'

        try:
            SafetyBarrier(data=sample_data)
            assert False
        except ValueError as e:
            assert str(e) == f"Invalid mode '{sample_data['mode']}' for Safety Barrier calculations."

    def test_it_requires_a_valid_timing_for_safety_barriers(self, sample_data):
        sample_data['mode'] = 'Safety'
        sample_data['timing'] = 'Invalid'
        safety = SafetyBarrier(data=sample_data)

        try:
            safety.calculate()
            assert False
        except ValueError as e:
            assert str(e) == f"Invalid timing '{sample_data['timing']}' for Safety Barrier calculations."


class TestDiscreteTimeLinearBarrier:

    def test_it_returns_a_response_for_discrete_time_linear_system(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual is not None

    @pytest.mark.skip()
    def test_it_returns_an_error_for_invalid_solutions(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = None

        try:
            actual = SafetyBarrier(sample_data).calculate()
            assert False
        except Exception as e:
            assert actual['error'] == 'An unknown error occurred.'
            assert actual['description'] is not None

    def test_it_returns_the_barrier(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(sample_data).calculate()

        assert actual['barrier']['expression'] == 'x<sup>T</sup>Px'
        assert 'P' in actual['barrier']['values']
        assert actual['barrier']['values']['P'] is not None

    def test_it_returns_the_correct_controller_expression(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(sample_data).calculate()

        assert actual['controller']['expression'] == 'U<sub>0</sub>HPx'
        assert 'H' in actual['controller']['values']
        assert actual['controller']['values']['H'] is not None

    def test_it_returns_valid_level_sets(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(sample_data).calculate()

        assert actual['gamma'] is not None
        assert actual['lambda'] is not None
        assert actual['gamma'] < actual['lambda']

    @pytest.mark.skip()
    def test_it_returns_a_valid_barrier_for_the_level_sets(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(sample_data).calculate()

        # TODO: Assert that B(x) <= gamma for all x in X_initial aka initial_state
        # TODO: Assert that B(x) >= lambda for all x in X_unsafe aka unsafe_states

    @pytest.mark.skip()
    def test_the_next_step_is_less_than_the_current_step(self, sample_data):
        sample_data = _discrete_linear_setup(sample_data)

        actual = SafetyBarrier(sample_data).calculate()

        # TODO: Assert that B(x+) <= B(x)


class TestDiscreteTimeNonlinearPolynomialBarrier:

    def test_it_returns_a_response_for_discrete_time_polynomial_system(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual is not None

    def test_it_returns_the_barrier(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['barrier']['expression'] == 'x<sup>T</sup>Px'
        assert 'P' in actual['barrier']['values']
        assert actual['barrier']['values']['P'] is not None

    def test_it_returns_the_controller(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['controller']['expression'] == 'U<sub>0</sub>H(x)[N<sub>0</sub>H(x)]<sup>-1</sup>x'
        assert 'H' in actual['controller']['values']
        assert actual['controller']['values']['H'] is not None
        assert 'N' in actual['controller']['values']
        assert actual['controller']['values']['N'] is not None

    def test_it_returns_the_level_sets(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['gamma'] is not None
        assert actual['lambda'] is not None
        assert actual['gamma'] < actual['lambda']
        print(actual)


class TestContinuousTimeLinearBarrier:

    def test_it_returns_a_response_for_continuous_time_linear_system(self, sample_data):
        sample_data = _continuous_linear_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual is not None

    def test_it_returns_the_barrier(self, sample_data):
        sample_data = _continuous_linear_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['barrier']['expression'] == 'x^T @ P @ x'
        assert 'P' in actual['barrier']['values']
        assert actual['barrier']['values']['P'] is not None

    def test_it_returns_the_controller(self, sample_data):
        sample_data = _continuous_linear_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['controller']['expression'] == 'U_{0,T} @ Q @ x'
        assert 'Q' in actual['controller']['values']
        assert actual['controller']['values']['Q'] is not None

    def test_it_returns_the_level_sets(self, sample_data):
        sample_data = _continuous_linear_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['gamma'] is not None
        assert actual['lambda'] is not None
        assert actual['gamma'] < actual['lambda']


class TestContinuousTimeNonlinearPolynomialBarrier:

    def test_it_returns_a_response_for_continuous_time_nonlinear_polynomial_system(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual is not None

    def test_it_returns_the_barrier(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['barrier']['expression'] == 'M(x)^T @ P @ M(x)'
        assert 'P' in actual['barrier']['values']
        assert actual['barrier']['values']['P'] is not None

    def test_it_returns_the_controller(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

        assert actual['controller']['expression'] == 'U0 @ H(x) @ P @ M(x)'
        assert 'H(x)' in actual['controller']['values']
        assert actual['controller']['values']['H(x)'] is not None

    def test_it_returns_the_level_sets(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        actual = SafetyBarrier(data=sample_data).calculate()

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
    sample_data['model'] = 'Non-Linear Polynomial'
    sample_data['monomials'] = ['x1', 'x2', 'x1*x2']

    return sample_data


def _continuous_polynomial_setup(sample_data):
    sample_data['mode'] = 'Safety'
    sample_data['timing'] = 'Continuous-Time'
    sample_data['model'] = 'Non-Linear Polynomial'
    sample_data['monomials'] = ['x1', 'x2', 'x1*x2', 'x2 - x1']

    return sample_data
