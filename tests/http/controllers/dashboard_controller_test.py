import pytest
from pytest import mark
from tests import app, client


def test_it_renders_the_dashboard(client):
    response = client.get('/')

    assert response.status_code == 200
    assert response.inertia("app").component == "Dashboard"


def test_it_passes_in_the_models(client):
    response = client.get('/')

    models = response.inertia("app").props.models
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0].title == "Linear"
    assert models[1].title == "Polynomial"


def test_it_passes_in_the_timings(client):
    response = client.get('/')

    timings = response.inertia("app").props.timings
    assert isinstance(timings, list)
    assert len(timings) == 2
    assert timings[0].title == "Discrete-Time"
    assert timings[1].title == "Continuous-Time"


def test_it_passes_in_the_modes(client):
    response = client.get('/')

    modes = response.inertia("app").props.modes
    assert isinstance(modes, list)
    assert len(modes) == 4
    assert modes[0].title == "Stability"
    assert modes[1].title == "Safety Barrier"
    assert modes[2].title == "Reachability Barrier"
    assert modes[2].disabled is True
    assert modes[3].title == "Reach and Avoid Barrier"
    assert modes[3].disabled is True


def test_it_has_a_lazy_loaded_result(client):
    headers = {
        'X-Inertia': 'true',
        'X-Inertia-Partial-Data': ['result'],
        'X-Inertia-Partial-Component': 'Dashboard',
    }

    response = client.get('/?result=0', headers=headers)

    expected = str(0)
    actual = response.inertia("app").props.result

    assert response.status_code == 200
    assert expected == actual


@mark.skip
def test_it_calculates_the_stability_function_and_redirects_to_the_dashboard_with_the_result(client):
    response = client.post('/', data=sample_config())

    assert response.status_code == 302
    assert response.inertia("app").props.result == 0


@mark.skip
def test_it_requires_a_model(client):
    pass


@mark.skip
def test_it_requires_a_timing(client):
    pass


@mark.skip
def test_it_requires_a_mode(client):
    pass


@mark.skip
def test_it_requires_data(client):
    pass


@mark.skip
def test_it_requires_monomials_for_non_linear_models(client):
    pass


@mark.skip
def test_it_requires_a_state_space(client):
    pass


@mark.skip
def test_it_requires_an_n_dimensional_state_space(client):
    pass


@mark.skip
def test_it_requires_an_initial_state(client):
    pass


@mark.skip
def test_it_requires_an_unsafe_state(client):
    pass


@mark.skip
def test_it_has_many_unsafe_states(client):
    pass


@mark.skip
def test_it_requires_states_match_dimensionality(client):
    pass


def sample_config():
    return {
        'model': 'Linear',
        'timing': 'Discrete-Time',
        'mode': 'Stability',
        'data': [],
        'state_space': {
            'x1': [17, 20],
        },
        'initial_state': {
            'x1': [17, 18],
        },
        'unsafe_state': {
            'x1': [19, 20],
        }
    }
