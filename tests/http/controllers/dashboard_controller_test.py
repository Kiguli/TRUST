import pytest
import os

from tests import app, client


def test_it_renders_the_dashboard(app, client):
    response = client.get('/')

    assert response.status_code == 200
    assert response.inertia("app").component == "Dashboard"


def test_it_passes_in_the_models(app, client):
    response = client.get('/')

    models = response.inertia("app").props.models
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0].title == "Linear"
    assert models[1].title == "Polynomial"


def test_it_passes_in_the_timings(app, client):
    response = client.get('/')

    timings = response.inertia("app").props.timings
    assert isinstance(timings, list)
    assert len(timings) == 2
    assert timings[0].title == "Discrete-Time"
    assert timings[1].title == "Continuous-Time"


def test_it_passes_in_the_modes(app, client):
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
