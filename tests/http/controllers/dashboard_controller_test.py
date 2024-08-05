import pytest
import os

from tests import app, client


def test_it_renders_the_dashboard(app, client):
    response = client.get('/')

    assert response.status_code == 200
    assert response.inertia("app").component == "Dashboard"


def test_it_passes_in_the_model_props(app, client):
    response = client.get('/')

    models = response.inertia("app").props.models
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0].title == "Linear"
    assert models[1].title == "Polynomial"
