import pytest
import os

from tests import app, client


def test_it_renders_the_dashboard(app, client):
    response = client.get('/')

    assert response.status_code == 200
    assert response.inertia("app").component == "Dashboard"
