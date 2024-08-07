from flask.testing import FlaskClient, FlaskCliRunner
from flask import Flask
from faker import Faker
from flask_inertia.unittest import InertiaTestResponse
import pytest

from app import create_app

fake = Faker()


@pytest.fixture
def app() -> Flask:
    """Create a new Flask app instance for testing"""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SERVER_NAME': 'localhost',
    })
    app.debug = True
    app.response_class = InertiaTestResponse

    with app.app_context():
        yield app


@pytest.fixture
def client(app) -> FlaskClient:
    """Create a test client for the Flask app"""
    yield app.test_client()


@pytest.fixture
def runner(app) -> FlaskCliRunner:
    """Create a test CLI runner for the Flask app"""
    return app.test_cli_runner()


@pytest.fixture
def sample_data():
    return {
        'model': fake.random_element(['Linear', 'Polynomial']),
        'timing': fake.random_element(['Discrete-Time', 'Continuous-Time']),
        'mode': fake.random_element(['Stability', 'Safety Barrier', 'Reachability Barrier', 'Reach and Avoid Barrier']),
        'X0': [],
        'X1': [],
        'U0': [],
        'stateSpace': {
            'x1': [17, 20],
        },
        'initialState': {
            'x1': [17, 18],
        },
        'unsafeStates': {
            'x1': [19, 20],
        }
    }
