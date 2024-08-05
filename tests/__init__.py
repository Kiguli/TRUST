from flask.testing import FlaskClient, FlaskCliRunner
from flask import Flask
from faker import Faker
import pytest

from app import create_app


@pytest.fixture
def app() -> Flask:
    """Create a new Flask app instance for testing"""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SERVER_NAME': 'localhost',
    })

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
def fake() -> Faker:
    """Create a Faker instance for generating test data"""
    return Faker()
