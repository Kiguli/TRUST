import os

import sentry_sdk
from dotenv import load_dotenv
from flask import Flask, request
from flask_inertia import Inertia
from flask_vite import Vite
from sentry_sdk.integrations.flask import FlaskIntegration

load_dotenv()


def create_app(test_config=None):
    # -- App Config --
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="vite",
        static_folder="vite/dist",
        static_url_path="/",
    )

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    app.instance_path = os.path.abspath(os.path.dirname(__file__))
    app.root_path = os.path.abspath(os.path.join(app.instance_path, ".."))

    app.config["INERTIA_TEMPLATE"] = "index.html"
    app.config["VITE_AUTO_INSERT"] = True
    app.config["SECRET_KEY"] = os.environ.get("FLASK_KEY")
    app.config["FLASK_ENV"] = os.environ.get("FLASK_ENV")

    if app.config["FLASK_ENV"] == "production":
        app.config["PREFERRED_URL_SCHEME"] = "https"

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    Inertia(app)
    Vite(app)

    # --- Sentry ---
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=app.config["FLASK_ENV"],
    )

    @app.before_request
    def before_request():
        request.scheme = app.config["PREFERRED_URL_SCHEME"]

    # --- Routes ---

    # --- Controllers ---
    from app.http.controllers import dashboard_controller

    app.register_blueprint(dashboard_controller.bp)

    return app
