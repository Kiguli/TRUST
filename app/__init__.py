import os
from dotenv import load_dotenv
from flask import Flask
from flask_inertia import Inertia, render_inertia
from flask_vite import Vite
from urllib.parse import urlparse


def create_app(test_config=None):
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder='vite',
        static_folder='vite')

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    app.instance_path = os.path.abspath(os.path.dirname(__file__))
    app.root_path = os.path.abspath(os.path.join(app.instance_path, '..'))

    app.config['INERTIA_TEMPLATE'] = "index.html"
    app.config['VITE_AUTO_INSERT'] = True

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    Inertia(app)
    Vite(app)

    # --- Controllers ---
    @app.route('/')
    def index():
        return render_inertia('Dashboard', {
            'foo': 'bar',
        })

    return app
