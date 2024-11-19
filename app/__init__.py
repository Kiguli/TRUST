import os
from dotenv import load_dotenv
from flask import Flask, redirect, request
from flask_inertia import Inertia
from flask_vite import Vite

load_dotenv()

def create_app(test_config=None):
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder='vite',
        static_folder='vite/dist',
        static_url_path='/'
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    app.instance_path = os.path.abspath(os.path.dirname(__file__))
    app.root_path = os.path.abspath(os.path.join(app.instance_path, '..'))

    app.config['INERTIA_TEMPLATE'] = "index.html"
    app.config['VITE_AUTO_INSERT'] = False
    app.config['SECRET_KEY'] = os.environ.get('APP_KEY')

    if os.environ.get('APP_ENV') == 'production':
        app.config['PREFERRED_URL_SCHEME'] = 'https'

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    Inertia(app)
    Vite(app)

    # @app.before_request
    # def before_request():
    #     if not request.is_secure and os.environ.get('APP_ENV') == 'production':
    #         url = request.url.replace('http://', 'https://', 1)
    #         return redirect(url, code=302)

    # --- Controllers ---
    from app.http.controllers import dashboard_controller
    app.register_blueprint(dashboard_controller.bp)

    return app
