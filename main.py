import os

from app import create_app

if __name__ == '__main__':
    app = create_app()
    # The development server only listens on localhost unless FLASK_RUN_HOST says otherwise.
    app.run(host=os.environ.get('FLASK_RUN_HOST', '127.0.0.1'), port=5000, ssl_context='adhoc')
