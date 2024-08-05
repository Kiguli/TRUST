from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia

bp = Blueprint('dashboard', __name__)


def getResult():
    return request.args.get('result', None)


@bp.get('/', endpoint='index')
def index():
    models = [
        {'title': "Linear", 'description': ""},
        {'title': "Polynomial", 'description': ""},
    ]

    timings = [
        {'title': "Discrete-Time", 'description': ""},
        {'title': "Continuous-Time", 'description': ""},
    ]

    modes = [
        {'title': "Stability", 'description': ""},
        {'title': "Safety Barrier", 'description': ""},
        {'title': "Reachability Barrier", 'description': "", 'disabled': True},
        {'title': "Reach and Avoid Barrier", 'description': "", 'disabled': True},
    ]

    return render_inertia('Dashboard', {
        'models': models,
        'timings': timings,
        'modes': modes,
        'result': lazy_include(getResult),
    })
