from flask import Blueprint
from flask_inertia import render_inertia

bp = Blueprint('dashboard', __name__)


@bp.route('/')
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
    })
