from flask import Blueprint
from flask_inertia import render_inertia

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    models = [
        {'title': "Linear", 'description': ""},
        {'title': "Polynomial", 'description': ""},
    ]

    return render_inertia('Dashboard', {
        'models': models,
    })
