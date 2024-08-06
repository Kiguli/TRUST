from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia
from time import time

bp = Blueprint('dashboard', __name__)


def calculate_result():
    # Validate the request data
    data = request.get_json()

    start_time = time()

    if data['mode'] == 'Stability':
        function_name = 'stability_function'
        stability_function = 1#Stability().calculate(data)
    else:
        function_name = 'barrier_function'
        barrier_function = 2#Barrier().calculate(data)

    time_taken = time() - start_time

    return {
        function_name: locals()[function_name],
        'time_taken': time_taken,
    }


@bp.route('/', endpoint='index', methods=['GET', 'POST'])
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
        'result': lazy_include(calculate_result),
    })
