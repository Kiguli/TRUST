from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia
from time import time, sleep
from json import dumps

import tests
from app.models.stability import Stability
from app.models.safety_barrier import SafetyBarrier

bp = Blueprint('dashboard', __name__)


def calculate_result(update_cache=False):
    """
    Calculate the result of the user's input.

    :param  update_cache whether to update the cache or not
    :return: the result of the calculation
    """

    # TODO: validate data
    # data = request.get_json()

    # TODO: DEBUG - use test data
    data = tests.fake_data()
    data['mode'] = 'Safety'
    data['timing'] = 'Discrete-Time'
    data['model'] = 'Linear'

    start_time = time()

    if data['mode'] == 'Stability':
        function_name = 'stability_function'
        stability_function = Stability().create(data).calculate()
    else:
        function_name = 'barrier_function'
        # barrier = BarrierFactory()
        barrier_function = SafetyBarrier(data).calculate()

    time_taken = time() - start_time

    result = {
        function_name: locals()[function_name],
        'time_taken': f"{time_taken:.5f}s",
    }

    return result


@bp.route('/', endpoint='index', methods=['GET', 'POST'])
def index():
    # TODO: use enums for easier refactoring
    models = [
        {'title': "Linear", 'description': ""},
        {'title': "Non-Linear Polynomial", 'description': ""},
    ]

    timings = [
        {'title': "Discrete-Time", 'description': ""},
        {'title': "Continuous-Time", 'description': ""},
    ]

    modes = [
        {'title': "Stability", 'description': ""},
        {'title': "Safety", 'description': ""},
        {'title': "Reachability", 'description': "", 'disabled': True},
        {'title': "Reach and Avoid", 'description': "", 'disabled': True},
    ]

    return render_inertia('Dashboard', {
        'models': models,
        'timings': timings,
        'modes': modes,
        'result': lazy_include(calculate_result),
    })
