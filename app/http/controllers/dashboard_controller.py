from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia
from time import time, sleep
from json import dumps

from app.models.stability import Stability
from app.models.safety_barrier import SafetyBarrier

bp = Blueprint('dashboard', __name__)


def calculate_result():
    # TODO: validate data
    data = request.get_json()

    start_time = time()

    if data['mode'] == 'Stability':
        function_name = 'stability_function'
        stability_function = Stability(data).calculate()
    else:
        function_name = 'barrier_function'
        # barrier = BarrierFactory()
        barrier_function = SafetyBarrier(data).calculate()

    time_taken = time() - start_time

    result = {
        function_name: locals()[function_name],
        'time_taken': time_taken,
    }

    return result


@bp.route('/', endpoint='index', methods=['GET', 'POST'])
def index():
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
