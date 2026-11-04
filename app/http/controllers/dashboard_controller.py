import csv
import io
import json
import os
import tempfile
import threading
import tracemalloc
from contextlib import contextmanager
from time import time
from typing import Any, Optional

import numpy as np
import sympy as sp
from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia
from picos import SolutionFailure
from sentry_sdk import capture_exception
from werkzeug.datastructures.file_storage import FileStorage

from app.models.expressions import ExpressionError, parse_monomials
from app.models.safety_barrier import SafetyBarrier
from app.models.stability import Stability

bp = Blueprint("dashboard", __name__)

DATASET_KEYS = ("X0", "U0", "X1")
"""Form fields that may hold an uploaded dataset file."""

LICENSE_KEY = "mosek_lic"
"""Form field that holds the uploaded MOSEK licence file."""

MAX_DIMENSIONS = 100
"""Largest state dimension accepted when validating monomials."""

# MOSEK reads its licence location from the process environment and tracemalloc is process-wide,
# so only one calculation may run at a time in each worker process. This keeps one user's licence
# from being used for another user's calculation if the server runs threaded workers.
_calculation_lock = threading.Lock()


class UploadError(ValueError):
    """Raised when an uploaded dataset file cannot be read."""


def calculate_result() -> dict:
    """
    Calculate the result of the user's input.

    :return: The result of the calculation
    """

    data = request.form.to_dict()

    with _calculation_lock:
        tracemalloc.start()

        start_time = time()

        try:
            data = _parse_uploaded_files(data)

            with _mosek_license(request.files.get(LICENSE_KEY)):
                if data["mode"] == "Stability":
                    results = Stability(data).calculate()
                else:
                    results = SafetyBarrier(data).calculate()
        except UploadError as e:
            results = {
                "error": "Unable to parse uploaded file(s).",
                "description": str(e),
            }
        except ExpressionError as e:
            results = {
                "error": "Invalid polynomial expression.",
                "description": str(e),
            }
        except SolutionFailure as e:
            results = {
                "error": "Solution Failure",
                "description": str(e),
            }
        except Exception as e:
            # raise e
            results = {
                "error": "An unknown error occurred.",
                "description": str(e),
            }
            capture_exception(e)

        time_taken = time() - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    results["time_taken"] = f"{time_taken:.5f}s"
    results["memory_used"] = f"{peak / 10**6:.1f}MB"

    return results


def validate_monomials() -> bool | dict[str, Any]:
    """
    Validate the user's input monomials.
    Monomials should be a list of strings separated by a semicolon, e.g. x1; 2 * x2; x3 - x1
    Each monomial term should be a valid polynomial expression, e.g. 2 * x2
    Each term should also only represent a valid dimension, e.g. x1 to xn, where n is the dimension
    of the dataset.

    :return: The monomials if they are valid, otherwise False
    """

    monomials = _get_monomials_payload()
    if monomials is None:
        return False

    try:
        parse_monomials(monomials["terms"], max_index=monomials["dimensions"])
    except ExpressionError:
        return False

    return monomials


def generate_theta_x():
    """
    Generate the Theta_x matrix for a given list of terms and variables.
    """

    monomials = _get_monomials_payload()
    if monomials is None:
        return False

    try:
        terms = parse_monomials(monomials["terms"], max_index=monomials["dimensions"])
    except ExpressionError:
        return False

    variables = [sp.Symbol(f"x{i}") for i in range(1, monomials["dimensions"] + 1)]

    Theta_x = np.zeros((len(terms), len(variables)), dtype=object)

    for row_idx, term in enumerate(terms):
        term_poly = sp.Poly(term, variables)

        # Check each variable in the term
        for col_idx, var in enumerate(variables):
            exponent = term_poly.degree(var)

            if exponent > 0:  # If the variable has a positive exponent in the term
                # Create a modified term by reducing the exponent of `var` by 1
                modified_term = term / var
                # Set this modified term in the corresponding column
                Theta_x[row_idx, col_idx] = modified_term
                break  # Stop after setting the term in the correct column

    return np.array2string(Theta_x, separator=", ").replace("[", "").replace("],", "").replace("]", "")


@bp.route("/", endpoint="index", methods=["GET", "POST"])
def index():
    # TODO: use enums for easier refactoring
    models = [
        {"title": "Linear", "description": ""},
        {"title": "Non-Linear Polynomial", "description": ""},
    ]

    timings = [
        {"title": "Discrete-Time", "description": ""},
        {"title": "Continuous-Time", "description": ""},
    ]

    modes = [
        {"title": "Stability", "description": ""},
        {"title": "Safety", "description": ""},
        {"title": "Reachability", "description": "", "disabled": True},
        {"title": "Reach and Avoid", "description": "", "disabled": True},
    ]

    return render_inertia(
        "Dashboard",
        {
            "models": models,
            "timings": timings,
            "modes": modes,
            "monomials": lazy_include(validate_monomials),
            "theta_x": lazy_include(generate_theta_x),
            "result": lazy_include(calculate_result),
        },
    )


def _get_monomials_payload() -> Optional[dict]:
    """
    Return the ``monomials`` object of a JSON request, if it is well formed.

    It must contain a non-empty list of ``terms`` and the number of ``dimensions``.
    """

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None

    monomials = payload.get("monomials")
    if not isinstance(monomials, dict):
        return None

    terms = monomials.get("terms")
    if not isinstance(terms, list) or not terms:
        return None

    dimensions = monomials.get("dimensions")
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or not 1 <= dimensions <= MAX_DIMENSIONS
    ):
        return None

    return monomials


def _parse_uploaded_files(data: dict) -> dict:
    """
    Parse the uploaded dataset files and add them to the data dictionary.

    Files are parsed in memory. The client-supplied filename is only used to choose the format,
    and is never used as a path on disk.

    :param data: The data dictionary
    """

    for key in DATASET_KEYS:
        file = request.files.get(key)
        if not file or not file.filename:
            continue

        extension = os.path.splitext(file.filename)[1].lower()
        if extension not in (".csv", ".json", ".txt"):
            raise UploadError(f"{key}: unsupported file type. Please upload a .csv, .json or .txt file.")

        try:
            contents = file.stream.read().decode("utf-8-sig")

            # Parse CSV, JSON or txt files
            if extension == ".csv":
                data[key] = np.array(list(csv.reader(io.StringIO(contents, newline=""))))
            elif extension == ".json":
                data[key] = json.loads(contents)
            else:
                data[key] = contents.splitlines()
        except (UnicodeDecodeError, ValueError, csv.Error) as e:
            raise UploadError(f"{key}: {e}") from None

    return data


@contextmanager
def _mosek_license(file: Optional[FileStorage]):
    """
    Make an uploaded MOSEK licence available to the solver for the current calculation only.

    MOSEK finds its licence through the MOSEKLM_LICENSE_FILE environment variable. The licence is
    written to a private temporary file, and afterwards the file is deleted and the variable is
    restored, so a user's licence is neither left on the server nor reused for other users.
    """

    if file is None or not file.filename:
        yield
        return

    previous = os.environ.get("MOSEKLM_LICENSE_FILE")
    descriptor, path = tempfile.mkstemp(prefix="mosek-", suffix=".lic")

    try:
        with os.fdopen(descriptor, "wb") as license_file:
            license_file.write(file.stream.read())

        os.environ["MOSEKLM_LICENSE_FILE"] = path

        yield
    finally:
        if previous is None:
            os.environ.pop("MOSEKLM_LICENSE_FILE", None)
        else:
            os.environ["MOSEKLM_LICENSE_FILE"] = previous

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
