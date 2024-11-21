import csv
import json
import os
import tempfile
import tracemalloc
from time import time
from typing import Any

import numpy as np
import sympy as sp
from flask import Blueprint, request
from flask_inertia import lazy_include, render_inertia
from picos import SolutionFailure
from sentry_sdk import capture_exception
from werkzeug.datastructures.file_storage import FileStorage

from app.models.safety_barrier import SafetyBarrier
from app.models.stability import Stability

bp = Blueprint("dashboard", __name__)

monomial_terms = []

def calculate_result() -> dict:
    """
    Calculate the result of the user's input.

    :return: The result of the calculation
    """

    data = request.form.to_dict()

    tracemalloc.start()

    start_time = time()

    try:
        data = _parse_uploaded_files(data)

        if data["mode"] == "Stability":
            results = Stability(data).calculate()
        else:
            results = SafetyBarrier(data).calculate()
    except SolutionFailure as e:
        results = {
            "error": "Solution Failure",
            "description": str(e),
        }
    except Exception as e:
        raise e
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


def validate_monomials() -> bool | tuple[Any, list[Any]]:
    """
    Validate the user's input monomials.
    Monomials should be a list of strings separated by a semicolon, e.g. x1; 2 * x2; x3 - x1
    Each monomial term should be a valid mathematical expression, e.g. 2 * x2
    Each term should also only represent a valid dimension, e.g. x1 to xn, where n is the dimension
    of the dataset.

    :return: Whether the monomials are valid
    """

    monomials = request.get_json()["monomials"]
    if not monomials["terms"]:
        return False

    # Use sympy to validate the monomials
    terms = []
    for monomial in monomials["terms"]:
        try:
            terms.append(sp.sympify(monomial))
        except sp.SympifyError:
            return False

    global monomial_terms
    monomial_terms = terms

    # Get the x terms, and check if they are in the correct format
    # i.e. x1 to xn, where n is the number of dimensions
    # e.g. x1**2; x2 + 2 is valid if dimensions = 2
    dimensions = monomials["dimensions"]
    for term in terms:
        try:
            atoms = term.atoms(sp.Symbol)
        except AttributeError:
            break

        for x in atoms:
            if (
                not x.name.startswith("x")
                or not x.name[1:].isdigit()
                or int(x.name[1:]) > dimensions
            ):
                return False

    return monomials


def generate_theta_x():
    """
    Generate the Theta_x matrix for a given list of terms and variables.
    """

    terms = monomial_terms
    monomials = request.get_json()["monomials"]

    if not terms or not monomials:
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


def _parse_uploaded_files(data: dict) -> dict:
    """
    Parse the uploaded files and add them to the data dictionary.

    :param data: The data dictionary
    """

    for key, file in request.files.items():
        if not file or not file.filename:
            continue

        # Create the `storage/uploads` subfolder if it doesn't exist:
        if not os.path.exists("storage/uploads"):
            os.makedirs("storage/uploads")

        # If file is MOSEK.lic, save it to a subfolder in uploads that we can then add to the env path.
        if key == "mosek_lic":
            __load_mosek_license(file)
        else:
            file.save(f"storage/uploads/{file.filename}")

            with open(f"storage/uploads/{file.filename}", "r") as f:
                # Parse CSV, JSON or txt files
                if file.filename.endswith(".csv"):
                    data[key] = np.array(list(csv.reader(f)))
                elif file.filename.endswith(".json"):
                    data[key] = json.load(f)
                elif file.filename.endswith(".txt"):
                    data[key] = f.read().splitlines()

            # Remove the file from disk
            os.remove(f"storage/uploads/{file.filename}")

    return data


def __load_mosek_license(_file: FileStorage) -> bool:
    """
    Read the MOSEK licence file into this request.
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        license_contents = _file.stream.read()
        temp.write(license_contents)

        os.environ["MOSEKLM_LICENSE_FILE"] = temp.name

        temp.close()

    return True
