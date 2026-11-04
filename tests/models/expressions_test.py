import time

import pytest
import sympy as sp

from app.models.expressions import (
    MAX_EXPRESSION_LENGTH,
    MAX_MONOMIALS,
    ExpressionError,
    parse_monomials,
    parse_polynomial,
)

x1, x2, x3 = sp.symbols("x1 x2 x3")

TOO_LONG = "1" + "+1" * (MAX_EXPRESSION_LENGTH // 2)
"""A valid sum of ones that is one character longer than allowed."""


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("x1", x1),
        ("x1*x2", x1 * x2),
        ("x1**2 * x2", x1**2 * x2),
        (" x3 - x1 ", x3 - x1),
        ("x1^2", x1**2),
        ("(x1 + x2)**2", (x1 + x2) ** 2),
        ("x1/2", x1 / 2),
        ("1/2*x1", x1 / 2),
        ("0.5*x1", sp.Float("0.5") * x1),
        ("x1**2.0", x1**2),
        (0, sp.Integer(0)),
        (2.5, sp.Float(2.5)),
    ],
)
def test_it_parses_polynomials(expression, expected):
    assert parse_polynomial(expression) == expected


def test_it_parses_monomials():
    assert parse_monomials(["x1", "x2", "x1*x2"], max_index=2) == [x1, x2, x1 * x2]


def test_it_never_executes_code(tmp_path):
    marker = tmp_path / "executed"
    payload = f"__import__('pathlib').Path('{marker}').touch()"
    assert len(payload) <= MAX_EXPRESSION_LENGTH  # rejected for its content, not its length

    with pytest.raises(ExpressionError):
        parse_polynomial(payload)

    with pytest.raises(ExpressionError):
        parse_monomials(["x1", payload])

    assert not marker.exists()


@pytest.mark.parametrize(
    "expression",
    [
        "sin(x1)",  # function calls
        "pi*x1",  # names other than the state variables
        "x0",  # state variables start at x1
        "x1/x2",  # division by a variable
        "x1**-1",  # negative powers of variables
        "x1**0.5",  # fractional powers
        "x1/0",
        "x1.real",  # attribute access
        "x1 if x2 else x3",
        "'x1'",
        "True",
        "",
        "   ",
    ],
)
def test_it_rejects_anything_but_polynomials(expression):
    with pytest.raises(ExpressionError):
        parse_polynomial(expression)


@pytest.mark.parametrize("expression", [None, True, ["x1"], {"x1": 1}, float("inf"), float("nan")])
def test_it_rejects_values_that_are_not_expressions(expression):
    with pytest.raises(ExpressionError):
        parse_polynomial(expression)


def test_it_only_accepts_the_given_state_variables():
    assert parse_polynomial("x1 * x2", max_index=2) == x1 * x2

    with pytest.raises(ExpressionError):
        parse_polynomial("x3", max_index=2)

    with pytest.raises(ExpressionError):
        parse_monomials(["x1", "x3"], max_index=2)


@pytest.mark.parametrize(
    "expression",
    [
        "x1**7",  # degree above the limit
        "x1**99999999999",
        "9**9**9**9",
        "((9**32)**32)**32",
        "1e999999999",
        "x1/1e-999999999",
        "(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)**6",  # expands to 5005 terms
        TOO_LONG,
    ],
)
def test_it_quickly_rejects_expressions_that_are_too_large(expression):
    start = time.perf_counter()

    with pytest.raises(ExpressionError):
        parse_polynomial(expression)

    assert time.perf_counter() - start < 1


def test_the_long_expression_is_one_character_over_the_limit():
    assert len(TOO_LONG) == MAX_EXPRESSION_LENGTH + 1 == 501
    assert parse_polynomial(TOO_LONG[:-2]) == 250  # the same input within the limit is accepted


def test_it_quickly_rejects_too_many_monomials():
    start = time.perf_counter()

    with pytest.raises(ExpressionError):
        parse_monomials(["x1"] * (MAX_MONOMIALS + 1))

    assert time.perf_counter() - start < 1


def test_it_requires_at_least_one_monomial():
    with pytest.raises(ExpressionError):
        parse_monomials([])

    with pytest.raises(ExpressionError):
        parse_monomials("x1")
