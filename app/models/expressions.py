"""
Safe parsing of user-supplied polynomial expressions.

``sympy.sympify`` (and ``parse_expr``) evaluate their input with ``eval``, so they must never be
given untrusted text: a "monomial" such as ``__import__('os').system('...')`` would run arbitrary
code on the server. Monomials and Theta(x) entries are therefore parsed here with Python's ``ast``
module and converted to SymPy objects node by node. Only numbers, the state variables ``x1``,
``x2``, ..., parentheses and the operators ``+ - * / **`` are accepted (``^`` is treated as
``**``, as ``sympify`` does), so the result is always a polynomial in the state variables.

The size of the polynomial is bounded as well, because expanding something like
``(x1 + x2 + x3) ** 1000`` later on would tie up a worker for a very long time.
"""

import ast
import decimal
import math
import operator
import re
from typing import List, Optional

import sympy as sp

MAX_EXPRESSION_LENGTH = 500
"""Maximum number of characters in a single expression."""

MAX_DEPTH = 300
"""Maximum nesting depth of a single expression."""

MAX_DEGREE = 6
"""
Maximum total degree of a polynomial. The benchmarks use degree 3 at most, and the cost of the
SOS multipliers grows steeply with the degree (for continuous-time systems it is doubled).
"""

MAX_EXPONENT = 32
"""Largest product of nested exponents, which bounds the size of constants such as 10**-8."""

MAX_FLOAT_EXPONENT = 308
"""Largest decimal exponent of a number, roughly the range of a double."""

MAX_EXPANDED_TERMS = 1_000
"""Upper bound on the number of monomials a polynomial may expand to."""

MAX_MONOMIALS = 256
"""Maximum number of monomial terms, N."""

_STATE_VARIABLE = re.compile(r"x([1-9][0-9]{0,5})")

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class ExpressionError(ValueError):
    """Raised when user input is not a supported polynomial expression."""


class _Node:
    """A converted sub-expression together with bounds on its size once expanded."""

    __slots__ = ("expr", "degree", "terms", "power")

    def __init__(self, expr: sp.Expr, degree: int, terms: int, power: int):
        self.expr = expr
        self.degree = degree  # upper bound on the total degree
        self.terms = terms  # upper bound on the number of monomials after expansion
        self.power = power  # product of the exponents applied along any path


def parse_monomials(terms, max_index: Optional[int] = None) -> List[sp.Expr]:
    """
    Parse a list of monomial terms with :func:`parse_polynomial`.

    :param terms: The monomial terms, e.g. ``["x1", "x2", "x1*x2"]``
    :param max_index: If given, only the variables ``x1`` to ``x{max_index}`` are allowed
    :return: The SymPy expressions
    :raises ExpressionError: If any term is invalid
    """

    if not isinstance(terms, list) or not terms:
        raise ExpressionError("Provide at least one monomial term.")

    if len(terms) > MAX_MONOMIALS:
        raise ExpressionError(f"At most {MAX_MONOMIALS} monomial terms are supported.")

    return [parse_polynomial(term, max_index) for term in terms]


def parse_polynomial(expression, max_index: Optional[int] = None) -> sp.Expr:
    """
    Convert user input such as ``"2 * x1**2 - x2"`` into a SymPy expression without using ``eval``.

    :param expression: The expression, as a string or a plain number
    :param max_index: If given, only the variables ``x1`` to ``x{max_index}`` are allowed
    :return: The SymPy expression
    :raises ExpressionError: If the input is not a supported polynomial expression
    """

    if isinstance(expression, bool):
        raise ExpressionError("Expected a polynomial expression, not a boolean.")

    if isinstance(expression, int):
        return sp.Integer(expression)

    if isinstance(expression, float):
        if not math.isfinite(expression):
            raise ExpressionError("Numbers must be finite.")
        return sp.Float(expression)

    if not isinstance(expression, str):
        raise ExpressionError("Expected a polynomial expression.")

    source = expression.strip()
    if not source:
        raise ExpressionError("Expressions cannot be empty.")

    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"Expressions must be at most {MAX_EXPRESSION_LENGTH} characters long."
        )

    # Like sympify, accept ^ as the power operator.
    source = source.replace("^", "**")

    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ExpressionError(f"'{expression}' is not a valid polynomial expression.") from None

    return _convert(tree.body, source, max_index, depth=0).expr


def _convert(node: ast.AST, source: str, max_index: Optional[int], depth: int) -> _Node:
    if depth > MAX_DEPTH:
        raise ExpressionError("Expression is nested too deeply.")

    if isinstance(node, ast.Constant):
        return _constant(node, source)

    if isinstance(node, ast.Name):
        return _variable(node.id, max_index)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _convert(node.operand, source, max_index, depth + 1)
        expr = _UNARY_OPERATORS[type(node.op)](operand.expr)
        return _Node(expr, operand.degree, operand.terms, operand.power)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _convert(node.left, source, max_index, depth + 1)
        right = _convert(node.right, source, max_index, depth + 1)

        if isinstance(node.op, ast.Pow):
            return _power(left, right)

        if isinstance(node.op, (ast.Add, ast.Sub)):
            bounds = (
                max(left.degree, right.degree),
                left.terms + right.terms,
                max(left.power, right.power),
            )
        elif isinstance(node.op, ast.Mult):
            bounds = (
                left.degree + right.degree,
                left.terms * right.terms,
                max(left.power, right.power),
            )
        else:  # Division is only allowed by a non-zero constant.
            if right.degree != 0 or not right.expr.is_number:
                raise ExpressionError("Only division by a constant is supported.")
            if right.expr.is_zero:
                raise ExpressionError("Division by zero.")
            bounds = (left.degree, left.terms, max(left.power, right.power))

        _check_bounds(*bounds)
        expr = _BINARY_OPERATORS[type(node.op)](left.expr, right.expr)
        return _Node(expr, *bounds)

    raise ExpressionError(
        "Only numbers, the variables x1 to xn, parentheses and + - * / ** are allowed."
    )


def _constant(node: ast.Constant, source: str) -> _Node:
    value = node.value

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpressionError(
            "Only numbers, the variables x1 to xn, parentheses and + - * / ** are allowed."
        )

    if isinstance(value, int):
        return _Node(sp.Integer(value), 0, 1, 1)

    # Build Floats from their source text, as sympify does, so precision is preserved. SymPy
    # converts the text exactly, so check the exponent first: "1e999999999" would never finish.
    literal = ast.get_source_segment(source, node) or repr(value)
    try:
        exponent = decimal.Decimal(literal).adjusted()
    except decimal.InvalidOperation:
        literal, exponent = repr(value), 0

    if not math.isfinite(value) or abs(exponent) > MAX_FLOAT_EXPONENT:
        raise ExpressionError(
            f"Numbers must be between 1e-{MAX_FLOAT_EXPONENT} and 1e{MAX_FLOAT_EXPONENT}."
        )

    try:
        number = sp.Float(literal)
    except (TypeError, ValueError):
        number = sp.Float(value)

    return _Node(number, 0, 1, 1)


def _variable(name: str, max_index: Optional[int]) -> _Node:
    match = _STATE_VARIABLE.fullmatch(name)
    if match is None:
        raise ExpressionError(f"Unknown variable '{name}'. Use the state variables x1 to xn.")

    if max_index is not None and int(match.group(1)) > max_index:
        raise ExpressionError(
            f"Unknown variable '{name}'. Use the state variables x1 to x{max_index}."
        )

    return _Node(sp.Symbol(name), 1, 1, 1)


def _power(base: _Node, exponent: _Node) -> _Node:
    value = exponent.expr

    # Accept integral floats such as 2.0, but nothing that would make the result non-polynomial.
    if value.is_Float and math.isfinite(float(value)) and float(value).is_integer():
        value = sp.Integer(int(float(value)))

    if exponent.degree != 0 or not value.is_Integer:
        raise ExpressionError("Exponents must be whole numbers.")

    k = int(value)
    if abs(k) > MAX_EXPONENT:
        raise ExpressionError(f"Exponents must be at most {MAX_EXPONENT}.")

    if k < 0 and (base.degree != 0 or base.expr.is_zero):
        raise ExpressionError("Negative exponents are not allowed on variables.")

    bounds = (
        base.degree * abs(k),
        math.comb(base.terms + abs(k) - 1, abs(k)) if k else 1,
        max(base.power, exponent.power) * max(abs(k), 1),
    )
    _check_bounds(*bounds)

    return _Node(base.expr ** sp.Integer(k), *bounds)


def _check_bounds(degree: int, terms: int, power: int) -> None:
    if degree > MAX_DEGREE:
        raise ExpressionError(f"Polynomials must have degree at most {MAX_DEGREE}.")

    if power > MAX_EXPONENT:
        raise ExpressionError(f"Exponents must be at most {MAX_EXPONENT}.")

    if terms > MAX_EXPANDED_TERMS:
        raise ExpressionError("Polynomial is too large.")
