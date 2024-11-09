from app.models.stability import Stability
from tests import sample_data


def test_it_calculates_the_lyapunov_function_and_controller(sample_data):
    stability = Stability(sample_data)

    actual = stability.calculate()

    assert isinstance(actual, dict)
    assert "lyapunov" in actual
    assert "controller" in actual


def test_it_solves_discrete_time_linear_systems(sample_data):
    """
    The Lyapunov function to return is then
    $\\mathcal{V}(x) = x^\\top Px$ and the controller to return is $u=\\mathcal{U}_{0,T}HP^{-1}x$.
    """
    sample_data["mode"] = "Stability"
    sample_data["model"] = "Linear"
    sample_data["timing"] = "Discrete-Time"
    stability = Stability(sample_data)

    actual = stability.calculate()

    # discrete-time linear Lyapunov function is V(x) = x^T @ P @ x
    assert actual["lyapunov"]["expression"] == "x^T @ P @ x"
    assert "P" in actual["lyapunov"]["values"]

    # discrete-time linear controller expression is u = U_{0,T} @ H @ P^{-1} @ x
    assert actual["controller"]["expression"] == "U_{0,T} @ H @ P^{-1} @ x"
    assert "H" in actual["controller"]["values"]
    assert "P" in actual["controller"]["values"]


def test_it_solves_continuous_time_linear_systems(sample_data):
    """
    The Lyapunov function to return is then
    $\\mathcal{V}(x) = x^\\top Px$ and the controller to return is $u=\\mathcal{U}_{0,T}HP^{-1}x$.
    """
    sample_data["mode"] = "Stability"
    sample_data["model"] = "Linear"
    sample_data["timing"] = "Continuous-Time"
    stability = Stability(sample_data)

    actual = stability.calculate()

    # discrete-time linear Lyapunov function is V(x) = x^T @ P @ x
    assert actual["lyapunov"]["expression"] == "x^T @ P @ x"
    assert "P" in actual["lyapunov"]["values"]

    # discrete-time linear controller expression is u = U_{0,T} @ H @ P^{-1} @ x
    assert actual["controller"]["expression"] == "U_{0,T} @ H @ P^{-1} @ x"
    assert "H" in actual["controller"]["values"]
    assert "P" in actual["controller"]["values"]


class TestContinuousTimeNonlinearPolynomialSafety:

    def test_it_returns_a_response(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        response = Stability(sample_data).calculate()

        assert isinstance(response, dict)
        assert "lyapunov" in response
        assert "controller" in response

    def test_it_returns_the_Lyapunov_function(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        response = Stability(sample_data).calculate()

        assert "expression" in response["lyapunov"]
        assert "values" in response["lyapunov"]
        assert "P" in response["lyapunov"]["values"]
        assert response["lyapunov"]["values"]["P"] is not None

    def test_it_returns_the_controller(self, sample_data):
        sample_data = _continuous_polynomial_setup(sample_data)

        response = Stability(sample_data).calculate()

        assert "expression" in response["controller"]
        assert "values" in response["controller"]
        assert "H(x)" in response["controller"]["values"]
        assert response["controller"]["values"]["H(x)"] is not None


class TestDiscreteTimeNonlinearPolynomialSafety:

    def test_it_returns_a_response(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        response = Stability(sample_data).calculate()

        assert isinstance(response, dict)
        assert "lyapunov" in response
        assert "controller" in response

    def test_it_returns_the_Lyapunov_function(self, sample_data):
        sample_data = _discrete_polynomial_setup(sample_data)

        response = Stability(sample_data).calculate()

        assert "expression" in response["lyapunov"]
        assert "values" in response["lyapunov"]
        assert "P" in response["lyapunov"]["values"]
        assert response["lyapunov"]["values"]["P"] is not None
        assert isinstance(response["lyapunov"]["values"]["P"], list)


def _continuous_polynomial_setup(sample_data):
    sample_data["mode"] = "Stability"
    sample_data["timing"] = "Continuous-Time"
    sample_data["model"] = "Non-Linear Polynomial"
    sample_data["monomials"] = ["x1", "x2", "x1*x2", "x2 - x1"]

    return sample_data


def _discrete_polynomial_setup(sample_data):
    sample_data["mode"] = "Stability"
    sample_data["timing"] = "Discrete-Time"
    sample_data["model"] = "Non-Linear Polynomial"
    sample_data["monomials"] = ["x1", "x2", "x1*x2", "x2 - x1"]

    return sample_data
