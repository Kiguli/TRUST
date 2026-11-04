import io
import json
import os
import stat
import tempfile

import mosek
import pytest
from picos.solvers.solver_mosek import MOSEKSolver
from werkzeug.datastructures import FileStorage

from app import create_app
from app.http.controllers import dashboard_controller
from app.http.controllers.dashboard_controller import (
    UploadError,
    _mosek_environment,
    _mosek_license,
)
from tests import app, client

LICENSE_VARIABLE = "MOSEKLM_LICENSE_FILE"


def partial_reload(client, prop: str, **kwargs):
    """Request a single lazy prop, as the dashboard does."""

    headers = {
        "X-Inertia": "true",
        "X-Inertia-Partial-Component": "Dashboard",
        "X-Inertia-Partial-Data": prop,
    }

    return client.post("/", headers=headers, **kwargs)


def monomials_payload(terms, dimensions) -> dict:
    return {"monomials": {"terms": terms, "dimensions": dimensions}}


def parse_upload(app, filename: str, contents: bytes, key: str = "X0") -> dict:
    """Run the upload parser on a single uploaded file."""

    with app.test_request_context(
        "/", method="POST", data={key: (io.BytesIO(contents), filename)}
    ):
        return dashboard_controller._parse_uploaded_files({})


def calculation_form(**files) -> dict:
    """A small, well-formed Stability request."""

    return {
        "mode": "Stability",
        "model": "Linear",
        "timing": "Discrete-Time",
        "X0": json.dumps([[0.0, -0.0166, -0.1004, -0.0703], [0.0, 0.0441, -0.0397, -0.0825]]),
        "U0": json.dumps([[-0.0166, -0.0999, -0.0706, -0.0627], [0.0441, -0.0395, -0.0815, -0.0309]]),
        "X1": json.dumps([[-0.0166, -0.1004, -0.0703, -0.0619], [0.0441, -0.0397, -0.0825, -0.0316]]),
        "stateSpace": json.dumps({"x1": [-1, 1], "x2": [-1, 1]}),
        **files,
    }


def files_under(path) -> set:
    return {str(p.relative_to(path)) for p in path.rglob("*")}


# --- Monomials and Theta(x) ---


def test_valid_monomials_are_returned_unchanged(client):
    payload = monomials_payload(["x1", "x2", "x1*x2"], 2)

    response = partial_reload(client, "monomials", json=payload)

    assert response.status_code == 200
    assert response.json["props"]["monomials"] == payload["monomials"]


def test_theta_x_is_generated_from_the_monomials(client):
    payload = monomials_payload(["x1", "x2", "x1*x2"], 2)

    response = partial_reload(client, "theta_x", json=payload)

    assert response.status_code == 200
    assert response.json["props"]["theta_x"] == "1, 0\n 0, 1\n x2, 0"


@pytest.mark.parametrize("prop", ["monomials", "theta_x"])
def test_monomials_can_only_use_the_state_variables(client, prop):
    response = partial_reload(client, prop, json=monomials_payload(["x3"], 2))

    assert response.status_code == 200
    assert response.json["props"][prop] is False


@pytest.mark.parametrize("prop", ["monomials", "theta_x"])
def test_monomials_never_execute_code(client, tmp_path, prop):
    marker = tmp_path / "executed"
    payload = monomials_payload([f"__import__('pathlib').Path('{marker}').touch()"], 2)

    response = partial_reload(client, prop, json=payload)

    assert response.status_code == 200
    assert response.json["props"][prop] is False
    assert not marker.exists()


@pytest.mark.parametrize("prop", ["monomials", "theta_x"])
@pytest.mark.parametrize("dimensions", [0, -1, "2", True, 2.5, 10**9])
def test_dimensions_must_be_a_small_positive_integer(client, prop, dimensions):
    response = partial_reload(client, prop, json=monomials_payload(["x1"], dimensions))

    assert response.status_code == 200
    assert response.json["props"][prop] is False


@pytest.mark.parametrize("prop", ["monomials", "theta_x"])
@pytest.mark.parametrize(
    "body",
    [
        {"data": "not json"},
        {"json": ["x1"]},
        {"json": {"monomials": ["x1"]}},
        {"json": {"monomials": {"terms": [], "dimensions": 2}}},
        {"json": {"monomials": {"terms": "x1", "dimensions": 2}}},
    ],
)
def test_malformed_monomial_requests_are_rejected(client, prop, body):
    response = partial_reload(client, prop, **body)

    assert response.status_code == 200
    assert response.json["props"][prop] is False


# --- Uploaded datasets ---


def test_uploads_are_parsed_in_memory_whatever_their_filename(app, tmp_path, monkeypatch):
    def save(*args, **kwargs):
        raise AssertionError("uploads must not be written to disk")

    monkeypatch.setattr(FileStorage, "save", save)

    # Run from a nested directory, so a write to the traversal path would land inside tmp_path,
    # and put a file at every level the path could reach, to show none is overwritten or deleted.
    working_directory = tmp_path / "a" / "b" / "c"
    working_directory.mkdir(parents=True)
    monkeypatch.chdir(working_directory)
    targets = [directory / "evil.csv" for directory in (working_directory, *working_directory.parents)]
    targets = [target for target in targets if target.is_relative_to(tmp_path)]
    for target in targets:
        target.write_text("original")
    before = files_under(tmp_path)

    data = parse_upload(app, "../../../evil.csv", b"1,2,3\n4,5,6\n")

    assert data["X0"].tolist() == [["1", "2", "3"], ["4", "5", "6"]]
    assert files_under(tmp_path) == before
    assert all(target.read_text() == "original" for target in targets)


@pytest.mark.parametrize(
    "filename, contents, expected",
    [
        ("data.JSON", b"[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("data.txt", "﻿1\n2\n".encode("utf-8"), ["1", "2"]),
        ("data.csv", "﻿1,2\n3,4\n".encode("utf-8"), [["1", "2"], ["3", "4"]]),
    ],
)
def test_supported_uploads_are_parsed(app, filename, contents, expected):
    data = parse_upload(app, filename, contents)

    parsed = data["X0"]
    assert (parsed.tolist() if hasattr(parsed, "tolist") else parsed) == expected


@pytest.mark.parametrize(
    "filename, contents",
    [
        ("data.exe", b"1,2\n3,4\n"),
        ("data", b"1,2\n3,4\n"),
        ("data.csv", b"\xff\xfe1,2\n"),  # not UTF-8
        ("data.json", b"[[1, 2], [3, 4]"),  # malformed JSON
        ("data.csv", b"1,2,3\n4,5\n"),  # ragged rows
    ],
)
def test_unreadable_uploads_raise_an_upload_error(app, filename, contents):
    with pytest.raises(UploadError):
        parse_upload(app, filename, contents)


def test_only_dataset_fields_are_parsed(app):
    data = parse_upload(app, "data.csv", b"1,2\n", key="stateSpace")

    assert "stateSpace" not in data


def test_a_calculation_with_an_unreadable_upload_reports_the_error(client):
    form = calculation_form(X0=(io.BytesIO(b"\xff\xfe\x00"), "X0.csv"))

    response = partial_reload(client, "result", data=form)

    assert response.status_code == 200
    result = response.json["props"]["result"]
    assert result["error"] == "Unable to parse uploaded file(s)."
    assert "X0" in result["description"]


# --- MOSEK licence ---


class FakeEnvironment:
    """Stands in for mosek.Env, recording the licence it was given and whether it was checked in."""

    def __init__(self, licensefile=None):
        self.licensefile = licensefile
        self.checked_in = False

    def checkinall(self):
        self.checked_in = True


@pytest.fixture
def no_picos_environment(monkeypatch):
    """Start without a MOSEK environment in PICOS, and restore whatever was there afterwards."""

    monkeypatch.delattr(MOSEKSolver, "mosekEnvironment", raising=False)


@pytest.fixture
def fake_environments(monkeypatch, no_picos_environment) -> list:
    """Replace mosek.Env with FakeEnvironment, and collect every environment created."""

    created = []

    def create(licensefile=None):
        created.append(FakeEnvironment(licensefile))
        return created[-1]

    monkeypatch.setattr(dashboard_controller.mosek, "Env", create)
    return created


def licence_file(contents: bytes = b"licence contents") -> FileStorage:
    return FileStorage(stream=io.BytesIO(contents), filename="mosek.lic")


def test_the_uploaded_licence_licenses_a_new_environment(fake_environments, monkeypatch):
    monkeypatch.delenv(LICENSE_VARIABLE, raising=False)

    with _mosek_license(licence_file()):
        [environment] = fake_environments
        path = environment.licensefile

        assert MOSEKSolver._get_environment() is environment
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        with open(path, "rb") as file:
            assert file.read() == b"licence contents"

    assert environment.checked_in
    assert "mosekEnvironment" not in vars(MOSEKSolver)
    assert not os.path.exists(path)
    assert LICENSE_VARIABLE not in os.environ  # the process environment is left alone


def test_without_a_licence_mosek_searches_as_usual(fake_environments, monkeypatch):
    monkeypatch.setenv(LICENSE_VARIABLE, "/opt/mosek/mosek.lic")

    for file in (None, FileStorage(stream=io.BytesIO(b""), filename="")):
        with _mosek_license(file):
            assert MOSEKSolver._get_environment() is fake_environments[-1]
            assert fake_environments[-1].licensefile is None

        assert os.environ[LICENSE_VARIABLE] == "/opt/mosek/mosek.lic"

    assert len(fake_environments) == 2
    assert all(environment.checked_in for environment in fake_environments)


def test_the_licence_is_removed_even_if_the_calculation_fails(fake_environments):
    with pytest.raises(RuntimeError):
        with _mosek_license(licence_file()):
            raise RuntimeError("solver failed")

    [environment] = fake_environments
    assert environment.checked_in
    assert "mosekEnvironment" not in vars(MOSEKSolver)
    assert not os.path.exists(environment.licensefile)


def test_picos_solves_with_a_new_mosek_environment_each_time(no_picos_environment):
    """Uses the real MOSEK and PICOS, so it fails if PICOS stops using the installed environment."""

    environments = []
    for _ in range(2):
        with _mosek_environment(None) as environment:
            assert isinstance(environment, mosek.Env)
            assert MOSEKSolver._get_environment() is environment
            environments.append(environment)

        assert "mosekEnvironment" not in vars(MOSEKSolver)

    assert environments[0] is not environments[1]


def test_every_calculation_uses_its_own_licence(client, tmp_path, monkeypatch, no_picos_environment):
    """
    Three calculations in one process with the real solver: two with different (invalid) uploaded
    licences and one without. MOSEK names the licence file it tried in its error message, which
    shows that a later upload is not ignored and that no licence carries over to the next user.
    """

    monkeypatch.delenv(LICENSE_VARIABLE, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    licence_paths = []
    mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        descriptor, path = mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == "mosek-":
            licence_paths.append(path)
        return descriptor, path

    monkeypatch.setattr(tempfile, "mkstemp", recording_mkstemp)

    descriptions = []
    for licence in (b"invalid licence A", b"invalid licence B", None):
        files = {} if licence is None else {"mosek_lic": (io.BytesIO(licence), "mosek.lic")}
        response = partial_reload(client, "result", data=calculation_form(**files))

        assert response.status_code == 200
        descriptions.append(response.json["props"]["result"].get("description", ""))

    first, second = licence_paths
    assert first in descriptions[0]
    assert second in descriptions[1] and first not in descriptions[1]
    assert first not in descriptions[2] and second not in descriptions[2]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fails", [False, True])
def test_a_calculation_leaves_no_licence_behind(client, tmp_path, monkeypatch, fake_environments, fails):
    """The model is replaced by a stub that reads the licence its solver would be given."""

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = {}

    class Model:
        def __init__(self, data):
            pass

        def calculate(self):
            seen["environment"] = MOSEKSolver._get_environment()
            with open(seen["environment"].licensefile, "rb") as file:
                seen["licence"] = file.read()

            if fails:
                raise RuntimeError("solver failed")

            return {"function": {}, "controller": {}}

    monkeypatch.setattr(dashboard_controller, "Stability", Model)
    form = calculation_form(mosek_lic=(io.BytesIO(b"licence contents"), "mosek.lic"))

    response = partial_reload(client, "result", data=form)

    assert response.status_code == 200
    assert seen["licence"] == b"licence contents"
    assert seen["environment"] is fake_environments[0]
    assert seen["environment"].checked_in
    assert os.path.dirname(seen["environment"].licensefile) == str(tmp_path)
    assert ("error" in response.json["props"]["result"]) is fails
    assert list(tmp_path.iterdir()) == []
    assert "mosekEnvironment" not in vars(MOSEKSolver)


# --- Request size ---


def test_oversized_requests_are_rejected(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    form = calculation_form(X0=(io.BytesIO(b"0," * 2048), "X0.csv"))  # a 4 KB upload

    response = partial_reload(client, "result", data=form)

    assert response.status_code == 413


def test_requests_are_limited_to_16_mib_by_default(monkeypatch):
    monkeypatch.delenv("MAX_CONTENT_LENGTH", raising=False)

    assert create_app().config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024


def test_the_request_limit_can_be_configured(monkeypatch):
    monkeypatch.setenv("MAX_CONTENT_LENGTH", "2048")

    assert create_app().config["MAX_CONTENT_LENGTH"] == 2048
