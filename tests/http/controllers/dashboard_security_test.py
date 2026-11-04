import io
import json
import os
import stat
import tempfile

import pytest
from werkzeug.datastructures import FileStorage

from app import create_app
from app.http.controllers import dashboard_controller
from app.http.controllers.dashboard_controller import UploadError, _mosek_license
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


def licence_file(contents: bytes = b"licence contents") -> FileStorage:
    return FileStorage(stream=io.BytesIO(contents), filename="mosek.lic")


def test_the_licence_is_only_available_during_the_calculation(monkeypatch):
    monkeypatch.delenv(LICENSE_VARIABLE, raising=False)

    with _mosek_license(licence_file()):
        path = os.environ[LICENSE_VARIABLE]

        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        with open(path, "rb") as file:
            assert file.read() == b"licence contents"

    assert LICENSE_VARIABLE not in os.environ
    assert not os.path.exists(path)


def test_a_previous_licence_setting_is_restored(monkeypatch):
    monkeypatch.setenv(LICENSE_VARIABLE, "/opt/mosek/mosek.lic")

    with _mosek_license(licence_file()):
        path = os.environ[LICENSE_VARIABLE]
        assert path != "/opt/mosek/mosek.lic"

    assert os.environ[LICENSE_VARIABLE] == "/opt/mosek/mosek.lic"
    assert not os.path.exists(path)


@pytest.mark.parametrize("previous", [None, "/opt/mosek/mosek.lic"])
def test_the_licence_is_removed_even_if_the_calculation_fails(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv(LICENSE_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(LICENSE_VARIABLE, previous)

    with pytest.raises(RuntimeError):
        with _mosek_license(licence_file()):
            path = os.environ[LICENSE_VARIABLE]
            raise RuntimeError("solver failed")

    assert os.environ.get(LICENSE_VARIABLE) == previous
    assert not os.path.exists(path)


def test_nothing_changes_without_a_licence(monkeypatch):
    monkeypatch.setenv(LICENSE_VARIABLE, "/opt/mosek/mosek.lic")

    with _mosek_license(None):
        assert os.environ[LICENSE_VARIABLE] == "/opt/mosek/mosek.lic"

    with _mosek_license(FileStorage(stream=io.BytesIO(b""), filename="")):
        assert os.environ[LICENSE_VARIABLE] == "/opt/mosek/mosek.lic"


@pytest.mark.parametrize("fails", [False, True])
def test_a_calculation_leaves_no_licence_behind(client, tmp_path, monkeypatch, fails):
    """
    The model is replaced by a stub, because PICOS shares one MOSEK environment per process and
    a fake licence would otherwise stay configured for the rest of the test session.
    """

    monkeypatch.delenv(LICENSE_VARIABLE, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = {}

    class Model:
        def __init__(self, data):
            pass

        def calculate(self):
            seen["path"] = os.environ[LICENSE_VARIABLE]
            with open(seen["path"], "rb") as file:
                seen["licence"] = file.read()

            if fails:
                raise RuntimeError("solver failed")

            return {"function": {}, "controller": {}}

    monkeypatch.setattr(dashboard_controller, "Stability", Model)
    form = calculation_form(mosek_lic=(io.BytesIO(b"licence contents"), "mosek.lic"))

    response = partial_reload(client, "result", data=form)

    assert response.status_code == 200
    assert seen["licence"] == b"licence contents"
    assert os.path.dirname(seen["path"]) == str(tmp_path)
    assert ("error" in response.json["props"]["result"]) is fails
    assert list(tmp_path.iterdir()) == []
    assert LICENSE_VARIABLE not in os.environ


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
