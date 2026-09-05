"""results.json assembly and the command-line interface."""
import json

import numpy as np
import pytest
import sympy as sp

from beyond32 import results


def test_jsonable_converts_everything():
    out = results.jsonable({"a": sp.sqrt(5) / 2, "b": np.array([1.0, 2.0]), "c": 1 + 2j,
                            "d": (sp.Integer(3), sp.Rational(1, 2)), "e": np.int64(4)})
    assert out == {"a": "sqrt(5)/2", "b": [1.0, 2.0], "c": {"re": 1.0, "im": 2.0}, "d": [3, "1/2"], "e": 4}
    json.dumps(out)


def test_rational_if_close():
    assert results.rational_if_close(1.8) == "9/5"
    assert results.rational_if_close(3374 / 2145) == "3374/2145"
    assert results.rational_if_close(1.5254512929) is None


def test_cli_all_creates_the_output_directory(tmp_path, monkeypatch):
    import beyond32.latex
    import beyond32.results
    from beyond32 import __version__
    from beyond32.cli import main

    stub = {"package": {"version": __version__, "runtime_seconds": {}}}
    monkeypatch.setattr(beyond32.results, "collect", lambda fast=False: stub)
    monkeypatch.setattr(beyond32.latex, "write_tables", lambda res, outdir: [])
    out = tmp_path / "does" / "not" / "exist"
    assert main(["all", "--fast", "--out", str(out)]) == 0
    assert json.load(open(out / "results.json")) == stub


def test_cli_tables_rejects_a_results_file_from_an_older_schema(tmp_path, capsys):
    from beyond32.cli import main, schema_problem

    stale = {"package": {"version": "0.9.0"}, "groups": {"I": {}, "2I": {}},
             "harmonics": {"branching_by_characters": {}, "hexad": {}}}       # no basis_functions_terms
    assert "harmonics/basis_functions_terms" in schema_problem(stale)
    p = tmp_path / "old.json"
    p.write_text(json.dumps(stale))
    out = tmp_path / "tables"
    assert main(["tables", "--results", str(p), "--out", str(out)]) == 1
    assert "regenerate" in capsys.readouterr().out
    assert not out.exists()                    # nothing half-written


def test_write_tables_renders_everything_before_writing(tmp_path):
    from beyond32 import latex

    with pytest.raises(KeyError):
        latex.write_tables({"groups": {"I": {"character_table": {}}}}, str(tmp_path / "t"))
    assert not (tmp_path / "t").exists()


@pytest.mark.slow
def test_cli_check_reproduces_the_paper(capsys):
    from beyond32.cli import main

    assert main(["check", "--fast"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "all reference values reproduced" in out
