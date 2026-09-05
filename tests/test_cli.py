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


@pytest.mark.slow
def test_cli_check_reproduces_the_paper(capsys):
    from beyond32.cli import main

    assert main(["check", "--fast"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "all reference values reproduced" in out
