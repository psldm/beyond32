"""LaTeX emitter: symbol conversion and, end to end (slow), the fragments of all tables."""
import os

import pytest

from beyond32 import latex


def test_tex_irrep_names():
    assert latex.tex_irrep("T1") == "T_1"
    assert latex.tex_irrep("T1u") == "T_{1u}"
    assert latex.tex_irrep("Ag") == "A_g"
    assert latex.tex_irrep("4s") == "4_s"
    assert latex.tex_irrep("2'") == "2'"
    assert latex.tex_irrep("1E") == "{}^{1}E"
    assert latex.tex_irrep("chi3") == r"\chi_{3}"
    assert latex.tex_irrep("E1") == "E_1"
    assert latex.tex_irrep("A") == "A"


def test_tex_decomposition_and_numbers():
    assert latex.tex_decomposition({"A": 1, "H": 2}) == r"A \oplus 2\,H"
    assert latex.tex_num("1 - varphi") == r"1 - \varphi"
    assert latex.tex_num("-2/35") == r"\tfrac{-2}{35}"
    assert latex.tex_poly("x**3 - 3*varphi*x*z**2") == r"x^{3} - 3 \varphi x z^{2}"


@pytest.mark.slow
def test_write_all_fragments(tmp_path):
    from beyond32.results import collect

    res = collect(fast=True)
    paths = latex.write_tables(res, str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == sorted(n + ".tex" for n in latex.TABLE_FILES)
    for p in paths:
        txt = open(p).read()
        assert r"\begin{tabular}" in txt or r"\begin{align*}" in txt
    states = open(tmp_path / "tab_states.tex").read()
    assert r"\tfrac{9}{5}" in states and r"\tfrac{6}{5}" in states and "1.5255" in states
    hc = open(tmp_path / "tab_Hcand.tex").read()
    assert "0.7619" in hc and "-0.1825" in hc and "-0.6186" in hc
    ch = open(tmp_path / "tab_charI.tex").read()
    assert r"\varphi" in ch
