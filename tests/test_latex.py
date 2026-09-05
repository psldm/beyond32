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
    assert latex.tex_num("-2/35") == r"-\tfrac{2}{35}"
    assert latex.tex_phi_coeff("1/15", "-7/15") == r"\tfrac{1 - 7\varphi}{15}"
    assert latex.tex_phi_coeff("0", "-3") == r"-3\varphi"
    assert latex.tex_phi_coeff("-3", "3") == r"3\varphi - 3"
    assert latex.tex_phi_coeff("-1/3", "1/3") == r"\tfrac{\varphi - 1}{3}"
    assert latex.tex_phi_coeff("0", "-1/3") == r"-\tfrac{\varphi}{3}"
    f_x = [[[3, 0, 0], "1", "0"], [[1, 2, 0], "-3", "3"], [[1, 0, 2], "0", "-3"]]
    assert latex.tex_poly_terms(f_x) == r"x^{3} + (3\varphi - 3)\,xy^{2} - 3\varphi\,xz^{2}"
    assert latex.tex_decomposition({"4s": 1, "6": 2}) == r"4_s \oplus 2\cdot 6"


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
