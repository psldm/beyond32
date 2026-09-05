"""Tests of beyond32.molien: Molien series of I on harmonic polynomials and the S^3/2I
dictionary (Section 4, Eqs. 15-16 of the paper).  Reference values are pinned literally."""
import sympy as sp

from beyond32 import harmonics as h
from beyond32 import molien as mo
from beyond32.groups import CHAR_I

# m_A(l), l = 0..30: A line of Table 3 continued, = coefficients of (1+t^15)/((1-t^6)(1-t^10))
M_A_REF = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 2]

# Eq. (16): (k, mult) of the Poincare space S^3/2I
SPECTRUM_REF = [(0, 1), (12, 13), (20, 21), (24, 25), (30, 31), (32, 33), (36, 37), (40, 41),
                (42, 43), (44, 45), (48, 49), (50, 51)]

# Table 3, l = 0..6 (character route), used for the twisted sectors
TABLE3 = {0: {"A": 1}, 1: {"T1": 1}, 2: {"H": 1}, 3: {"T2": 1, "G": 1}, 4: {"G": 1, "H": 1},
          5: {"T1": 1, "T2": 1, "H": 1}, 6: {"A": 1, "T1": 1, "G": 1, "H": 1}}


def test_m_A_table_literal():
    assert mo.m_A_table(30) == M_A_REF
    assert all(isinstance(m, int) for m in mo.m_A_table(30))
    assert mo.m_A(0) == 1 and mo.m_A(6) == 1 and mo.m_A(15) == 1 and mo.m_A(30) == 2


def test_m_A_equals_molien_coefficients():
    assert mo.molien_series_coefficients(30) == M_A_REF
    assert mo.check_molien(30)
    # independent route (as in the legacy script): sympy series of the closed form
    ser = sp.series(mo.molien_series(), mo.t, 0, 31).removeO()
    assert [int(ser.coeff(mo.t, l)) for l in range(31)] == M_A_REF
    # combinatorial meaning: #{6a + 10b = l} + #{6a + 10b = l - 15}
    def count(n):
        return sum(1 for a in range(n // 6 + 1) for b in range(n // 10 + 1) if 6 * a + 10 * b == n) if n >= 0 else 0
    assert [count(l) + count(l - 15) for l in range(31)] == M_A_REF
    assert mo.molien_series_string() == "(1 + t^15)/((1 - t^6)(1 - t^10))"
    assert mo.MOLIEN_PRIMARY_DEGREES == (6, 10) and mo.MOLIEN_SECONDARY_DEGREE == 15


def test_molien_closed_form_from_the_rotation_matrices():
    # Molien's theorem on the 60 matrices reproduces the closed form of Eq. (15) exactly
    assert mo.check_molien_closed_form()
    assert sp.cancel(mo.molien_series_from_group() - mo.molien_series()) == 0


def test_nontrivial_degrees():
    assert mo.nontrivial_degrees(24) == [6, 10, 12, 15, 16, 18, 20, 21, 22, 24]
    assert mo.nontrivial_degrees(30) == [l for l in range(1, 31) if M_A_REF[l] > 0]


def test_poincare_spectrum_literal():
    spec = mo.poincare_spectrum(60)
    assert spec[:12] == SPECTRUM_REF
    assert (60, 122) in spec
    assert spec[-1] == (60, 122)
    assert mo.first_k_with_m_A(2) == 60
    # no modes below k = 12 except the constant, all k even, mult = (k+1) m_A(k/2)
    assert [k for k, _ in spec if 0 < k < 12] == []
    for k, m in spec:
        assert k % 2 == 0
        assert m == (k + 1) * M_A_REF[k // 2]
    assert mo.poincare_multiplicity(13) == 0 and mo.poincare_multiplicity(12) == 13
    assert mo.poincare_multiplicity(60) == 122


def test_m_A_matches_projector_branching():
    # projector route of harmonics.py (Table 3) agrees with the character formula
    for l in range(0, 7):
        assert mo.m_A(l) == h.branching(l).get("A", 0), l


def test_m_Gamma_matches_table3_and_twisted_spectrum():
    for l, dec in TABLE3.items():
        for irr in CHAR_I.irreps:
            assert mo.m_Gamma(l, irr) == dec.get(irr, 0), (l, irr)
        assert mo.m_Gamma(l, "A") == mo.m_A(l)
        assert sum(mo.m_Gamma(l, irr) * CHAR_I.dim(irr) for irr in CHAR_I.irreps) == 2 * l + 1
    # twisted sectors: T1 appears at l = 1, 5, 6; H at l = 2, 4, 5, 6 (Table 3)
    assert mo.twisted_spectrum("T1", 12) == [(2, 3), (10, 11), (12, 13)]
    assert mo.twisted_spectrum("H", 12) == [(4, 5), (8, 9), (10, 11), (12, 13)]
    assert mo.twisted_spectrum("A", 60) == mo.poincare_spectrum(60)


def test_paper_dictionary_and_format_are_plain_data():
    d = mo.paper_dictionary()
    assert d["m_A"] == M_A_REF
    assert d["molien_coefficients"] == M_A_REF
    assert d["molien_series"] == "(1 + t^15)/((1 - t^6)(1 - t^10))"
    assert d["nontrivial_l"][:10] == [6, 10, 12, 15, 16, 18, 20, 21, 22, 24]
    assert d["spectrum"][:12] == SPECTRUM_REF and d["spectrum"][-1] == (60, 122)
    assert d["first_k_with_m_A_2"] == 60
    assert d["checks"] == {"character_formula_vs_series": True, "closed_form_from_group": True}
    f = mo.format_molien()
    assert f["columns"] == ["l", "m_A(l)", "k = 2l", "mult(k)"]
    assert f["rows"][:3] == [[0, 1, 0, 1], [6, 1, 12, 13], [10, 1, 20, 21]]
    assert f["rows"][-1] == [30, 2, 60, 122]
    assert f["m_A_row"] == M_A_REF
    assert f["spectrum"].startswith("(0, 1), (12, 13), (20, 21), (24, 25)")
    assert f["nontrivial_l"].startswith("6, 10, 12, 15, 16, 18, 20, 21, 22, 24")
    for row in f["rows"]:
        assert all(isinstance(v, int) for v in row)
    assert isinstance(f["series"], str) and isinstance(f["first_k_with_m_A_2"], int)
