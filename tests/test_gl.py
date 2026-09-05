"""Ginzburg-Landau theory: Tables 6-8 and Eqs. (17)-(23) of the paper, pinned literally."""
import numpy as np
import pytest
import sympy as sp
from sympy import Rational, sqrt

from beyond32._exact import SQRT5, canon
from beyond32 import gl
from beyond32.groups import CHAR_I, icosahedral_group
from beyond32._exact import to_matrix
from beyond32.harmonics import X, Y, Z, coeff_dict, exponents, is_harmonic, monomials, projector


# --------------------------------------------------------------------------- irrep matrices
def test_sphere_norms_of_basis_functions():
    assert gl.channel("T1").norms2 == (Rational(1, 3),) * 3
    assert gl.channel("H").norms2 == (Rational(1, 15),) * 5
    assert gl.channel("T2").norms2 == (Rational(4, 7),) * 3
    assert gl.channel("G").norms2 == (Rational(4, 21), Rational(4, 21), Rational(4, 21), Rational(1, 105))


@pytest.mark.parametrize("name", gl.CHANNELS)
def test_irrep_matrices_orthogonal_character_homomorphism(name):
    ch = gl.channel(name)
    G = icosahedral_group()
    d = ch.dim
    for i, D in enumerate(ch.matrices):
        assert (D.T * D).applyfunc(sp.expand) == sp.eye(d)
        assert sp.expand(D.trace() - G.character(name, i)) == 0
    for a in range(0, 60, 7):
        for b in range(0, 60, 11):
            assert (ch.matrices[a] * ch.matrices[b] - ch.matrices[G.mult[a][b]]).applyfunc(sp.expand) == sp.zeros(d)


# --------------------------------------------------------------------------- Table 6
def test_sym2_decompositions_and_counts():
    expected = {"T1": {"A": 1, "H": 1}, "T2": {"A": 1, "H": 1}, "G": {"A": 1, "G": 1, "H": 1},
                "H": {"A": 1, "G": 1, "H": 2}}
    for name, dec in expected.items():
        assert gl.sym2_by_characters(name) == dec
        assert gl.sym2_projector_ranks(name) == dec
    assert [gl.hermitian_form_count(n) for n in gl.CHANNELS] == [2, 2, 3, 6]
    assert [gl.tr_even_quartic_count(n) for n in gl.CHANNELS] == [2, 2, 3, 5]


# --------------------------------------------------------------------------- Eqs. (18)-(22)
def test_invariant_relations():
    r = gl.relations("T1")
    assert r["N0"] == {"I1": 0, "I2": 1}
    assert r["N2"] == {"I1": Rational(6, 5), "I2": Rational(-2, 5)}
    assert r["quartic"] == {"I1": Rational(6, 5), "I2": Rational(3, 5)}          # (3/5)(2 I1 + I2)
    r = gl.relations("T2")
    assert r["N0"] == {"I1": 0, "I2": 1}
    assert r["N2"] == {"I1": Rational(8, 15), "I2": Rational(-8, 45)}
    assert r["quartic"] == {"I1": Rational(3374, 2145), "I2": Rational(1687, 2145)}  # (1687/2145)(2 I1 + I2)
    r = gl.relations("G")
    assert r["N0"] == {"I1": 0, "I2": 1, "N2": 0}
    assert r["N4"] == {"I1": Rational(112, 121), "I2": Rational(-28, 121), "N2": Rational(-135, 121)}
    assert r["N6"] == {"I1": Rational(700, 1573), "I2": Rational(100, 1573), "N2": Rational(875, 1573)}
    assert r["quartic"] == {"I1": Rational(196, 143), "I2": Rational(119, 143), "N2": Rational(63, 143)}
    r = gl.relations("H")
    assert r["N0"] == {"I1": 0, "I2": 1}
    assert r["N2+N4"] == {"I1": Rational(10, 7), "I2": Rational(-2, 7)}
    assert r["quartic"] == {"I1": Rational(10, 7), "I2": Rational(5, 7)}          # (5/7)(2 I1 + I2)
    assert sorted(gl.quartic_invariants("T1").components) == [0, 2]
    assert sorted(gl.quartic_invariants("H").components) == [0, 2, 4]
    assert sorted(gl.quartic_invariants("G").components) == [0, 2, 4, 6]


def test_components_are_harmonic_and_sum_to_delta_squared():
    for name in gl.CHANNELS:
        q = gl.quartic_invariants(name)
        total = 0
        for L, C in q.components.items():
            h = q.component_expr(L)
            assert is_harmonic(h), (name, L)
            total += h * (X**2 + Y**2 + Z**2) ** ((2 * q.dim and 2 * gl.CHANNEL_L[name] - L) // 2)
        assert sp.expand(total - sp.expand(q.delta**2)) == 0, name


# --------------------------------------------------------------------------- H channel
def test_h_channel_intertwiner_and_cross_terms():
    H = gl.h_channel()
    ref = sp.expand((Rational(-7, 20) + 47 * SQRT5 / 30) ** 2)
    assert sp.expand(H.lam - ref) == 0
    assert H.six_independent
    assert (H.N4G + H.N4H - H.inv.N[4]).is_zero()
    h4G, h4H = H.component_expr("h4G"), H.component_expr("h4H")
    assert sp.expand(h4G + h4H - H.inv.component_expr(4)) == 0
    # the two parts lie in the G and H isotypic components of the l = 4 harmonics
    q = H.inv
    one = {**{e: 1 for e in q.eta}, **{e: 1 for e in q.etab}}
    for expr, irr in ((h4G, "G"), (h4H, "H")):
        f = sp.expand(expr.subs(one))
        assert is_harmonic(f)
        # P_irr f = f (coefficients lie in Q(sqrt3, sqrt5), so apply the projector as a sympy matrix)
        v = sp.Matrix([coeff_dict(f).get(e, 0) for e in exponents(4)])
        Pv = to_matrix(projector(4, irr)) * v
        assert sp.expand(sum(c * m for c, m in zip(Pv, monomials(4))) - f) == 0
    assert is_harmonic(sp.expand(H.component_expr("Jh4H").subs(one)))
    # Im C is odd under time reversal (eta <-> eta*), Re C even
    assert (H.CR.conj() - H.CR).is_zero()
    assert (H.CIi.conj() + H.CIi).is_zero()


# --------------------------------------------------------------------------- weak coupling
def test_weak_coupling_ratios_of_real_and_null_cone_states():
    def null(d):
        e = np.zeros(d, dtype=complex)
        e[0], e[1] = 1, 1j
        return e / np.sqrt(2)

    def real(d):
        e = np.zeros(d)
        e[0] = 1
        return e

    assert abs(gl.weak_coupling_ratio("T1", null(3)) - 6 / 5) < 1e-12
    assert abs(gl.weak_coupling_ratio("T1", real(3)) - 9 / 5) < 1e-12
    assert abs(gl.weak_coupling_ratio("T2", null(3)) - 3374 / 2145) < 1e-12
    assert abs(gl.weak_coupling_ratio("T2", real(3)) - 5061 / 2145) < 1e-12
    assert abs(gl.weak_coupling_ratio("H", null(5)) - 10 / 7) < 1e-12
    assert abs(gl.weak_coupling_ratio("H", real(5)) - 15 / 7) < 1e-12


def test_exact_weak_coupling_ratios_closed_forms():
    """Eq. (22) at the real state (1, 0, ...) and the null-cone state (1, i, 0, ...), in exact
    arithmetic (these are the 'real' / 'null_cone' entries of results.json)."""
    I = sp.I
    assert gl.weak_coupling_ratio_exact("T1", (1, 0, 0)) == Rational(9, 5)
    assert gl.weak_coupling_ratio_exact("T1", (1, I, 0)) == Rational(6, 5)
    assert gl.weak_coupling_ratio_exact("T2", (1, 0, 0)) == Rational(5061, 2145)
    assert gl.weak_coupling_ratio_exact("T2", (1, I, 0)) == Rational(3374, 2145)
    assert gl.weak_coupling_ratio_exact("H", (1, 0, 0, 0, 0)) == Rational(15, 7)
    assert gl.weak_coupling_ratio_exact("H", (1, I, 0, 0, 0)) == Rational(10, 7)
    # normalisation-independent, and the closed forms are c (2 I1 + I2) with c the I2 coefficient
    assert gl.weak_coupling_ratio_exact("T2", (Rational(1, 2), I / 2, 0)) == Rational(3374, 2145)
    for ch in ("T1", "T2", "H"):
        c = gl.relations(ch)["quartic"]
        assert c["I1"] == 2 * c["I2"]
    # the exact G states of Table 7: xyz (T A), g_x (D2), and the chiral (1, w, w^2, 0) (C3)
    assert gl.weak_coupling_ratio_exact("G", (0, 0, 0, 1)) == Rational(315, 143)      # 2.2028
    assert gl.weak_coupling_ratio_exact("G", (1, 0, 0, 0)) == Rational(1743, 715)     # 2.4378
    w = Rational(-1, 2) + sqrt(3) * I / 2
    assert gl.weak_coupling_ratio_exact("G", (1, w, sp.expand(w**2), 0)) == Rational(84, 55)   # 1.5273


def test_table7_ratios_from_exact_fixed_spaces():
    """The R_wc of the real-character states of Table 7, computed from the EXACT fixed spaces
    (null spaces over Q(sqrt3, sqrt5)) and the exact quartic forms; the chiral C5 state of G
    (complex character) is compared numerically with the recognised fraction 1148/715."""
    R = gl.exact_ratio_of_fixed_state
    assert R("T1", "D5", "A2") == Rational(9, 5)
    assert R("T1", "D3", "A2") == Rational(9, 5)
    assert R("T1", "D2", "B1") == Rational(9, 5)
    assert R("T2", "D5", "A2") == Rational(1687, 715)        # = 5061/2145
    assert R("T2", "D2", "B1") == Rational(1687, 715)
    assert R("G", "T", "A") == Rational(315, 143)            # xyz, 2.2028
    assert R("G", "D2", "B1") == Rational(1743, 715)         # g_z, 2.4378
    assert R("G", "D2", "B3") == Rational(1743, 715)         # g_x
    assert R("G", "D3", "A1") == Rational(350, 143)          # 2.4476
    assert R("G", "D3", "A2") == Rational(126, 55)           # 2.2909
    assert R("H", "D5", "A1") == Rational(15, 7)
    assert R("H", "D3", "A1") == Rational(15, 7)
    assert R("H", "D2", "B1") == Rational(15, 7)
    # complex characters are not handled exactly
    assert R("G", "C5", "chi1") is None and R("H", "T", "1E") is None
    # the fixed space of D3 A1 in G is one-dimensional and the exact vector is real
    (v,) = gl.fixed_space_exact("G", "D3", "A1")
    assert all(sp.im(c) == 0 for c in v) and len(v) == 4
    # chiral C5 states of G: recognised fraction 1148/715, agreeing to double precision
    rows = {(r.subgroup, r.character): r for r in gl.symmetry_fixed_states(["G"])}
    for m in ("chi1", "chi2", "chi3", "chi4"):
        assert abs(rows[("C5", m)].R - 1148 / 715) < 1e-12
    assert abs(rows[("C3", "chi1")].R - 84 / 55) < 1e-12


def test_h_isometry_scale_is_a_perfect_square():
    H = gl.h_channel()
    root = sp.sqrtdenest(sp.sqrt(H.lam))
    assert sp.expand(root - (Rational(-7, 20) + 47 * SQRT5 / 30)) == 0
    assert sp.expand(root**2 - H.lam) == 0


def test_number_of_terms_of_the_invariants_matches_legacy_gl_inv2():
    expected = {"T1": {0: 9, 2: 12}, "T2": {0: 9, 2: 12, 4: 12, 6: 12},
                "G": {0: 16, 2: 21, 4: 28, 6: 28}, "H": {0: 25, 2: 52, 4: 53}}
    for name, exp in expected.items():
        q = gl.quartic_invariants(name)
        assert {L: len(sp.Poly(q.expr(f"N{L}"), *q.gens).terms()) for L in q.N} == exp, name


def test_weak_coupling_minima():
    assert abs(gl.minimise_ratio("T1").value - 6 / 5) < 1e-6
    assert abs(gl.minimise_ratio("T2").value - 3374 / 2145) < 1e-6
    assert abs(gl.minimise_ratio("H").value - 10 / 7) < 1e-6
    m = gl.minimise_ratio("G")
    assert round(m.value, 5) == 1.52545
    assert 1e-3 < m.I2 < 3e-3


def test_g_ground_state():
    gs = gl.g_ground_state()
    assert round(gs["R"], 5) == 1.52545
    assert round(gs["null_cone_R"], 5) == 1.52721
    assert 1e-3 < gs["I2"] < 3e-3
    assert gs["stabiliser"] == {"E": 1, "C3": 2}
    assert gs["tr_stabiliser"] == {"C2": 3}


def test_g_stratum_kappa_phase():
    st = gl.g_stratum()
    assert round(st["kappa"], 3) == 1.752
    assert round(st["phi0_deg"], 1) == 92.6
    assert round(st["R"], 5) == 1.52545
    assert round(st["R_check"], 5) == 1.52545
    assert 1e-3 < st["I2_over_I1"] < 3e-3


def test_g_nodes():
    nd = gl.g_nodes()
    assert nd["n_nodes"] == 18
    assert nd["on_fivefold_axes"] == 12


# --------------------------------------------------------------------------- Table 7
def _fixed(rows, channel):
    return {(r.subgroup, r.character): r for r in rows if r.channel == channel}


def test_isotropy_table_symmetry_fixed_states():
    rows = gl.symmetry_fixed_states()
    assert all(r.dim_fixed == 1 for r in rows)
    t1 = _fixed(rows, "T1")
    assert set(t1) == {("D5", "A2"), ("D3", "A2"), ("D2", "B1"), ("D2", "B2"), ("D2", "B3"),
                       ("C5", "chi1"), ("C5", "chi4"), ("C3", "chi1"), ("C3", "chi2")}
    for k in [("D5", "A2"), ("D3", "A2"), ("D2", "B1"), ("D2", "B2"), ("D2", "B3")]:
        assert round(t1[k].R, 4) == 1.8 and t1[k].time_reversal
    for k in [("C5", "chi1"), ("C5", "chi4"), ("C3", "chi1"), ("C3", "chi2")]:
        assert round(t1[k].R, 4) == 1.2 and not t1[k].time_reversal
    t2 = _fixed(rows, "T2")
    assert set(t2) == {("D5", "A2"), ("D3", "A2"), ("D2", "B1"), ("D2", "B2"), ("D2", "B3"),
                       ("C5", "chi2"), ("C5", "chi3"), ("C3", "chi1"), ("C3", "chi2")}
    for k in [("D5", "A2"), ("D3", "A2"), ("D2", "B1")]:
        assert round(t2[k].R, 4) == round(5061 / 2145, 4) and t2[k].time_reversal
    for k in [("C5", "chi2"), ("C5", "chi3"), ("C3", "chi1"), ("C3", "chi2")]:
        assert round(t2[k].R, 4) == round(3374 / 2145, 4) and not t2[k].time_reversal
    g = _fixed(rows, "G")
    assert set(g) == {("T", "A"), ("D2", "B1"), ("D2", "B2"), ("D2", "B3"), ("D3", "A1"), ("D3", "A2"),
                      ("C5", "chi1"), ("C5", "chi2"), ("C5", "chi3"), ("C5", "chi4"), ("C3", "chi1"), ("C3", "chi2")}
    assert round(g[("T", "A")].R, 4) == 2.2028 and np.allclose(np.abs(g[("T", "A")].eta), [0, 0, 0, 1])
    for k in [("D2", "B1"), ("D2", "B2"), ("D2", "B3")]:
        assert round(g[k].R, 4) == 2.4378
    assert round(g[("D3", "A1")].R, 4) == 2.4476
    assert round(g[("D3", "A2")].R, 4) == 2.2909
    for k in [("C5", "chi1"), ("C5", "chi2"), ("C5", "chi3"), ("C5", "chi4")]:
        assert round(g[k].R, 4) == 1.6056 and not g[k].time_reversal
    for k in [("C3", "chi1"), ("C3", "chi2")]:
        assert round(g[k].R, 4) == 1.5273 and not g[k].time_reversal
    # every G state has min |Delta| = 0 (enforced nodes on the five-fold axes)
    assert all(r.min_gap < 1e-2 for r in rows if r.channel == "G")
    # G restricted to C5 has no trivial character: no (C5, chi0) fixed space at all
    assert ("C5", "chi0") not in {(r.subgroup, r.character) for r in gl.isotropy_table(["G"])}
    h = _fixed(rows, "H")
    assert set(h) == {("D5", "A1"), ("D3", "A1"), ("D2", "B1"), ("D2", "B2"), ("D2", "B3"),
                      ("C5", "chi1"), ("C5", "chi2"), ("C5", "chi3"), ("C5", "chi4"), ("T", "1E"), ("T", "2E")}
    for k in [("D5", "A1"), ("D3", "A1"), ("D2", "B1")]:
        assert round(h[k].R, 4) == round(15 / 7, 4) and h[k].time_reversal
    for k in [("C5", "chi1"), ("C5", "chi2"), ("C5", "chi3"), ("C5", "chi4"), ("T", "1E"), ("T", "2E")]:
        assert round(h[k].R, 4) == round(10 / 7, 4) and not h[k].time_reversal
    # the two cyclic chiralities are time-reversal partners not related by any element of I
    assert not h[("T", "1E")].tr_up_to_rotation and not h[("T", "2E")].tr_up_to_rotation
    assert h[("C5", "chi1")].tr_up_to_rotation      # chiral axial states are (via the perpendicular C2)


# --------------------------------------------------------------------------- Table 8
def test_h_candidates_table():
    rows = gl.h_candidates()
    expected = {"Y22 about C5": (0, 0.7619, 0.6667, 0), "Y22 about C3": (0, 0.5644, 0.8642, 0),
                "Y22 about C2": (0, 0.5952, 0.8333, 0), "Y21 about C5": (0.6122, 0.7619, 0.0544, -0.1825),
                "Y21 about C3": (0.6122, 0.1411, 0.6752, 0.1014), "Y21 about C2": (0.6122, 0.2381, 0.5782, 0.0570),
                "cyclic (T)": (0.8163, 0, 0.6122, 0.3423)}
    for name, (n2, n4g, n4h, rec) in expected.items():
        r = rows[name]
        assert (round(r["N2"], 4), round(r["N4G"], 4), round(r["N4H"], 4), round(r["ReC"], 4)) == (n2, n4g, n4h, rec), name
        assert abs(r["I2"]) < 1e-6
        assert abs(r["N2"] + r["N4"] - 10 / 7) < 1e-4
    assert round(rows["cyclic (T)"]["ImC"], 4) == -0.6186
    assert round(rows["cyclic (T), other chirality"]["ImC"], 4) == 0.6186


# --------------------------------------------------------------------------- slow cross-check
@pytest.mark.slow
@pytest.mark.parametrize("name", gl.CHANNELS)
def test_forms_agree_with_direct_symbolic_expansion(name):
    """The exact form matrices reproduce the legacy route (sympy expansion of |[Delta^2]_L|^2)."""
    q = gl.quartic_invariants(name)
    for L in q.components:
        direct = gl.sphere_norm2_complex_expr(q.component_expr(L), q.eta, q.etab)
        assert sp.expand(direct - q.expr(f"N{L}")) == 0, (name, L)
    direct = gl.sphere_norm2_complex_expr(sp.expand(q.delta**2), q.eta, q.etab)
    assert sp.expand(direct - q.expr("quartic")) == 0
