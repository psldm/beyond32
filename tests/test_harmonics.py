import sympy as sp
from sympy import Rational, sqrt

from beyond32._exact import PHI, canon, to_matrix
from beyond32 import harmonics as h
from beyond32.groups import CHAR_I, icosahedral_group

X, Y, Z = h.X, h.Y, h.Z
phi = PHI


def test_rep_matrices_are_a_representation():
    G = icosahedral_group()
    reps = h.rep_matrices(2)
    for i in range(0, 60, 13):
        for j in range(0, 60, 17):
            assert (reps[i] * reps[j]).to_list() == reps[G.mult[i][j]].to_list()
    # column j of rho_l(R) is the coefficient vector of (rho(R) m_j)(r) = m_j(R^T r)
    for l in (1, 2):
        M = to_matrix(h.rep_matrices(l)[7])
        for j, m in enumerate(h.monomials(l)):
            img = h.poly_from_vector(list(M[:, j]), l)
            assert sp.expand(img - h.act(G.rotations[7].matrix, m)) == 0


def test_harmonic_subspace_dimension_and_action():
    for l in range(0, 7):
        H = h.harmonic_subspace(l)
        assert H.shape == (h.n_monomials(l), 2 * l + 1)
    G = icosahedral_group()
    for f in h.isotypic_basis(3, "G"):
        assert h.is_harmonic(f)
        assert h.is_harmonic(h.act(G.rotations[7].matrix, f))


def test_branching_table():
    expected = {0: {"A": 1}, 1: {"T1": 1}, 2: {"H": 1}, 3: {"T2": 1, "G": 1}, 4: {"G": 1, "H": 1},
                5: {"T1": 1, "T2": 1, "H": 1}, 6: {"A": 1, "T1": 1, "G": 1, "H": 1}}
    for l, dec in expected.items():
        assert h.branching(l) == dec, l
        assert h.branching_by_characters(l) == dec, l
        assert sum(m * CHAR_I.dim(k) for k, m in dec.items()) == 2 * l + 1


def test_projectors_are_idempotent_and_complete():
    for l in (2, 3, 4):
        Ps = [h.projector(l, irr) for irr in CHAR_I.irreps]
        for P in Ps:
            assert (P * P).to_list() == P.to_list()
        tot = Ps[0]
        for P in Ps[1:]:
            tot = tot + P
        assert to_matrix(tot) == sp.eye(h.n_monomials(l))


def test_paper_basis_functions_literal():
    B = h.paper_basis_functions()
    phi_inv = phi - 1
    assert B[(1, "T1")] == {"x": X, "y": Y, "z": Z}
    H2 = B[(2, "H")]
    assert (H2["xy"], H2["yz"], H2["zx"]) == (X * Y, Y * Z, Z * X)
    assert sp.expand(H2["x^2-y^2"] - (X**2 - Y**2)) == 0
    assert sp.expand(H2["2z^2-x^2-y^2"] - (2 * Z**2 - X**2 - Y**2)) == 0
    fx = X**3 + 3 * phi_inv * X * Y**2 - 3 * phi * X * Z**2
    assert sp.expand(B[(3, "T2")]["f_x"] - fx) == 0
    assert sp.expand(B[(3, "T2")]["f_y"] - h.cyclic(fx)) == 0
    gx = X**3 - phi**2 * X * Y**2 - (2 - phi) * X * Z**2          # phi^-2 = 2 - phi
    assert sp.expand(B[(3, "G")]["g_x"] - gx) == 0
    assert sp.expand(B[(3, "G")]["xyz"] - X * Y * Z) == 0
    u0 = X**4 + Y**4 + Z**4 - 3 * (X**2 * Y**2 + Y**2 * Z**2 + Z**2 * X**2)
    ux = X**2 * Y * Z + (phi - 1) / 3 * Y**3 * Z - phi / 3 * Y * Z**3
    assert sp.expand(B[(4, "G")]["u_0"] - u0) == 0
    assert sp.expand(B[(4, "G")]["u_x"] - ux) == 0
    vx = X**4 - 3 * X**2 * Y**2 - 3 * X**2 * Z**2 - Y**4 / 2 + 6 * Y**2 * Z**2 - Z**4 / 2
    wx = X**2 * Y * Z + (1 - 7 * phi) / 15 * Y**3 * Z + (7 * phi - 6) / 15 * Y * Z**3
    assert sp.expand(B[(4, "H")]["v_x"] - vx) == 0
    assert sp.expand(B[(4, "H")]["w_x"] - wx) == 0
    assert sp.expand(B[(4, "H")]["v_x"] + B[(4, "H")]["v_y"] + B[(4, "H")]["v_z"]) == 0
    # every listed function is harmonic and lies in its isotypic component
    for (l, irr), funcs in B.items():
        for name, f in funcs.items():
            assert h.is_harmonic(f), (l, irr, name)
            assert sp.expand(h.project(l, irr, f) - f) == 0, (l, irr, name)


def test_invariant_P6_and_hexad():
    P6 = h.invariant_P6()
    lit = (X**6 + Y**6 + Z**6 + (3 - 21 * phi) * (X**4 * Y**2 + Y**4 * Z**2 + Z**4 * X**2)
           + (21 * phi - 18) * (X**4 * Z**2 + Y**4 * X**2 + Z**4 * Y**2) + 90 * X**2 * Y**2 * Z**2)
    assert sp.expand(P6 - lit) == 0
    assert h.is_harmonic(P6)
    G = icosahedral_group()
    for r in G.rotations[::9]:
        assert sp.expand(h.act(r.matrix, P6) - P6) == 0
    hx = h.hexad_identity()
    assert hx["c"] == Rational(-2, 35)
    assert hx["e"] == Rational(6, 7)
    assert sp.expand(hx["hexad"] - (Rational(-2, 35) * P6 + Rational(6, 7) * h.R2**3)) == 0


def test_harmonic_components_and_sphere_inner():
    F = sp.expand((X**2 - Y**2) ** 2)
    comps = h.harmonic_components(F)
    assert set(comps) <= {4, 2, 0}
    assert sp.expand(sum(hh * h.R2 ** ((4 - l) // 2) for l, hh in comps.items()) - F) == 0
    for hh in comps.values():
        assert h.is_harmonic(hh)
    assert h.sphere_inner(X, X) == Rational(1, 3)
    assert h.sphere_inner(X * Y, X * Y) == Rational(1, 15)
    assert h.sphere_inner(X, Y) == 0
    assert h.mono_int(2, 2, 2) == Rational(1, 105)
