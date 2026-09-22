"""Tests for beyond32.d6d: the dodecagonal point group D6d = -12m2 of Appendix B (Table 10).

The reference values are pinned literally; the module must compute them from the two
generator matrices S12 = diag(1, 1, -1) R_z(30 deg) and C2'(15 deg).
"""
import json

import sympy as sp
from sympy import ImmutableMatrix, Matrix, Rational, pi, sqrt

from beyond32 import d6d

S3 = sqrt(3)
PHI = d6d.ANGLE
c, s = sp.cos, sp.sin
HALF = Rational(1, 2)


def S(k):
    return d6d.D6dElement("S", k % 12)


def P(k):
    return d6d.D6dElement("P", k % 12)


def X(M):
    return M.applyfunc(sp.expand)


U = sp.Symbol("u")           # u = e^{i phi}


def _laurent(x):
    """A trigonometric polynomial in phi (arguments a phi + b, b a rational multiple of pi) as
    an exact Laurent polynomial in u = e^{i phi} with coefficients in Q(sqrt3, i)."""
    def conv(e):
        arg = sp.expand(e.args[0])
        a = arg.coeff(PHI)
        b = sp.expand(arg - a * PHI)
        eb, ebi = sp.expand(sp.cos(b) + sp.I * sp.sin(b)), sp.expand(sp.cos(b) - sp.I * sp.sin(b))
        if isinstance(e, sp.cos):
            return (eb * U ** a + ebi * U ** (-a)) / 2
        return (eb * U ** a - ebi * U ** (-a)) / (2 * sp.I)
    return sp.expand(sp.sympify(x).replace(lambda e: isinstance(e, (sp.sin, sp.cos)), conv))


def _vanishes(x) -> bool:
    """Exact zero test for trigonometric polynomials in phi with algebraic coefficients."""
    return _laurent(x) == 0


def _transform_axial(e, d):
    """The axial-vector law d -> det(g) g d(g^-1 k) applied by substitution to a d-vector given
    as three expressions in phi (independent of the sector matrices of the module)."""
    ginv = d6d.inverse(e)
    dphi = Matrix([sp.sympify(comp).subs(PHI, ginv.act(PHI)) for comp in d])
    return tuple(d6d.det(e) * d6d.matrix(e) * dphi)


def _span_rank(*vectors) -> int:
    """Rank of a family of d-vectors of trigonometric polynomials (|m| <= 12), from the exact
    coefficients of their Laurent expansions, component by component."""
    cols = [(i, n) for i in range(3) for n in range(-12, 13)]
    rows = [[_laurent(v[i]).coeff(U, n) for (i, n) in cols] for v in vectors]
    return Matrix(rows).rank(iszerofunc=lambda x: sp.expand(x) == 0)


# --------------------------------------------------------------------------- generators and closure
def test_generators_are_yamamotos():
    S12, C2p = d6d.generators()
    assert S12 == ImmutableMatrix([[S3 / 2, -HALF, 0], [HALF, S3 / 2, 0], [0, 0, -1]])
    assert S12 == X(d6d.SIGMA_H * d6d.rotation_z(pi / 6))                 # diag(1, 1, -1) . R_z(30 deg)
    assert C2p == ImmutableMatrix([[S3 / 2, HALF, 0], [HALF, -S3 / 2, 0], [0, 0, -1]])
    # C2' is the two-fold rotation about the in-plane axis at 15 degrees: fixes it, sends z -> -z
    n = Matrix([c(pi / 12), s(pi / 12), 0])
    assert (C2p * n - n).applyfunc(sp.simplify) == sp.zeros(3, 1)
    assert C2p * Matrix([0, 0, 1]) == Matrix([0, 0, -1])
    assert X(C2p * C2p) == sp.eye(3)
    assert sp.expand(S12.det()) == -1 and sp.expand(C2p.det()) == 1
    assert X(S12 ** 12) == sp.eye(3) and X(S12 ** 6) != sp.eye(3)


def test_closure_has_order_24_and_is_the_labelled_element_set():
    cl = d6d.closure()
    assert len(cl) == 24 and len(set(cl)) == 24
    els = d6d.elements()
    assert len(els) == 24
    assert {d6d.matrix(e) for e in els} == set(cl)
    assert [e.label for e in els[:7]] == ["E", "S12", "C6", "S4", "C3", "S12^5", "C2"]
    assert els[12].label == "sigma_d(0)" and els[13].label == "C2'(15)" and els[23].label == "C2'(165)"
    assert d6d.matrix(S(1)) == d6d.generator_S12() and d6d.matrix(P(1)) == d6d.generator_C2prime()
    for e in els:
        M = d6d.matrix(e)
        assert X(M.T * M) == sp.eye(3)
        assert d6d.det(e) in (1, -1) and d6d.zz(e) == (-1) ** e.k
        assert M[0, 2] == M[1, 2] == M[2, 0] == M[2, 1] == 0            # diag(O, eps)
        assert d6d.element_of_matrix(M) == e
    # S_k = S12^k
    for k in range(12):
        assert d6d.power(S(1), k) == S(k)
        assert X(d6d.generator_S12() ** k) == d6d.matrix(S(k))


def test_multiplication_table_is_a_group():
    mult = d6d.multiplication_table()
    assert mult[0] == tuple(range(24))                                   # S_0 is the identity
    for i in range(24):
        assert sorted(mult[i]) == list(range(24))                        # rows are permutations
        assert mult[i][d6d.inverse_table()[i]] == 0
    for a in range(24):
        for b in range(24):
            for cc in range(0, 24, 5):
                assert mult[mult[a][b]][cc] == mult[a][mult[b][cc]]      # associativity
    # composition law (abstractly the dihedral group of order 24):
    # S_k S_l = S_{k+l}, S_k P_l = P_{l+k}, P_k S_l = P_{k-l}, P_k P_l = S_{k-l}
    assert d6d.compose(S(3), S(11)) == S(2)
    assert d6d.compose(S(3), P(1)) == P(4)
    assert d6d.compose(P(3), S(1)) == P(2)
    assert d6d.compose(P(3), P(1)) == S(2)
    assert d6d.inverse(P(5)) == P(5) and d6d.inverse(S(5)) == S(7)
    # C2'(15) S12 = sigma_d(0): the mirror containing the x axis (a framework edge)
    assert d6d.element_of_matrix(d6d.generator_C2prime() * d6d.generator_S12()) == P(0)


def test_no_inversion_no_horizontal_mirror_no_true_twelvefold_rotation():
    assert d6d.has_inversion() is False
    assert d6d.has_horizontal_mirror() is False
    assert d6d.has_true_twelvefold_rotation() is False
    assert d6d.INVERSION not in d6d.closure() and d6d.SIGMA_H not in d6d.closure()
    assert d6d.rotation_z(pi / 6) not in d6d.closure()                  # C12 itself is absent
    assert d6d.element_order(S(1)) == 12                                 # S12 has order 12 ...
    assert not d6d.is_proper(S(1))                                       # ... but is improper
    assert d6d.proper_rotation_orders() == {1: 1, 2: 7, 3: 2, 6: 2}
    assert sum(1 for e in d6d.elements() if d6d.is_proper(e)) == 12
    assert {d6d.element_order(e) for e in d6d.elements()} == {1, 2, 3, 4, 6, 12}
    assert d6d.group_facts() == {"order": 24, "n_classes": 9, "inversion": False, "horizontal_mirror": False,
                                 "true_twelvefold_rotation": False, "S12_order": 12,
                                 "proper_rotation_orders": {1: 1, 2: 7, 3: 2, 6: 2},
                                 "max_proper_rotation_order": 6,
                                 "mirror_directions_deg": [0, 30, 60, 90, 120, 150],
                                 "twofold_axes_deg": [15, 45, 75, 105, 135, 165]}


def test_mirror_planes_and_twofold_axes():
    assert d6d.mirror_direction_angles() == tuple(j * pi / 6 for j in range(6))        # framework edges
    assert d6d.twofold_axis_angles() == tuple(pi / 12 + j * pi / 6 for j in range(6))
    z = Matrix([0, 0, 1])
    for e in d6d.elements():
        if e.kind != "P":
            continue
        n = Matrix([c(e.line_angle), s(e.line_angle), 0])
        M = d6d.matrix(e)
        assert (M * n - n).applyfunc(sp.simplify) == sp.zeros(3, 1)
        if e.k % 2 == 0:      # sigma_d: the vertical mirror plane containing n and z
            assert M * z == z and d6d.det(e) == -1 and e.label.startswith("sigma_d")
        else:                 # C2': the two-fold rotation about n
            assert M * z == -z and d6d.det(e) == 1 and e.label.startswith("C2'")
    # the matrices act on the polar angle as the maps S_k, P_k
    for e in d6d.elements():
        v = Matrix([c(pi / 5), s(pi / 5), 0])
        target = Matrix([c(e.act(pi / 5)), s(e.act(pi / 5)), 0])
        assert (d6d.matrix(e) * v - target).applyfunc(sp.simplify) == sp.zeros(3, 1)


# --------------------------------------------------------------------------- classes and character table
def test_conjugacy_classes():
    cl = d6d.conjugacy_classes()
    assert tuple(cl) == ("E", "S12", "C6", "S4", "C3", "S12^5", "C2", "C2'", "sigma_d")
    assert tuple(len(v) for v in cl.values()) == (1, 2, 2, 2, 2, 2, 1, 6, 6)
    assert cl["S12"] == (1, 11) and cl["C6"] == (2, 10) and cl["S4"] == (3, 9)
    assert cl["C3"] == (4, 8) and cl["S12^5"] == (5, 7) and cl["C2"] == (6,)
    assert cl["C2'"] == tuple(12 + k for k in range(1, 12, 2))          # P_k, k odd: axes at 15 + 30 j deg
    assert cl["sigma_d"] == tuple(12 + k for k in range(0, 12, 2))      # P_k, k even: planes through 30 j deg
    assert [e.label for e in d6d.class_representatives()] == \
        ["E", "S12", "C6", "S4", "C3", "S12^5", "C2", "C2'(15)", "sigma_d(0)"]
    els = d6d.elements()
    assert {cc for cc, idx in cl.items() if not d6d.is_proper(els[idx[0]])} == {"S12", "S4", "S12^5", "sigma_d"}
    for e in els:
        assert d6d.class_of(e) == d6d._class_label(e)


def test_character_table_literal():
    T = d6d.character_table_D6d()
    assert T.order == 24
    assert tuple(T.irreps) == ("A1", "A2", "B1", "B2", "E1", "E2", "E3", "E4", "E5")
    assert len(T.irreps) == 9 == len(T.classes)
    assert T.is_orthonormal()
    assert sum(T.dim(n) ** 2 for n in T.irreps) == 24
    #                    E  S12  C6  S4  C3 S12^5 C2  C2' sigma_d
    assert T.chars["A1"] == (1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert T.chars["A2"] == (1, 1, 1, 1, 1, 1, 1, -1, -1)
    assert T.chars["B1"] == (1, -1, 1, -1, 1, -1, 1, 1, -1)
    assert T.chars["B2"] == (1, -1, 1, -1, 1, -1, 1, -1, 1)
    assert T.chars["E1"] == (2, S3, 1, 0, -1, -S3, -2, 0, 0)
    assert T.chars["E2"] == (2, 1, -1, -2, -1, 1, 2, 0, 0)
    assert T.chars["E3"] == (2, 0, -2, 0, 2, 0, -2, 0, 0)
    assert T.chars["E4"] == (2, -1, -1, 2, -1, -1, 2, 0, 0)
    assert T.chars["E5"] == (2, -S3, 1, 0, -1, S3, -2, 0, 0)
    # E_m on S_k is 2 cos(2 pi m k/12), 0 on the C2' and sigma_d
    for m in range(1, 6):
        for e in d6d.elements():
            expected = 2 * sp.cos(2 * pi * m * e.k / 12) if e.kind == "S" else 0
            assert d6d.character(f"E{m}", e) == sp.expand(expected)


def test_one_dimensional_characters_from_the_matrices():
    for name in ("A1", "A2", "B1", "B2"):
        assert d6d.is_one_dim_representation(name)
    C2p, sd = P(1), P(0)
    assert (d6d.character("B1", C2p), d6d.character("B1", sd)) == (1, -1)     # B1: +1 on the two-fold axes, -1 on the mirrors
    assert (d6d.character("B2", C2p), d6d.character("B2", sd)) == (-1, 1)     # B2: the reverse
    assert (d6d.character("A2", C2p), d6d.character("A2", sd)) == (-1, -1)    # A2: -1 under both
    for e in d6d.elements():
        assert d6d.character("B1", e) == d6d.det(e)                     # B1 = det g
        assert d6d.character("B2", e) == d6d.zz(e)                      # B2 = g_zz: z transforms as B2
        assert d6d.character("A2", e) == d6d.det(e) * d6d.zz(e)         # A2: the axial z (R_z)
        assert d6d.character("A2", e) == (-1 if e.kind == "P" else 1)
    assert d6d.is_class_function(tuple(d6d.function_character(2, e) for e in d6d.elements()))


# --------------------------------------------------------------------------- functions on the circle
def test_function_matrices_are_a_representation():
    els = d6d.elements()
    mult = d6d.multiplication_table()
    for m in (1, 3, 6):
        mats = [d6d.function_matrix(m, e) for e in els]
        for M in mats:
            assert X(M.T * M) == sp.eye(2)
        for i in range(24):
            for j in range(0, 24, 3):
                assert X(mats[i] * mats[j]) == mats[mult[i][j]]
        for e in els:
            if e.kind == "P":
                assert d6d.function_character(m, e) == 0
                assert sp.expand(mats[d6d.index_of(e)].det()) == -1
            else:
                assert d6d.function_character(m, e) == sp.expand(2 * sp.cos(2 * pi * m * e.k / 12))
    # S12 acts on the circle as the rotation by 30 degrees: f(phi) -> f(phi - pi/6)
    assert d6d.function_matrix(1, S(1)) == ImmutableMatrix([[S3 / 2, -HALF], [HALF, S3 / 2]])
    # sigma_d(0) acts as phi -> -phi (cos -> cos, sin -> -sin); C2'(15) as phi -> pi/6 - phi
    assert d6d.function_matrix(1, P(0)) == ImmutableMatrix([[1, 0], [0, -1]])
    assert d6d.function_matrix(1, P(1)) == ImmutableMatrix([[S3 / 2, HALF], [HALF, -S3 / 2]])
    assert d6d.function_matrix(0, S(5)) == ImmutableMatrix([[1]])
    assert d6d.function_matrix(6, S(1)) == -sp.eye(2)                   # e^{+-6 i phi} is odd under S12
    assert d6d.function_matrix(12, S(1)) == sp.eye(2)


def test_sector_matrices_are_representations():
    els = d6d.elements()
    mult = d6d.multiplication_table()
    for sector, m, dim in (("singlet", 6, 2), ("triplet_inplane", 1, 4), ("triplet_inplane", 5, 4), ("triplet_z", 3, 2)):
        mats = [d6d.sector_matrix(sector, m, e) for e in els]
        assert mats[0] == sp.eye(dim)
        for M in mats:
            assert X(M.T * M) == sp.eye(dim)
        for i in range(24):
            for j in range(0, 24, 3):
                assert X(mats[i] * mats[j]) == mats[mult[i][j]]
        assert d6d.is_class_function(tuple(sp.expand(M.trace()) for M in mats))
    # the in-plane triplet with m = 1 is E5 x E1 (axial in-plane vector times first harmonic): -3 on S12
    assert d6d.sector_character("triplet_inplane", 1) == (4, -3, 1, 0, 1, -3, 4, 0, 0)
    # d_z e^{+-i m phi} is A2 x E_m
    assert d6d.sector_character("triplet_z", 3) == (2, 0, -2, 0, 2, 0, -2, 0, 0)
    # the constant (m = 0) axial vector: in-plane E5 like (R_x, R_y), d_z A2 like R_z
    assert d6d.sector_decomposition("triplet_inplane", 0) == {"E5": 1}
    assert d6d.sector_decomposition("triplet_z", 0) == {"A2": 1}
    assert d6d.basis_functions("triplet_inplane", 1) == ((c(PHI), 0, 0), (s(PHI), 0, 0), (0, c(PHI), 0), (0, s(PHI), 0))
    assert d6d.basis_functions("singlet", 0) == (1,)


# --------------------------------------------------------------------------- Table 10
def test_singlet_sectors():
    assert d6d.sector_decomposition("singlet", 0) == {"A1": 1}
    assert d6d.sector_decomposition("singlet", 2) == {"E2": 1}
    assert d6d.sector_decomposition("singlet", 4) == {"E4": 1}
    assert d6d.sector_decomposition("singlet", 6) == {"B1": 1, "B2": 1}
    assert d6d.sector_bases("singlet", 0) == {"A1": (1,)}
    assert d6d.sector_bases("singlet", 2) == {"E2": (c(2 * PHI), s(2 * PHI))}
    assert d6d.sector_bases("singlet", 4) == {"E4": (c(4 * PHI), s(4 * PHI))}
    assert d6d.sector_basis("singlet", 6, "B1") == (s(6 * PHI),)        # sin 6 phi in B1
    assert d6d.sector_basis("singlet", 6, "B2") == (c(6 * PHI),)        # cos 6 phi in B2
    assert d6d.sector_basis("singlet", 6, "A1") == ()
    for m in (1, 3, 5):                                                 # odd scalar harmonics: E_m
        assert d6d.sector_decomposition("singlet", m) == {f"E{m}": 1}
    assert d6d.sector_decomposition("singlet", 12) == {"A1": 1, "A2": 1}
    assert d6d.sector_decomposition("singlet", 18) == {"B1": 1, "B2": 1}


def test_triplet_inplane_sectors():
    assert d6d.sector_decomposition("triplet_inplane", 1) == {"B1": 1, "B2": 1, "E4": 1}
    assert d6d.sector_decomposition("triplet_inplane", 3) == {"E2": 1, "E4": 1}
    assert d6d.sector_decomposition("triplet_inplane", 5) == {"A1": 1, "A2": 1, "E2": 1}
    b = d6d.sector_bases("triplet_inplane", 1)
    assert b["B1"] == ((c(PHI), s(PHI), 0),)
    assert b["B2"] == ((s(PHI), -c(PHI), 0),)
    assert b["E4"] == ((c(PHI), -s(PHI), 0), (s(PHI), c(PHI), 0))
    b = d6d.sector_bases("triplet_inplane", 3)
    assert b["E2"] == ((c(3 * PHI), -s(3 * PHI), 0), (s(3 * PHI), c(3 * PHI), 0))
    assert b["E4"] == ((c(3 * PHI), s(3 * PHI), 0), (s(3 * PHI), -c(3 * PHI), 0))
    b = d6d.sector_bases("triplet_inplane", 5)
    assert b["A1"] == ((s(5 * PHI), c(5 * PHI), 0),)
    assert b["A2"] == ((c(5 * PHI), -s(5 * PHI), 0),)
    assert b["E2"] == ((c(5 * PHI), s(5 * PHI), 0), (s(5 * PHI), -c(5 * PHI), 0))


def test_triplet_z_sectors():
    for m, irrep in ((1, "E1"), (3, "E3"), (5, "E5")):
        assert d6d.sector_decomposition("triplet_z", m) == {irrep: 1}
        assert d6d.sector_bases("triplet_z", m) == {irrep: ((0, 0, c(m * PHI)), (0, 0, s(m * PHI)))}


def test_basis_functions_obey_the_transformation_laws():
    """The one-dimensional basis functions of Table 10 are eigenfunctions with eigenvalue
    chi(g) under the transformation laws applied by direct substitution (independently of the
    sector matrices), for every element."""
    for e in d6d.elements():
        for m, irrep in ((1, "B1"), (1, "B2"), (5, "A1"), (5, "A2")):
            (d,) = d6d.sector_basis("triplet_inplane", m, irrep)
            img = _transform_axial(e, d)
            assert all(_vanishes(u - d6d.character(irrep, e) * v) for u, v in zip(img, d)), (e.label, irrep)
        for m, irrep in ((6, "B1"), (6, "B2"), (0, "A1")):
            (f,) = d6d.sector_basis("singlet", m, irrep)
            img = sp.sympify(f).subs(PHI, d6d.inverse(e).act(PHI))
            assert _vanishes(img - d6d.character(irrep, e) * f), (e.label, irrep)
    # the two-dimensional ones span invariant subspaces: the images of an E pair lie in its span
    # (rank 2 of {f1, f2, g f1, g f2}), and the pairs of different irreps are independent
    for m, irrep in ((1, "E4"), (3, "E2"), (3, "E4"), (5, "E2")):
        f1, f2 = d6d.sector_basis("triplet_inplane", m, irrep)
        assert _span_rank(f1, f2) == 2
        for e in (S(1), P(1), P(0), S(3), S(5)):
            assert _span_rank(f1, f2, _transform_axial(e, f1), _transform_axial(e, f2)) == 2, (irrep, e.label)
    e2, e4 = d6d.sector_basis("triplet_inplane", 3, "E2"), d6d.sector_basis("triplet_inplane", 3, "E4")
    assert _span_rank(*e2, *e4) == 4
    # S12 rotates the E4 pair of m = 1 by 120 degrees (character 2 cos 120 = -1):
    # g f1 = cos 120 f1 - sin 120 f2, g f2 = sin 120 f1 + cos 120 f2
    f1, f2 = d6d.sector_basis("triplet_inplane", 1, "E4")
    g1, g2 = _transform_axial(S(1), f1), _transform_axial(S(1), f2)
    cc, ss = c(2 * pi / 3), s(2 * pi / 3)
    assert all(_vanishes(u - (cc * v1 - ss * v2)) for u, v1, v2 in zip(g1, f1, f2))
    assert all(_vanishes(u - (ss * v1 + cc * v2)) for u, v1, v2 in zip(g2, f1, f2))


def test_isotypic_projectors():
    T = d6d.character_table_D6d()
    for sector, m in (("singlet", 6), ("triplet_inplane", 1), ("triplet_inplane", 3), ("triplet_inplane", 5), ("triplet_z", 1)):
        dim = d6d.sector_matrix(sector, m, d6d.identity()).rows
        total = sp.zeros(dim)
        dec = d6d.sector_decomposition(sector, m)
        for irrep in d6d.IRREPS_D6D:
            Pm = d6d.isotypic_projector(sector, m, irrep)
            assert X(Pm * Pm) == Pm and X(Pm.T) == Pm
            assert Pm.rank(iszerofunc=lambda x: sp.expand(x) == 0) == dec.get(irrep, 0) * T.dim(irrep)
            total += Pm
        assert X(total) == sp.eye(dim)
        assert sum(v * T.dim(k) for k, v in dec.items()) == dim
    # the A1 projector of the in-plane m = 5 sector is invariant under every element
    Pm = d6d.isotypic_projector("triplet_inplane", 5, "A1")
    assert Pm == ImmutableMatrix([[0, 0, 0, 0], [0, HALF, HALF, 0], [0, HALF, HALF, 0], [0, 0, 0, 0]])
    for e in d6d.elements():
        M = d6d.sector_matrix("triplet_inplane", 5, e)
        assert X(M * Pm) == Pm and X(Pm * M) == Pm


def test_antisymmetric_spin_orbit_vector():
    assert d6d.asoc_A1_content(5) == {1: {"inplane": 0, "z": 0}, 3: {"inplane": 0, "z": 0}, 5: {"inplane": 1, "z": 0}}
    assert d6d.asoc_vector() == (s(5 * PHI), c(5 * PHI), 0)
    # no g_z for any odd m: the sectors depend on m mod 12 only, so m <= 11 covers all odd m
    assert all(v["z"] == 0 for v in d6d.asoc_A1_content(11).values())
    for m in range(1, 12, 2):
        assert "A1" not in d6d.sector_decomposition("triplet_z", m)
        assert d6d.sector_decomposition("triplet_z", m + 12) == d6d.sector_decomposition("triplet_z", m)
        assert d6d.sector_decomposition("triplet_inplane", m + 12) == d6d.sector_decomposition("triplet_inplane", m)
    # m = 7 = -5 (mod 12) is the same E5 x E5 sector as m = 5; m = 1, 3, 9, 11 have no invariant
    assert {m: v["inplane"] for m, v in d6d.asoc_A1_content(11).items()} == {1: 0, 3: 0, 5: 1, 7: 1, 9: 0, 11: 0}
    # g is invariant under all 24 elements by direct substitution, and odd in k (phi -> phi + pi)
    g = d6d.asoc_vector()
    for e in d6d.elements():
        assert all(_vanishes(u - v) for u, v in zip(_transform_axial(e, g), g)), e.label
    assert all(_vanishes(sp.sympify(u).subs(PHI, PHI + pi) + u) for u in g)


# --------------------------------------------------------------------------- restriction to D2d
def test_restriction_to_D2d():
    sub = d6d.d2d_elements()
    assert sub == (0, 3, 6, 9, 13, 16, 19, 22)
    els = d6d.elements()
    assert [els[i].label for i in sub] == ["E", "S4", "C2", "S4^3", "C2'(15)", "sigma_d(60)", "C2'(105)", "sigma_d(150)"]
    assert d6d.power(S(1), 3) == S(3)                                    # S4 = S12^3
    mult = d6d.multiplication_table()
    assert all(mult[a][b] in sub for a in sub for b in sub)               # closed
    assert d6d.d2d_classes() == {"E": (0,), "S4": (3, 9), "C2": (6,), "C2'": (13, 19), "sigma_d": (16, 22)}
    T = d6d.character_table_D2d()
    assert T.order == 8 and T.is_orthonormal() and sum(T.dim(n) ** 2 for n in T.irreps) == 8
    assert tuple(T.irreps) == ("A1", "A2", "B1", "B2", "E")
    #                    E  S4  C2  C2' sigma_d
    assert T.chars["A1"] == (1, 1, 1, 1, 1)
    assert T.chars["A2"] == (1, 1, 1, -1, -1)
    assert T.chars["B1"] == (1, -1, 1, 1, -1)
    assert T.chars["B2"] == (1, -1, 1, -1, 1)
    assert T.chars["E"] == (2, 0, -2, 0, 0)
    assert d6d.restriction_table_D2d() == {"A1": {"A1": 1}, "A2": {"A2": 1}, "B1": {"B1": 1}, "B2": {"B2": 1},
                                           "E1": {"E": 1}, "E2": {"B1": 1, "B2": 1}, "E3": {"E": 1},
                                           "E4": {"A1": 1, "A2": 1}, "E5": {"E": 1}}


# --------------------------------------------------------------------------- in-plane field orbit
def test_field_orbit_and_period():
    twelve = tuple(k * pi / 6 for k in range(12))
    assert d6d.field_orbit(0) == twelve                                  # with H -> -H
    assert d6d.field_orbit(0, reversal=False) == twelve                  # already all multiples of 30 deg
    assert d6d.field_angle_period() == pi / 6
    # a generic direction alpha: alpha + k pi/6 and -alpha + k pi/6 (24 directions)
    a = pi / 5
    expected = sorted({(sp.Rational(x / pi) % 2) * pi
                       for x in [a + k * pi / 6 for k in range(12)] + [-a + k * pi / 6 for k in range(12)]})
    assert list(d6d.field_orbit(a)) == expected and len(expected) == 24
    assert len(d6d.field_orbit(a, reversal=False)) == 24
    # the angles agree with the exact matrix images det(g) g H of H = (1, 0, 0)
    H = Matrix([1, 0, 0])
    for e in d6d.elements():
        img = d6d.det(e) * d6d.matrix(e) * H
        ang = d6d.axial_image_angle(e, 0)
        assert X(img - Matrix([c(ang), s(ang), 0])) == sp.zeros(3, 1)
    # the improper operations rotate the axial vector by an extra pi: S12 sends 0 to 210 deg
    assert d6d.axial_image_angle(S(1), 0) == 7 * pi / 6
    assert d6d.axial_image_angle(S(2), 0) == pi / 3
    assert d6d.axial_image_angle(P(1), 0) == pi / 6                    # C2'(15): reflection across 15 deg
    assert d6d.axial_image_angle(P(0), 0) == pi                        # sigma_d(0): H -> -H


# --------------------------------------------------------------------------- formatters
def test_table10_and_formatters_are_plain_data():
    rows = {r["irrep"]: r for r in d6d.format_table10()}
    assert list(rows) == list(d6d.IRREPS_D6D)
    assert rows["A1"] == {"irrep": "A1", "singlet": [{"m": 0, "functions": ["1"]}],
                          "triplet": [{"m": 5, "sector": "in-plane", "functions": [["sin(5*phi)", "cos(5*phi)", "0"]]}]}
    assert rows["A2"] == {"irrep": "A2", "singlet": [],
                          "triplet": [{"m": 5, "sector": "in-plane", "functions": [["cos(5*phi)", "-sin(5*phi)", "0"]]}]}
    assert rows["B1"] == {"irrep": "B1", "singlet": [{"m": 6, "functions": ["sin(6*phi)"]}],
                          "triplet": [{"m": 1, "sector": "in-plane", "functions": [["cos(phi)", "sin(phi)", "0"]]}]}
    assert rows["B2"] == {"irrep": "B2", "singlet": [{"m": 6, "functions": ["cos(6*phi)"]}],
                          "triplet": [{"m": 1, "sector": "in-plane", "functions": [["sin(phi)", "-cos(phi)", "0"]]}]}
    assert rows["E1"] == {"irrep": "E1", "singlet": [],
                          "triplet": [{"m": 1, "sector": "d_z", "functions": [["0", "0", "cos(phi)"], ["0", "0", "sin(phi)"]]}]}
    assert rows["E2"] == {"irrep": "E2", "singlet": [{"m": 2, "functions": ["cos(2*phi)", "sin(2*phi)"]}],
                          "triplet": [{"m": 3, "sector": "in-plane",
                                       "functions": [["cos(3*phi)", "-sin(3*phi)", "0"], ["sin(3*phi)", "cos(3*phi)", "0"]]},
                                      {"m": 5, "sector": "in-plane",
                                       "functions": [["cos(5*phi)", "sin(5*phi)", "0"], ["sin(5*phi)", "-cos(5*phi)", "0"]]}]}
    assert rows["E3"] == {"irrep": "E3", "singlet": [],
                          "triplet": [{"m": 3, "sector": "d_z", "functions": [["0", "0", "cos(3*phi)"], ["0", "0", "sin(3*phi)"]]}]}
    assert rows["E4"] == {"irrep": "E4", "singlet": [{"m": 4, "functions": ["cos(4*phi)", "sin(4*phi)"]}],
                          "triplet": [{"m": 1, "sector": "in-plane",
                                       "functions": [["cos(phi)", "-sin(phi)", "0"], ["sin(phi)", "cos(phi)", "0"]]},
                                      {"m": 3, "sector": "in-plane",
                                       "functions": [["cos(3*phi)", "sin(3*phi)", "0"], ["sin(3*phi)", "-cos(3*phi)", "0"]]}]}
    assert rows["E5"] == {"irrep": "E5", "singlet": [],
                          "triplet": [{"m": 5, "sector": "d_z", "functions": [["0", "0", "cos(5*phi)"], ["0", "0", "sin(5*phi)"]]}]}
    ct = d6d.format_character_table()
    assert ct["classes"] == ["E", "S12", "C6", "S4", "C3", "S12^5", "C2", "C2'", "sigma_d"]
    assert ct["sizes"] == [1, 2, 2, 2, 2, 2, 1, 6, 6]
    assert ct["rows"][8] == {"irrep": "E5", "chars": ["2", "-sqrt(3)", "1", "0", "-1", "sqrt(3)", "-2", "0", "0"]}
    assert d6d.format_sector("triplet_inplane", 1) == {
        "m": 1, "irreps": "B1 + B2 + E4",
        "basis": {"B1": [["cos(phi)", "sin(phi)", "0"]], "B2": [["sin(phi)", "-cos(phi)", "0"]],
                  "E4": [["cos(phi)", "-sin(phi)", "0"], ["sin(phi)", "cos(phi)", "0"]]}}
    summ = d6d.summary()
    assert summ["order"] == 24 and summ["facts"]["n_classes"] == 9
    assert summ["generators"]["S12"] == [["sqrt(3)/2", "-1/2", "0"], ["1/2", "sqrt(3)/2", "0"], ["0", "0", "-1"]]
    assert summ["generators"]["C2prime_15deg"] == [["sqrt(3)/2", "1/2", "0"], ["1/2", "-sqrt(3)/2", "0"], ["0", "0", "-1"]]
    assert [r["irreps"] for r in summ["singlet"]] == ["A1", "E2", "E4", "B1 + B2"]
    assert [r["irreps"] for r in summ["triplet_inplane"]] == ["B1 + B2 + E4", "E2 + E4", "A1 + A2 + E2"]
    assert [r["irreps"] for r in summ["triplet_z"]] == ["E1", "E3", "E5"]
    assert summ["asoc"] == {"A1_content": {"1": {"inplane": 0, "z": 0}, "3": {"inplane": 0, "z": 0},
                                           "5": {"inplane": 1, "z": 0}},
                            "g": ["sin(5*phi)", "cos(5*phi)", "0"]}
    assert summ["restriction_D2d"]["table"] == {"A1": "A1", "A2": "A2", "B1": "B1", "B2": "B2", "E1": "E",
                                                "E2": "B1 + B2", "E3": "E", "E4": "A1 + A2", "E5": "E"}
    assert summ["field_orbit"] == {"angles_deg": list(range(0, 360, 30)), "period_deg": 30}
    assert summ["elements"][1] == {"index": 1, "label": "S12", "class": "S12", "det": -1, "zz": -1, "order": 12}
    json.dumps(summ)
