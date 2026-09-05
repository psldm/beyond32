"""Table 4 of the paper: permutation representations of I and I_h on the 12/20/30/60 orbits.

Exact comparisons in these tests use ``sympy.expand`` on products and sums of canonical
numbers a + b*sqrt5 (which yields the same canonical form as ``_exact.canon`` and is
several thousand times faster)."""
import pytest
import sympy as sp
from sympy import Matrix, S

from beyond32._exact import PHI, canon
from beyond32 import groups as g
from beyond32 import shells as sh

phi = canon(PHI)


def _parallel(p, a) -> bool:
    """p x a = 0, exactly (p a tuple of canonical numbers, a a column Matrix)."""
    cross = (p[1] * a[2] - p[2] * a[1], p[2] * a[0] - p[0] * a[2], p[0] * a[1] - p[1] * a[0])
    return all(sp.expand(c) == 0 for c in cross)


def _dot(p, a):
    return sp.expand(p[0] * a[0] + p[1] * a[1] + p[2] * a[2])


def test_orbit_sizes_and_exactness():
    G = g.icosahedral_group()
    for name, size in (("12", 12), ("20", 20), ("30", 30), ("60", 60)):
        seed = sh.SEEDS[name]
        pts = sh.orbit(seed)
        assert len(pts) == size
        assert len(set(pts)) == size                      # hashable, exact, distinct
        assert seed in pts
        for p in pts:
            assert len(p) == 3 and all(isinstance(c, sp.Expr) and c.free_symbols == set() for c in p)
        # closed under I (exact), transitive, centrosymmetric; the I_h orbit is the same set
        pset = set(pts)
        for R in G.rotations[::7]:
            for p in pts:
                assert tuple((R.matrix * Matrix(p)).applyfunc(sp.expand)) in pset
        assert set(sh.orbit(pts[-1])) == pset
        assert sh.is_centrosymmetric(pts)
        assert set(sh.orbit_Ih(seed)) == pset
        assert sh.shell(name).size == size and set(sh.shell(name).points) == pset


def test_orbit_points_are_the_symmetry_axes():
    # 12: the two ends of the six five-fold axes, cyclic permutations of (+-phi, +-1, 0)
    pts12 = sh.orbit((PHI, 1, 0))
    expected = set()
    for v in ((phi, 1, 0), (phi, -1, 0), (-phi, 1, 0), (-phi, -1, 0)):
        for perm in (v, (v[2], v[0], v[1]), (v[1], v[2], v[0])):
            expected.add(tuple(S(c) for c in perm))
    assert set(pts12) == expected
    # 12 / 20 / 30: every axis of the class carries exactly two orbit points (its two ends)
    for pts, axes in ((pts12, g.fivefold_axes()), (sh.orbit((1, 1, 1)), g.threefold_axes()),
                      (sh.orbit((0, 0, 1)), g.twofold_axes())):
        assert len(pts) == 2 * len(axes)
        for a in axes:
            assert sum(1 for p in pts if _parallel(p, a)) == 2
    pts20 = sh.orbit((1, 1, 1))
    assert all(tuple(S(c) for c in v) in pts20
               for v in ((1, 1, 1), (1, 1, -1), (-1, -1, -1), (1, -1, 1)))
    # 60: on no rotation axis, and on exactly one mirror plane (perpendicular to a two-fold axis)
    axes = list(g.fivefold_axes()) + list(g.threefold_axes()) + list(g.twofold_axes())
    for p in sh.orbit((0, 1, 3 * PHI)):
        assert not any(_parallel(p, a) for a in axes)
        assert sum(1 for a in g.twofold_axes() if _dot(p, a) == 0) == 1


def test_permutation_characters():
    # fixed points of one representative of each class E, C5, C5^2, C3, C2 ...
    expected_I = {"12": (12, 2, 2, 0, 0), "20": (20, 0, 0, 2, 0),
                  "30": (30, 0, 0, 0, 2), "60": (60, 0, 0, 0, 0)}
    # ... and of the improper elements i, S10^3, S10, S6, sigma = -E, -C5, -C5^2, -C3, -C2:
    # every mirror plane contains four points of each orbit
    expected_improper = (0, 0, 0, 0, 4)
    for name, chars in expected_I.items():
        pts = sh.orbit(sh.SEEDS[name])
        assert sh.permutation_character(pts) == chars
        assert sh.permutation_character(pts, improper=True) == expected_improper
        assert sh.permutation_character_Ih(pts) == chars + expected_improper
        assert sh.shell(name).character_I == chars
        assert sh.shell(name).character_Ih == chars + expected_improper
        # raw counts: E fixes everything, the inversion nothing; Burnside for a transitive
        # action: sum over I of the fixed points is |I| = 60, and so is the improper sum
        proper = sh.fixed_point_counts(pts)
        improper = sh.fixed_point_counts(pts, improper=True)
        assert proper[0] == len(pts) and improper[0] == 0
        assert sum(proper) == 60 and sum(improper) == 60
    assert g.character_table_Ih().classes == g.CLASSES_I + ("i", "S10^3", "S10", "S6", "sigma")


def test_decomposition_under_I():
    expected = {"12": {"A": 1, "T1": 1, "T2": 1, "H": 1},
                "20": {"A": 1, "T1": 1, "T2": 1, "G": 2, "H": 1},
                "30": {"A": 1, "T1": 1, "T2": 1, "G": 2, "H": 3},
                "60": {"A": 1, "T1": 3, "T2": 3, "G": 4, "H": 5}}
    for name, dec in expected.items():
        assert sh.decomposition_I(name) == dec, name
        assert list(sh.decomposition_I(name)) == [k for k in g.IRREPS_I if k in dec]
        assert sh.decomposition_I(int(name)) == dec
        assert g.CHAR_I.decompose(sh.permutation_character(sh.orbit(sh.SEEDS[name]))) == dec
    # the 60-orbit carries the regular representation: every irrep d_Gamma times
    assert sh.decomposition_I("60") == {k: g.CHAR_I.dim(k) for k in g.IRREPS_I}


def test_decomposition_under_Ih():
    expected = {"12": {"Ag": 1, "Hg": 1, "T1u": 1, "T2u": 1},
                "20": {"Ag": 1, "Gg": 1, "Hg": 1, "T1u": 1, "T2u": 1, "Gu": 1},
                "30": {"Ag": 1, "Gg": 1, "Hg": 2, "T1u": 1, "T2u": 1, "Gu": 1, "Hu": 1},
                "60": {"Ag": 1, "T1g": 1, "T2g": 1, "Gg": 2, "Hg": 3,
                       "T1u": 2, "T2u": 2, "Gu": 2, "Hu": 2}}
    T = g.character_table_Ih()
    for name, dec in expected.items():
        assert sh.decomposition_Ih(name) == dec, name
        assert list(sh.decomposition_Ih(name)) == [k for k in sh.IRREPS_IH_ORDER if k in dec]
        assert T.decompose(sh.permutation_character_Ih(sh.orbit(sh.SEEDS[name]))) == dec


def test_dimensions_restriction_and_enforced_node():
    T = g.character_table_Ih()
    for name in sh.SHELL_NAMES:
        size = int(name)
        dec_I, dec_Ih = sh.decomposition_I(name), sh.decomposition_Ih(name)
        assert sum(m * g.CHAR_I.dim(k) for k, m in dec_I.items()) == size
        assert sum(m * T.dim(k) for k, m in dec_Ih.items()) == size
        # restricting I_h -> I: Gamma_g and Gamma_u both become Gamma
        restricted = {}
        for k, m in dec_Ih.items():
            restricted[k[:-1]] = restricted.get(k[:-1], 0) + m
        assert restricted == dec_I
        # A (Ag) occurs exactly once (transitive action), Au never
        assert dec_I["A"] == 1 and dec_Ih["Ag"] == 1 and "Au" not in dec_Ih
    # enforced nodes in real space: G contains no trivial character of C5, so it cannot
    # live on the twelve five-fold directions
    assert "G" not in sh.decomposition_I("12")
    assert "Gg" not in sh.decomposition_Ih("12") and "Gu" not in sh.decomposition_Ih("12")
    c5 = g.generate_subgroup([g.rotation_about((PHI, 1, 0), "C5").index])
    G = g.icosahedral_group()
    assert len(c5) == 5
    assert canon(sum(G.character("G", i) for i in c5)) == 0
    assert canon(sum(G.character("H", i) for i in c5)) == 5


def test_frobenius_reciprocity_cross_check():
    for name in sh.SHELL_NAMES:
        assert sh.decomposition_by_frobenius(name, "I") == sh.decomposition_I(name)
        assert sh.decomposition_by_frobenius(name, "Ih") == sh.decomposition_Ih(name)


def test_stabilisers():
    expected = {"12": (5, "C5", "C5v"), "20": (3, "C3", "C3v"), "30": (2, "C2", "C2v"),
                "60": (1, "C1", "Cs")}
    G = g.icosahedral_group()
    for name, (n, lab_I, lab_Ih) in expected.items():
        seed = sh.SEEDS[name]
        stab = sh.stabiliser_I(seed)
        assert len(stab) == n and 0 in stab and n * int(name) == 60
        stab_h = sh.stabiliser_Ih(seed)
        assert len(stab_h) == 2 * n and 2 * n * int(name) == 120
        assert [i for i, s in stab_h if s == 1] == list(stab)
        # the improper elements of the stabiliser are reflections -C2 in planes through p
        assert all(G.rotations[i].cls == "C2" for i, s in stab_h if s == -1)
        assert sh.stabiliser_labels(seed) == (lab_I, lab_Ih)
        s = sh.shell(name)
        assert (s.stabiliser_I, s.stabiliser_Ih) == (lab_I, lab_Ih)


def test_table_and_formatting():
    rows = sh.shells_table()
    assert [r["orbit"] for r in rows] == ["12", "20", "30", "60"]
    assert [r["size"] for r in rows] == [12, 20, 30, 60]
    assert [r["stabiliser_I"] for r in rows] == ["C5", "C3", "C2", "C1"]
    assert [r["stabiliser_Ih"] for r in rows] == ["C5v", "C3v", "C2v", "Cs"]
    assert [r["seed"] for r in rows] == ["(phi, 1, 0)", "(1, 1, 1)", "(0, 0, 1)", "(0, 1, 3*phi)"]
    assert [r["polyhedron"] for r in rows] == ["icosahedron", "dodecahedron",
                                               "icosidodecahedron", "truncated icosahedron"]
    assert rows[0]["character_I"] == [12, 2, 2, 0, 0]
    assert rows[0]["character_Ih"] == [12, 2, 2, 0, 0, 0, 0, 0, 0, 4]
    assert rows[0]["decomposition_Ih"] == {"Ag": 1, "Hg": 1, "T1u": 1, "T2u": 1}
    fmt = {r["orbit"]: r for r in sh.format_shells_table()}
    assert fmt["12"]["Ih"] == "Ag + Hg + T1u + T2u"
    assert fmt["20"]["Ih"] == "Ag + Gg + Hg + T1u + T2u + Gu"
    assert fmt["30"]["Ih"] == "Ag + Gg + 2Hg + T1u + T2u + Gu + Hu"
    assert fmt["60"]["Ih"] == "Ag + T1g + T2g + 2Gg + 3Hg + 2T1u + 2T2u + 2Gu + 2Hu"
    assert fmt["12"]["I"] == "A + T1 + T2 + H"
    assert fmt["20"]["I"] == "A + T1 + T2 + 2G + H"
    assert fmt["30"]["I"] == "A + T1 + T2 + 2G + 3H"
    assert fmt["60"]["I"] == "A + 3T1 + 3T2 + 4G + 5H"
    assert fmt["12"]["stabiliser"] == "C5v" and fmt["60"]["size"] == "60"
    # the formatter orders irreps as the paper does, whatever the input order
    assert sh.format_decomposition({"T2u": 1, "Hg": 1, "Ag": 1, "T1u": 1}) == "Ag + Hg + T1u + T2u"
    assert sh.format_decomposition({"H": 3, "A": 1}, plus="\\oplus") == "A \\oplus 3H"
    # every value in the table rows is plain Python data
    for r in rows:
        assert all(isinstance(v, (str, int, list, dict)) for v in r.values())
        assert all(isinstance(c, int) for c in r["character_I"] + r["character_Ih"])
        assert all(isinstance(m, int) for m in r["decomposition_Ih"].values())


def test_invalid_input_is_rejected():
    seed = sh.SEEDS["12"]
    with pytest.raises(KeyError):
        sh.decomposition_I("13")
    with pytest.raises(ValueError):
        sh.orbit((0, 0, 0))
    # a single point is not an I-orbit: the fixed-point count is not a class function
    with pytest.raises(ValueError):
        sh.permutation_character([seed])
    # a set that is not centrosymmetric carries no action of I_h
    with pytest.raises(ValueError):
        sh.permutation_character([seed, (1, 0, 0)], improper=True)
    with pytest.raises(ValueError):
        sh.decomposition_by_frobenius("12", "D5")
