"""Tests for beyond32.d12: the dodecagonal group D12 of Appendix B.

The reference values are pinned literally; the module must compute them from the group.
"""
import sympy as sp
from sympy import Matrix, Rational, pi, sqrt

from beyond32 import d12

S3 = sqrt(3)


# --------------------------------------------------------------------------- group structure
def test_elements_and_multiplication_table():
    els = d12.elements()
    assert len(els) == 24
    assert [e.label for e in els[:3]] == ["R_0", "R_1", "R_2"]
    assert els[12].label == "P_0"
    mult = d12.multiplication_table()
    assert mult[0] == tuple(range(24))                       # R_0 is the identity
    for i in range(24):
        assert sorted(mult[i]) == list(range(24))            # rows are permutations
        assert mult[i][d12.inverse_table()[i]] == 0
    # associativity on the full table
    for a in range(24):
        for b in range(24):
            for c in range(0, 24, 5):
                assert mult[mult[a][b]][c] == mult[a][mult[b][c]]
    # the composition law: R_k R_l = R_{k+l}, R_k P_l = P_{l+k}, P_k R_l = P_{k-l}, P_k P_l = R_{k-l}
    R, P = (lambda k: d12.D12Element("R", k % 12)), (lambda k: d12.D12Element("P", k % 12))
    assert d12.compose(R(3), R(11)) == R(2)
    assert d12.compose(R(3), P(1)) == P(4)
    assert d12.compose(P(3), R(1)) == P(2)
    assert d12.compose(P(3), P(1)) == R(2)
    assert d12.inverse(P(5)) == P(5) and d12.inverse(R(5)) == R(7)


def test_matrices_are_a_faithful_SO3_representation():
    els = d12.elements()
    mats = [d12.matrix(e) for e in els]
    for M in mats:
        assert (M.T * M).applyfunc(sp.expand) == sp.eye(3)
        assert sp.expand(M.det()) == 1
    assert len({tuple(M.applyfunc(sp.expand)) for M in mats}) == 24
    mult = d12.multiplication_table()
    for i in range(24):
        for j in range(24):
            assert (mats[i] * mats[j] - mats[mult[i][j]]).applyfunc(sp.expand) == sp.zeros(3, 3)
    # the matrices act on the polar angle as the maps R_k, P_k
    for e in els:
        v = Matrix([sp.cos(pi / 5), sp.sin(pi / 5), 0])
        img = d12.matrix(e) * v
        target = Matrix([sp.cos(e.act(pi / 5)), sp.sin(e.act(pi / 5)), 0])
        assert (img - target).applyfunc(sp.simplify) == sp.zeros(3, 1)


def test_conjugacy_classes():
    cl = d12.conjugacy_classes()
    assert tuple(cl) == ("E", "C12", "C6", "C4", "C3", "C12^5", "C2", "C2'", "C2''")
    assert tuple(len(v) for v in cl.values()) == (1, 2, 2, 2, 2, 2, 1, 6, 6)
    assert cl["C2'"] == tuple(12 + k for k in range(0, 12, 2))     # P_k, k even: axes at k pi/6
    assert cl["C2''"] == tuple(12 + k for k in range(1, 12, 2))    # P_k, k odd: axes at pi/12 + k pi/6
    assert cl["C12"] == (1, 11) and cl["C2"] == (6,)
    for e in d12.elements():
        if e.is_twofold:
            assert e.axis_angle == e.k * pi / 12
            assert e.twofold_type == ("C2'" if e.k % 2 == 0 else "C2''")


# --------------------------------------------------------------------------- character table
def test_character_table_literal():
    T = d12.character_table_D12()
    assert T.order == 24
    assert tuple(T.irreps) == ("A1", "A2", "B1", "B2", "E1", "E2", "E3", "E4", "E5")
    assert len(T.irreps) == 9 == len(T.classes)
    assert T.is_orthonormal()
    assert sum(T.dim(n) ** 2 for n in T.irreps) == 24
    #                    E  C12  C6  C4  C3 C12^5 C2  C2' C2''
    assert T.chars["A1"] == (1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert T.chars["A2"] == (1, 1, 1, 1, 1, 1, 1, -1, -1)
    assert T.chars["B1"] == (1, -1, 1, -1, 1, -1, 1, 1, -1)
    assert T.chars["B2"] == (1, -1, 1, -1, 1, -1, 1, -1, 1)
    assert T.chars["E1"] == (2, S3, 1, 0, -1, -S3, -2, 0, 0)
    assert T.chars["E2"] == (2, 1, -1, -2, -1, 1, 2, 0, 0)
    assert T.chars["E3"] == (2, 0, -2, 0, 2, 0, -2, 0, 0)
    assert T.chars["E4"] == (2, -1, -1, 2, -1, -1, 2, 0, 0)
    assert T.chars["E5"] == (2, -S3, 1, 0, -1, S3, -2, 0, 0)
    # E_m on the rotations is 2 cos(2 pi m k/12), 0 on the two-fold rotations
    for m in range(1, 6):
        for e in d12.elements():
            expected = 2 * sp.cos(2 * pi * m * e.k / 12) if e.is_rotation else 0
            assert d12.character(f"E{m}", e) == sp.expand(expected)


def test_one_dim_characters_and_twofold_parities():
    P0, P1 = d12.D12Element("P", 0), d12.D12Element("P", 1)     # a C2' and a C2'' element
    assert P0.twofold_type == "C2'" and P1.twofold_type == "C2''"
    for name in ("A1", "A2", "B1", "B2"):
        assert d12.is_one_dim_representation(name)
    assert (d12.character("B1", P0), d12.character("B1", P1)) == (1, -1)   # B1: even C2', odd C2''
    assert (d12.character("B2", P0), d12.character("B2", P1)) == (-1, 1)   # B2: the reverse
    assert (d12.character("A2", P0), d12.character("A2", P1)) == (-1, -1)  # A2: odd under both
    assert all(d12.character("A2", e) == -1 for e in d12.elements() if e.is_twofold)


def test_pair_matrices_are_a_representation_with_trace_zero_on_C2():
    els = d12.elements()
    mult = d12.multiplication_table()
    for m in (1, 3):
        mats = [d12.pair_matrix(m, e) for e in els]
        for M in mats:
            assert (M.T * M).applyfunc(sp.expand) == sp.eye(2)
        for i in range(24):
            for j in range(0, 24, 3):
                assert (mats[i] * mats[j] - mats[mult[i][j]]).applyfunc(sp.expand) == sp.zeros(2, 2)
        for e in els:
            if e.is_twofold:
                assert d12.pair_character(m, e) == 0
                assert sp.expand(mats[d12.index_of(e)].det()) == -1
            else:
                assert d12.pair_character(m, e) == sp.expand(2 * sp.cos(2 * pi * m * e.k / 12))
    # the exact rotation matrix for m = 1, k = 1 (rotation by 2 pi/12)
    M = d12.pair_matrix(1, d12.D12Element("R", 1))
    assert M == Matrix([[S3 / 2, -Rational(1, 2)], [Rational(1, 2), S3 / 2]])
    assert d12.pair_character(0, els[5]) == 1
    assert d12.is_class_function(tuple(d12.pair_character(2, e) for e in els))


# --------------------------------------------------------------------------- harmonics -> irreps
def test_harmonic_assignment_m_0_to_12():
    expected = {0: {"A1": 1}, 1: {"E1": 1}, 2: {"E2": 1}, 3: {"E3": 1}, 4: {"E4": 1}, 5: {"E5": 1},
                6: {"B1": 1, "B2": 1}, 7: {"E5": 1}, 8: {"E4": 1}, 9: {"E3": 1}, 10: {"E2": 1},
                11: {"E1": 1}, 12: {"A1": 1, "A2": 1}}
    assert d12.harmonic_assignment(12) == expected
    T = d12.character_table_D12()
    for m, dec in expected.items():
        assert sum(v * T.dim(k) for k, v in dec.items()) == (1 if m == 0 else 2)


def test_residue_table_eq_24():
    rt = d12.residue_table()
    assert rt[0] == {"A1": 1, "A2": 1}
    assert rt[6] == {"B1": 1, "B2": 1}
    for r in range(1, 6):
        assert rt[r] == {f"E{r}": 1}
    for r in range(7, 12):
        assert rt[r] == {f"E{12 - r}": 1}
    assert d12.harmonic_irreps(24) == {"A1": 1, "A2": 1}
    assert d12.harmonic_irreps(18) == {"B1": 1, "B2": 1}
    assert d12.harmonic_irreps(13) == {"E1": 1}


# --------------------------------------------------------------------------- enforced nodes
def test_stabilisers():
    plane, axis = d12._stabilisers()
    assert axis == tuple(range(12))                                # C12 fixes the twelve-fold axis
    for j in range(24):
        assert plane[j] == (0, 12 + j % 12)                        # {E, P_{j mod 12}}
    assert d12.stabiliser(Matrix([sp.cos(pi / 7), sp.sin(pi / 7), 0])) == (0,)   # generic direction


def test_enforced_nodes_literal():
    nodes = d12.enforced_nodes()
    assert tuple(nodes) == ("A1", "A2", "B1", "B2", "E1", "E2", "E3", "E4", "E5")
    a2, b1, b2, a1 = nodes["A2"], nodes["B1"], nodes["B2"], nodes["A1"]
    # A2: nodes on all 24 in-plane axis directions
    assert (a2.node_C2prime, a2.node_C2double, a2.n_in_plane_nodes) == (True, True, 24)
    assert a2.node_angles == tuple(j * pi / 12 for j in range(24))
    # B2 on the twelve C2' directions (angles k pi/6), B1 on the twelve C2'' (pi/12 + k pi/6)
    assert (b2.node_C2prime, b2.node_C2double, b2.n_in_plane_nodes) == (True, False, 12)
    assert b2.node_angles == tuple(k * pi / 6 for k in range(12))
    assert (b1.node_C2prime, b1.node_C2double, b1.n_in_plane_nodes) == (False, True, 12)
    assert b1.node_angles == tuple(pi / 12 + k * pi / 6 for k in range(12))
    # A1 and the E_m: no in-plane node enforced
    assert (a1.node_C2prime, a1.node_C2double, a1.n_in_plane_nodes) == (False, False, 0)
    for m in range(1, 6):
        e = nodes[f"E{m}"]
        assert (e.node_C2prime, e.node_C2double, e.n_in_plane_nodes) == (False, False, 0)
        assert (e.trivial_C2prime, e.trivial_C2double) == (1, 1)   # E_m|{E, P_k} contains the trivial character
    # on the twelve-fold axis every irrep except A1, A2 vanishes
    assert {n for n, r in nodes.items() if not r.node_axis} == {"A1", "A2"}
    assert {n for n, r in nodes.items() if r.node_axis} == {"B1", "B2", "E1", "E2", "E3", "E4", "E5"}
    assert nodes["A1"].trivial_axis == 1 and nodes["A2"].trivial_axis == 1


# --------------------------------------------------------------------------- Sym^2 E_m
def test_sym2_and_quartic_counts():
    assert d12.sym2_decomposition(1) == {"A1": 1, "E2": 1}
    assert d12.sym2_decomposition(2) == {"A1": 1, "E4": 1}
    assert d12.sym2_decomposition(3) == {"A1": 1, "B1": 1, "B2": 1}
    assert d12.sym2_decomposition(4) == {"A1": 1, "E4": 1}          # E_8 = E_4
    assert d12.sym2_decomposition(5) == {"A1": 1, "E2": 1}          # E_10 = E_2
    assert [d12.quartic_invariant_count(m) for m in range(1, 6)] == [2, 2, 3, 2, 2]
    for m in range(1, 6):
        assert d12.sym2_character(m)[0] == 3                         # dim Sym^2 E_m = 3


# --------------------------------------------------------------------------- weak coupling on the circle
def test_circle_weak_coupling_ratios():
    for m in (1, 2, 5):
        assert d12.weak_coupling_ratio(m, chiral=False) == Rational(3, 2)
        assert d12.weak_coupling_ratio(m, chiral=True) == 1
    assert d12.circle_average(sp.cos(3 * d12.ANGLE) ** 2) == Rational(1, 2)
    assert d12.circle_ratio(sp.cos(d12.ANGLE) + sp.I * sp.sin(d12.ANGLE)) == 1


# --------------------------------------------------------------------------- formatters
def test_formatters_are_plain_data():
    ct = d12.format_character_table()
    assert ct["classes"] == ["E", "C12", "C6", "C4", "C3", "C12^5", "C2", "C2'", "C2''"]
    assert ct["sizes"] == [1, 2, 2, 2, 2, 2, 1, 6, 6]
    assert ct["rows"][4] == {"irrep": "E1", "chars": ["2", "sqrt(3)", "1", "0", "-1", "-sqrt(3)", "-2", "0", "0"]}
    rows = d12.format_assignment(12)
    assert len(rows) == 13
    assert rows[6] == {"m": 6, "residue": 6, "irreps": "B1 + B2"}
    assert rows[12] == {"m": 12, "residue": 0, "irreps": "A1 + A2"}
    res = d12.format_residue_table()
    assert [r["irreps"] for r in res] == ["A1 + A2", "E1", "E2", "E3", "E4", "E5", "B1 + B2",
                                          "E5", "E4", "E3", "E2", "E1"]
    nt = d12.format_node_table()
    assert nt[1] == {"irrep": "A2", "C2'": "node", "C2''": "node", "axis": "-", "in-plane nodes": 24}
    assert nt[2] == {"irrep": "B1", "C2'": "-", "C2''": "node", "axis": "node", "in-plane nodes": 12}
    assert nt[3] == {"irrep": "B2", "C2'": "node", "C2''": "-", "axis": "node", "in-plane nodes": 12}
    assert nt[4] == {"irrep": "E1", "C2'": "-", "C2''": "-", "axis": "node", "in-plane nodes": 0}
    s2 = d12.format_sym2_table()
    assert [r["Sym^2"] for r in s2] == ["A1 + E2", "A1 + E4", "A1 + B1 + B2", "A1 + E4", "A1 + E2"]
    assert [r["quartic invariants"] for r in s2] == [2, 2, 3, 2, 2]
    summ = d12.summary()
    assert summ["order"] == 24 and summ["weak_coupling"] == {"real": "3/2", "chiral": "1"}
    # everything is JSON-serialisable plain data
    import json
    json.dumps(summ)
