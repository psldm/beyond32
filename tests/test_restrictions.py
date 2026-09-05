"""Table 5 (restrictions I -> T, D5, D3, D2) and the cyclic restrictions of Section 7."""
import sympy as sp
from sympy import S

from beyond32._exact import PHI, canon
from beyond32 import restrictions as r
from beyond32.groups import CHAR_I, IRREPS_I, icosahedral_group
from beyond32 import harmonics as h

X, Y, Z = h.X, h.Y, h.Z


# --------------------------------------------------------------------------- the subgroups
def test_subgroups_generated_in_I_have_the_stated_orders_and_class_sizes():
    G = icosahedral_group()
    expected = {"T": (12, ("E", "C3", "C2"), (1, 8, 3)),
                "D5": (10, ("E", "C5", "C5^2", "C2'"), (1, 2, 2, 5)),
                "D3": (6, ("E", "C3", "C2'"), (1, 2, 3)),
                "D2": (4, ("E", "C2x", "C2y", "C2z"), (1, 1, 1, 1))}
    for name, (order, classes, sizes) in expected.items():
        sub = r.subgroup(name)
        assert sub.order == order
        assert sub.table.classes == classes
        assert sub.table.sizes == sizes
        assert sub.table.order == order
        # every element of a subgroup class lies in the I-class it fuses into
        for lab, idx in sub.classes.items():
            assert all(G.class_of(i) == sub.fusion[lab] for i in idx), (name, lab)
        # closed under multiplication
        elems = set(sub.elements)
        assert all(G.mult[a][b] in elems for a in elems for b in elems)


def test_subgroup_character_tables():
    # D5, D3, D2: orthonormal irreducible characters, sum d^2 = |K|
    for name in ("D5", "D3", "D2"):
        T = r.character_table(name)
        assert T.is_orthonormal(), name
        assert sum(T.dim(k) ** 2 for k in T.irreps) == T.order
    # T: A and T irreducible, E = 1E + 2E real of norm 2, all mutually orthogonal
    T = r.character_table("T")
    assert T.irreps == ("A", "E", "T")
    assert T.inner(T.chars["A"], T.chars["A"]) == 1
    assert T.inner(T.chars["T"], T.chars["T"]) == 1
    assert T.inner(T.chars["E"], T.chars["E"]) == 2
    for a in T.irreps:
        for b in T.irreps:
            if a != b:
                assert T.inner(T.chars[a], T.chars[b]) == 0
    # D5 characters use cos 72 = (phi-1)/2 and cos 144 = -phi/2
    D5 = r.character_table("D5")
    assert D5.chars["E1"] == (2, canon(PHI - 1), canon(-PHI), 0)
    assert D5.chars["E2"] == (2, canon(-PHI), canon(PHI - 1), 0)


def test_restricted_characters_are_values_on_the_fused_I_classes():
    phi = canon(PHI)
    assert r.restricted_character("T1", "D5") == (3, phi, canon(1 - PHI), -1)
    assert r.restricted_character("H", "T") == (5, -1, 1)
    assert r.restricted_character("G", "D3") == (4, 1, 0)
    assert r.restricted_character("T2", "D2") == (3, -1, -1, -1)


# --------------------------------------------------------------------------- Table 5
def test_restriction_to_T():
    assert r.restrict("A", "T") == {"A": 1}
    assert r.restrict("T1", "T") == {"T": 1}
    assert r.restrict("T2", "T") == {"T": 1}
    assert r.restrict("G", "T") == {"A": 1, "T": 1}
    assert r.restrict("H", "T") == {"E": 1, "T": 1}


def test_restriction_to_D5():
    assert r.restrict("A", "D5") == {"A1": 1}
    assert r.restrict("T1", "D5") == {"A2": 1, "E1": 1}
    assert r.restrict("T2", "D5") == {"A2": 1, "E2": 1}
    assert r.restrict("G", "D5") == {"E1": 1, "E2": 1}
    assert r.restrict("H", "D5") == {"A1": 1, "E1": 1, "E2": 1}


def test_restriction_to_D3():
    assert r.restrict("A", "D3") == {"A1": 1}
    assert r.restrict("T1", "D3") == {"A2": 1, "E": 1}
    assert r.restrict("T2", "D3") == {"A2": 1, "E": 1}
    assert r.restrict("G", "D3") == {"A1": 1, "A2": 1, "E": 1}
    assert r.restrict("H", "D3") == {"A1": 1, "E": 2}


def test_restriction_to_D2_by_parities():
    assert r.restrict("A", "D2") == {"A": 1}
    assert r.restrict("T1", "D2") == {"B1": 1, "B2": 1, "B3": 1}
    assert r.restrict("T2", "D2") == {"B1": 1, "B2": 1, "B3": 1}
    assert r.restrict("G", "D2") == {"A": 1, "B1": 1, "B2": 1, "B3": 1}
    assert r.restrict("H", "D2") == {"A": 2, "B1": 1, "B2": 1, "B3": 1}
    # the labels of the individual basis functions of Section 3.3
    assert r.d2_labels("A") == {"1": "A"}
    assert r.d2_labels("T1") == {"x": "B3", "y": "B2", "z": "B1"}
    assert r.d2_labels("T2") == {"f_x": "B3", "f_y": "B2", "f_z": "B1"}
    assert r.d2_labels("G") == {"g_x": "B3", "g_y": "B2", "g_z": "B1", "xyz": "A"}
    assert r.d2_labels("H") == {"xy": "B1", "yz": "B3", "zx": "B2", "x^2-y^2": "A", "2z^2-x^2-y^2": "A"}
    # parities directly: z is even under C2z only, xyz under all three
    assert r.d2_parities(Z) == (-1, -1, 1)
    assert r.d2_parities(X * Y * Z) == (1, 1, 1)
    assert r.d2_parities(X) == (1, -1, -1)
    # the coordinate two-folds are diag(1,-1,-1), diag(-1,1,-1), diag(-1,-1,1)
    c2x, c2y, c2z = r.coordinate_twofolds()
    assert c2x.matrix == sp.diag(1, -1, -1)
    assert c2y.matrix == sp.diag(-1, 1, -1)
    assert c2z.matrix == sp.diag(-1, -1, 1)


def test_D2_by_characters_agrees_with_parities():
    for irr in IRREPS_I:
        assert r.d2_is_consistent(irr), irr
        chars = r.restrict_by_characters(irr, "D2")
        # characters force B1 = B2 = B3 = (chi(E) - chi(C2))/4
        nb = (CHAR_I.chars[irr][0] - CHAR_I.chars[irr][4]) / 4
        for b in ("B1", "B2", "B3"):
            assert chars.get(b, 0) == nb, (irr, b)
        assert chars.get("A", 0) == (CHAR_I.chars[irr][0] + 3 * CHAR_I.chars[irr][4]) / 4
    assert r.d2_character_counts("A") == {"A": 1}
    assert r.d2_character_counts("T1") == {"B": 3}
    assert r.d2_character_counts("T2") == {"B": 3}
    assert r.d2_character_counts("G") == {"A": 1, "B": 3}
    assert r.d2_character_counts("H") == {"A": 2, "B": 3}


def test_dimensions_add_up_in_every_restriction():
    for irr in IRREPS_I:
        for name in r.SUBGROUPS:
            T = r.character_table(name)
            dec = r.restrict(irr, name)
            assert sum(m * T.dim(k) for k, m in dec.items()) == CHAR_I.dim(irr), (irr, name)


def test_restriction_table_rows_and_format():
    rows = r.restriction_table_rows()
    assert [row["irrep"] for row in rows] == ["A", "T1", "T2", "G", "H"]
    assert rows[0] == {"irrep": "A", "T": "A", "D5": "A1", "D3": "A1", "D2": "A"}
    assert rows[1] == {"irrep": "T1", "T": "T", "D5": "A2 + E1", "D3": "A2 + E", "D2": "B1 + B2 + B3"}
    assert rows[2] == {"irrep": "T2", "T": "T", "D5": "A2 + E2", "D3": "A2 + E", "D2": "B1 + B2 + B3"}
    assert rows[3] == {"irrep": "G", "T": "A + T", "D5": "E1 + E2", "D3": "A1 + A2 + E",
                       "D2": "A + B1 + B2 + B3"}
    assert rows[4] == {"irrep": "H", "T": "E + T", "D5": "A1 + E1 + E2", "D3": "A1 + 2E",
                       "D2": "2A + B1 + B2 + B3"}
    tab = r.restriction_table()
    assert list(tab) == ["A", "T1", "T2", "G", "H"]
    assert all(list(tab[irr]) == ["T", "D5", "D3", "D2"] for irr in tab)
    data = r.restriction_table_data()
    assert data["columns"] == ["irrep", "T", "D5", "D3", "D2"]
    assert data["rows"] == rows
    assert data["d2_labels"]["G"]["xyz"] == "A"
    assert r.format_decomposition({"A1": 1, "E": 2}) == "A1 + 2E"
    assert r.format_decomposition({"A1": 1, "E": 2}, plus="+") == "A1+2E"


# --------------------------------------------------------------------------- cyclic groups (Section 7)
def test_cyclic_generators_and_powers():
    G = icosahedral_group()
    assert r.cyclic_generator(5).cls == "C5"
    assert r.cyclic_generator(3).cls == "C3"
    assert r.cyclic_generator(2).cls == "C2"
    assert [G.class_of(i) for i in r.cyclic_powers(5)] == ["E", "C5", "C5^2", "C5^2", "C5"]
    assert [G.class_of(i) for i in r.cyclic_powers(3)] == ["E", "C3", "C3"]
    assert [G.class_of(i) for i in r.cyclic_powers(2)] == ["E", "C2"]
    for n in (2, 3, 5):
        pw = r.cyclic_powers(n)
        assert len(set(pw)) == n and pw[0] == 0
        g = r.cyclic_generator(n).index
        assert G.mult[pw[-1]][g] == 0
    # chi_m(g^p) = exp(2 pi i m p / n)
    assert r.cyclic_character(1, 5)[2] == sp.exp(4 * sp.pi * sp.I / 5)
    assert r.cyclic_character(0, 3) == (1, 1, 1)


def test_restriction_to_C5_enforced_node_theorem():
    assert r.restrict_to_cyclic("A", 5) == {0: 1}
    assert r.restrict_to_cyclic("T1", 5) == {0: 1, 1: 1, 4: 1}
    assert r.restrict_to_cyclic("T2", 5) == {0: 1, 2: 1, 3: 1}
    assert r.restrict_to_cyclic("G", 5) == {1: 1, 2: 1, 3: 1, 4: 1}
    assert r.restrict_to_cyclic("H", 5) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert 0 not in r.restrict_to_cyclic("G", 5)
    assert not r.has_trivial_character("G", 5)
    for irr in ("A", "T1", "T2", "H"):
        assert r.has_trivial_character(irr, 5), irr


def test_restriction_to_C3_and_C2_all_contain_the_trivial_character():
    assert r.restrict_to_cyclic("A", 3) == {0: 1}
    assert r.restrict_to_cyclic("T1", 3) == {0: 1, 1: 1, 2: 1}
    assert r.restrict_to_cyclic("T2", 3) == {0: 1, 1: 1, 2: 1}
    assert r.restrict_to_cyclic("G", 3) == {0: 2, 1: 1, 2: 1}
    assert r.restrict_to_cyclic("H", 3) == {0: 1, 1: 2, 2: 2}
    assert r.restrict_to_cyclic("A", 2) == {0: 1}
    assert r.restrict_to_cyclic("T1", 2) == {0: 1, 1: 2}
    assert r.restrict_to_cyclic("T2", 2) == {0: 1, 1: 2}
    assert r.restrict_to_cyclic("G", 2) == {0: 2, 1: 2}
    assert r.restrict_to_cyclic("H", 2) == {0: 3, 1: 2}
    for irr in IRREPS_I:
        for n in (3, 2):
            assert r.has_trivial_character(irr, n), (irr, n)
        for n in (2, 3, 5):
            assert sum(r.restrict_to_cyclic(irr, n).values()) == CHAR_I.dim(irr)


def test_cyclic_table_rows_and_format():
    rows = r.cyclic_table_rows()
    assert [row["irrep"] for row in rows] == ["A", "T1", "T2", "G", "H"]
    g = rows[3]
    assert g["C5"] == "chi_1 + chi_2 + chi_3 + chi_4"
    assert g["C3"] == "2chi_0 + chi_1 + chi_2"
    assert g["C2"] == "2chi_0 + 2chi_1"
    assert (g["node_on_C5"], g["node_on_C3"], g["node_on_C2"]) == ("yes", "no", "no")
    for row in rows[:3] + rows[4:]:
        assert row["node_on_C5"] == "no"
    assert r.format_cyclic({0: 1, 1: 2}) == "chi_0 + 2chi_1"
    tab = r.cyclic_table()
    assert list(tab["H"]) == [5, 3, 2]
