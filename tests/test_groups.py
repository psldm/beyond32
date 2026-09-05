import sympy as sp
from sympy import Matrix, S, sqrt

from beyond32._exact import PHI, canon, galois
from beyond32 import groups as g


def test_binary_icosahedral_has_120_unit_quaternions():
    qs = g.binary_icosahedral()
    assert len(qs) == 120
    assert len(set(qs)) == 120
    for q in qs:
        assert canon(sum(c * c for c in q)) == 1


def test_2I_closed_under_multiplication():
    assert g.check_closure_2I()


def test_I_has_60_exact_rotations():
    G = g.icosahedral_group()
    assert G.order == 60
    for r in G.rotations:
        M = r.matrix
        assert (M.T * M).applyfunc(canon) == sp.eye(3)
        assert canon(M.det()) == 1
    assert len(set(g._key(r.matrix) for r in G.rotations)) == 60


def test_class_sizes_of_I():
    assert g.class_sizes_I() == (1, 12, 12, 20, 15)
    G = g.icosahedral_group()
    for cls, order in [("E", 1), ("C5", 5), ("C5^2", 5), ("C3", 3), ("C2", 2)]:
        for i in G.classes[cls]:
            assert G.rotations[i].order == order


def test_multiplication_table_is_a_group():
    G = g.icosahedral_group()
    assert G.mult[0] == tuple(range(60))
    for i in range(60):
        assert G.mult[i][G.inverse[i]] == 0
        assert sorted(G.mult[i]) == list(range(60))
    # matrix product agrees with the table on a sample
    for i in range(0, 60, 7):
        for j in range(0, 60, 11):
            P = (G.rotations[i].matrix * G.rotations[j].matrix).applyfunc(canon)
            assert G.index_of(P) == G.mult[i][j]


def test_character_table_I():
    T = g.CHAR_I
    phi = canon(PHI)
    assert T.chars["A"] == (1, 1, 1, 1, 1)
    assert T.chars["T1"] == (3, phi, canon(1 - PHI), 0, -1)
    assert T.chars["T2"] == (3, canon(1 - PHI), phi, 0, -1)
    assert T.chars["G"] == (4, -1, -1, 1, 0)
    assert T.chars["H"] == (5, 0, 0, -1, 1)
    assert T.is_orthonormal()
    assert sum(T.dim(n) ** 2 for n in T.irreps) == 60


def test_character_table_2I():
    T = g.character_table_2I()
    assert T.classes == ("0", "72", "120", "144", "180", "216", "240", "288", "360")
    assert T.sizes == (1, 12, 20, 12, 30, 12, 20, 12, 1)
    assert sorted(T.dim(n) for n in T.irreps) == [1, 2, 2, 3, 3, 4, 4, 5, 6]
    assert tuple(T.irreps) == ("A", "T1", "T2", "G", "H", "2", "2'", "4s", "6")
    assert T.is_orthonormal()
    assert sum(T.dim(n) ** 2 for n in T.irreps) == 120
    phi = canon(PHI)
    one_m_phi = canon(1 - PHI)
    # Table 2 of the paper, literally
    assert T.chars["A"] == (1,) * 9
    assert T.chars["T1"] == (3, phi, 0, one_m_phi, -1, one_m_phi, 0, phi, 3)
    assert T.chars["T2"] == (3, one_m_phi, 0, phi, -1, phi, 0, one_m_phi, 3)
    assert T.chars["G"] == (4, -1, 1, -1, 0, -1, 1, -1, 4)
    assert T.chars["H"] == (5, 0, -1, 0, 1, 0, -1, 0, 5)
    assert T.chars["2"] == (2, phi, 1, canon(PHI - 1), 0, one_m_phi, -1, canon(-PHI), -2)
    assert T.chars["2'"] == (2, one_m_phi, 1, canon(-PHI), 0, phi, -1, canon(PHI - 1), -2)
    assert T.chars["4s"] == (4, 1, -1, -1, 0, 1, 1, -1, -4)
    assert T.chars["6"] == (6, -1, 0, 1, 0, -1, 0, 1, -6)
    # spinor irreps: character -dim on -1; vector irreps: +dim
    for n in g.SPINOR_IRREPS:
        assert T.chars[n][-1] == -T.dim(n)
    for n in g.IRREPS_I:
        assert T.chars[n][-1] == T.dim(n)
    # 2 x 2' = G
    assert tuple(canon(a * b) for a, b in zip(T.chars["2"], T.chars["2'"])) == T.chars["G"]


def test_2I_class_sizes_from_quaternions():
    cl = g.two_i_classes()
    assert [c[0] for c in cl] == list(g.ANGLES_2I)
    assert [len(c[2]) for c in cl] == [1, 12, 20, 12, 30, 12, 20, 12, 1]


def test_character_table_Ih():
    T = g.character_table_Ih()
    assert T.order == 120 and T.is_orthonormal()
    assert T.chars["T1u"][5:] == tuple(-c for c in g.CHAR_I.chars["T1"])


def test_axes_in_paper_orientation():
    phi = canon(PHI)
    five = {tuple(a) for a in g.fivefold_axes()}
    expected = set()
    for v in [(phi, 1, 0), (phi, -1, 0), (0, phi, 1), (0, phi, -1), (1, 0, phi), (-1, 0, phi)]:
        expected.add(tuple(g.axis_canonical(v)))
    assert len(five) == 6 and five == expected
    three = {tuple(a) for a in g.threefold_axes()}
    assert len(three) == 10
    assert tuple(g.axis_canonical((1, 1, 1))) in three
    two = {tuple(a) for a in g.twofold_axes()}
    assert len(two) == 15
    for e in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        assert tuple(Matrix(e)) in two


def test_so3_characters_match_dimension_and_trace():
    G = g.icosahedral_group()
    for l in range(0, 7):
        chars = g.so3_characters(l)
        assert chars[0] == 2 * l + 1
    # l = 1 is the vector representation: characters are the traces of the matrices
    for cls in g.CLASSES_I:
        i = G.classes[cls][0]
        assert canon(G.rotations[i].matrix.trace()) == g.so3_character(1, cls)


def test_rotation_about_and_subgroups():
    r5 = g.rotation_about((PHI, 1, 0), "C5")
    r3 = g.rotation_about((1, 1, 1), "C3")
    r2z = g.rotation_about((0, 0, 1), "C2")
    r2x = g.rotation_about((1, 0, 0), "C2")
    assert r5.order == 5 and r3.order == 3 and r2z.order == 2
    assert len(g.generate_subgroup([r5.index])) == 5
    assert len(g.generate_subgroup([r3.index])) == 3
    assert len(g.generate_subgroup([r2z.index, r2x.index])) == 4
    assert len(g.generate_subgroup([r2z.index, r2x.index, r3.index])) == 12
    assert len(g.generate_subgroup([r5.index, r3.index])) == 60
    # the C2 about z acts as diag(-1,-1,1)
    assert r2z.matrix == sp.diag(-1, -1, 1)
    # the C3 about (1,1,1) is the cyclic permutation x->y->z->x
    assert r3.matrix * Matrix([1, 0, 0]) in (Matrix([0, 1, 0]), Matrix([0, 0, 1]))
