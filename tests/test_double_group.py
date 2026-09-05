"""Section 3.5 of the paper: SU(2) -> 2I branching (Eq. 12) and pair decompositions (Eq. 13)."""
import sympy as sp
from sympy import Rational

from beyond32._exact import PHI, canon, to_K
from beyond32 import double_group as dg
from beyond32 import harmonics as h
from beyond32.groups import (IRREPS_I, SPINOR_IRREPS, binary_icosahedral, character_table_2I,
                             quat_mult, two_i_classes)

# Eq. (12), literally: 2j -> {irrep of 2I: multiplicity}
EQ12 = {1: {"2": 1},
        3: {"4s": 1},
        5: {"6": 1},
        7: {"2'": 1, "6": 1},
        9: {"4s": 1, "6": 1},
        11: {"2": 1, "4s": 1, "6": 1},
        13: {"2": 1, "2'": 1, "4s": 1, "6": 1},
        15: {"4s": 1, "6": 2}}

# Eq. (13), literally: irrep -> (antisymmetric square, symmetric square)
EQ13 = {"2": ({"A": 1}, {"T1": 1}),
        "2'": ({"A": 1}, {"T2": 1}),
        "4s": ({"A": 1, "H": 1}, {"T1": 1, "T2": 1, "G": 1}),
        "6": ({"A": 1, "G": 1, "H": 2}, {"T1": 2, "T2": 2, "G": 1, "H": 1})}

# Table 3 (SO(3) -> I), reached here through the 2I character table at integer j = l
TABLE3 = {0: {"A": 1}, 1: {"T1": 1}, 2: {"H": 1}, 3: {"T2": 1, "G": 1}, 4: {"G": 1, "H": 1},
          5: {"T1": 1, "T2": 1, "H": 1}, 6: {"A": 1, "T1": 1, "G": 1, "H": 1}}


def test_class_cos_half_matches_table_2_labels():
    phi = canon(PHI)
    ws = dg.class_cos_half()
    assert ws == (1, canon(phi / 2), Rational(1, 2), canon((phi - 1) / 2), 0,
                  canon(-(phi - 1) / 2), Rational(-1, 2), canon(-phi / 2), -1)
    assert character_table_2I().classes == ("0", "72", "120", "144", "180", "216", "240", "288", "360")


def test_su2_characters_dimension_and_central_element():
    for tj in range(0, 16):
        chars = dg.su2_characters(tj)
        assert chars[0] == tj + 1
        assert chars[-1] == (-1) ** tj * (tj + 1)        # -1 in SU(2) acts as (-1)^{2j}


def test_eq12_half_integer_branching_literal():
    T = character_table_2I()
    for tj, dec in EQ12.items():
        got = dg.su2_branching(tj)
        assert got == dec, (tj, got)
        assert dg.dimension(got) == tj + 1
        assert set(got) <= set(SPINOR_IRREPS)          # half-integer spin: spinor irreps only
        assert not dg.is_vector(got)
        assert all(T.chars[k][-1] == -T.dim(k) for k in got)


def test_eq12_text():
    assert dg.format_decomposition(dg.su2_branching(11)) == "2 + 4s + 6"
    assert dg.format_decomposition(dg.su2_branching(15)) == "4s + 2*6"
    assert dg.format_decomposition(dg.su2_branching(7)) == "2' + 6"
    assert dg.eq12_lines() == ["j = 1/2: 2", "j = 3/2: 4s", "j = 5/2: 6", "j = 7/2: 2' + 6",
                               "j = 9/2: 4s + 6", "j = 11/2: 2 + 4s + 6",
                               "j = 13/2: 2 + 2' + 4s + 6", "j = 15/2: 4s + 2*6"]
    rows = dg.branching_table()
    assert [r["two_j"] for r in rows] == list(range(1, 16, 2))
    assert rows[5] == {"two_j": 11, "j": "11/2", "dim": 12, "irreps": {"2": 1, "4s": 1, "6": 1},
                       "text": "2 + 4s + 6"}


def test_integer_spin_agrees_with_harmonics_branching():
    for l, dec in TABLE3.items():
        got = dg.su2_branching(2 * l)
        assert got == dec, (l, got)
        assert got == h.branching(l), l                    # explicit projectors
        assert got == h.branching_by_characters(l), l      # character formula (3)
        assert dg.is_vector(got) and dg.dimension(got) == 2 * l + 1


def test_square_class_map_matches_quaternion_squaring():
    sq = dg.square_class_map()
    assert sq == (0, 3, 6, 7, 8, 7, 6, 3, 0)
    ws = [to_K(w) for w in dg.class_cos_half()]
    qs = binary_icosahedral()
    for k, (_, w, idx) in enumerate(two_i_classes()):
        assert to_K(w) == ws[k]
        for i in idx:
            q2 = quat_mult(qs[i], qs[i])
            assert q2[0] == ws[sq[k]], (k, i)      # Re(q^2) = 2 w^2 - 1 lands in class sq[k]


def test_eq13_pair_decompositions_literal():
    T = character_table_2I()
    for irrep, (anti, sym) in EQ13.items():
        d = T.dim(irrep)
        a = dg.pair_decomposition(irrep, "antisym")
        s = dg.pair_decomposition(irrep, "sym")
        assert a == anti, (irrep, a)
        assert s == sym, (irrep, s)
        assert dg.dimension(a) == d * (d - 1) // 2
        assert dg.dimension(s) == d * (d + 1) // 2
        assert dg.is_vector(a) and dg.is_vector(s)
        assert dg.pair_decompositions(irrep) == (anti, sym)


def test_pair_characters_add_up_to_the_square():
    T = character_table_2I()
    for irrep in T.irreps:
        chi = T.chars[irrep]
        a = dg.pair_character(irrep, "antisym")
        s = dg.pair_character(irrep, "sym")
        assert tuple(canon(x + y) for x, y in zip(a, s)) == tuple(canon(c * c) for c in chi)
        assert a[0] == T.dim(irrep) * (T.dim(irrep) - 1) // 2
        assert s[0] == T.dim(irrep) * (T.dim(irrep) + 1) // 2
    # spinor squares: character of -1 is +dim (vector), never -dim
    for irrep in SPINOR_IRREPS:
        assert dg.pair_character(irrep, "sym")[-1] > 0
        assert dg.pair_character(irrep, "antisym")[-1] > 0


def test_sym2_of_vector_irreps_matches_quartic_invariant_count():
    # Sym^2 of the vector irreps (legacy chi_class_sq; pinned again in test_gl)
    assert dg.pair_decomposition("T1", "sym") == {"A": 1, "H": 1}
    assert dg.pair_decomposition("T2", "sym") == {"A": 1, "H": 1}
    assert dg.pair_decomposition("G", "sym") == {"A": 1, "G": 1, "H": 1}
    assert dg.pair_decomposition("H", "sym") == {"A": 1, "G": 1, "H": 2}
    assert dg.pair_decomposition("T1", "antisym") == {"T1": 1}
    assert dg.pair_decomposition("A", "antisym") == {}
    assert dg.pair_decomposition("A", "sym") == {"A": 1}


def test_eq13_text_and_table():
    assert dg.eq13_lines() == ["2 x 2 = (A)_a + (T1)_s",
                               "2' x 2' = (A)_a + (T2)_s",
                               "4s x 4s = (A + H)_a + (T1 + T2 + G)_s",
                               "6 x 6 = (A + G + 2*H)_a + (2*T1 + 2*T2 + G + H)_s"]
    rows = dg.pair_table()
    assert [r["irrep"] for r in rows] == list(SPINOR_IRREPS)
    assert rows[3]["dim"] == 6
    assert rows[3]["antisym"] == {"A": 1, "G": 1, "H": 2}
    assert rows[3]["sym_text"] == "2*T1 + 2*T2 + G + H"
    assert dg.format_decomposition({"H": 2}, times="\\cdot ") == "2\\cdot H"
    s = dg.summary()
    assert set(s) == {"eq12", "eq13", "square_class_map", "cos_half"}
    assert len(s["eq12"]) == 8 and len(s["eq13"]) == 4


def test_bad_arguments():
    import pytest
    with pytest.raises(ValueError):
        dg.pair_decomposition("6", "mixed")
    with pytest.raises(ValueError):
        dg.pair_decomposition("7", "sym")
    with pytest.raises(ValueError):
        dg.su2_characters(-1)
