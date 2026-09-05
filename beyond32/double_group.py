"""Spin-orbit coupling: SU(2) -> 2I branching and pair decompositions (Section 3.5).

With spin-orbit coupling a single quasiparticle carries an irrep of the double group
2I in SU(2) (the binary icosahedral group, Table 2 of the paper).  This module
reproduces

* Eq. (12): the restriction to 2I of the SU(2) irrep D^j of half-integer spin
  j = 1/2, 3/2, ..., 15/2 (``su2_branching``).  For integer j = l the same function
  gives the SO(3) -> I branching of the degree-l spherical harmonics (Table 3, Eq. 3),
  which ``harmonics.branching`` obtains independently by explicit projection.
* Eq. (13): a pair of quasiparticles from the same spinor irrep Gamma in {2, 2', 4s, 6}
  decomposes into the antisymmetric ("singlet-like") and the symmetric
  ("triplet-like") square, both of which contain vector irreps of I only
  (``pair_decomposition``).

Method
------
The nine classes of 2I are labelled by the SU(2) rotation angle alpha = 0, 72, 120, 144,
180, 216, 240, 288, 360 degrees; on the class with w = cos(alpha/2) the character of
D^j is sin((2j+1) alpha/2)/sin(alpha/2) = U_{2j}(w), the Chebyshev polynomial of the
second kind (``groups.su2_character``).  Multiplicities are the inner products with
the rows of the character table of 2I (``groups.character_table_2I``).  The
characters of the two squares of a representation with character chi are

    chi_{Alt^2}(g) = (chi(g)^2 - chi(g^2)) / 2,      chi_{Sym^2}(g) = (chi(g)^2 + chi(g^2)) / 2,

where g^2 lies in the class with cos(alpha) = 2 w^2 - 1; that class is found by exact
comparison in Q(sqrt5).  Everything here is exact; nothing is tabulated.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import sympy as sp
from sympy import Rational

from ._exact import K_ONE, canon, from_ab, to_K
from .groups import (ANGLES_2I, IRREPS_2I, IRREPS_I, SPINOR_IRREPS, character_table_2I,
                     su2_character, two_i_classes)

PARTS = ("antisym", "sym")
_SIGN = {"antisym": -1, "sym": +1}
HALF_INTEGER_TWO_J = tuple(range(1, 16, 2))         # 2j for j = 1/2, ..., 15/2 (Eq. 12)


# --------------------------------------------------------------------------- classes of 2I
@lru_cache(maxsize=None)
def class_cos_half() -> Tuple[sp.Expr, ...]:
    """w = cos(alpha/2) of the nine classes of 2I, in the order of
    ``character_table_2I().classes`` (angles 0, 72, ..., 360 degrees):
    1, phi/2, 1/2, (phi-1)/2, 0, -(phi-1)/2, -1/2, -phi/2, -1.

    Taken from the quaternion real parts of the classes (``groups.two_i_classes``) and
    checked against the class labels of the character table."""
    cl = two_i_classes()
    labels = character_table_2I().classes
    assert tuple(str(angle) for angle, _, _ in cl) == labels == tuple(str(a) for a in ANGLES_2I)
    return tuple(w for _, w, _ in cl)


@lru_cache(maxsize=None)
def square_class_map() -> Tuple[int, ...]:
    """For every class C of 2I (by position), the position of the class containing g^2
    for g in C.  The lift of g has real part w = cos(alpha/2); its square has real part
    w^2 - |v|^2 = 2 w^2 - 1 (exact in Q(sqrt5)), which identifies the class uniquely
    because the classes of 2I are separated by w."""
    ws = [to_K(w) for w in class_cos_half()]
    two = from_ab(2)
    out = []
    for w in ws:
        w2 = two * w * w - K_ONE
        matches = [i for i, v in enumerate(ws) if v == w2]
        if len(matches) != 1:
            raise ValueError(f"class of g^2 not unique for w = {w}: {matches}")
        out.append(matches[0])
    return tuple(out)


# --------------------------------------------------------------------------- SU(2) -> 2I
def su2_characters(two_j: int) -> Tuple[sp.Expr, ...]:
    """Character of the SU(2) irrep D^j, j = two_j/2 (dimension 2j+1 = two_j+1), on the
    nine classes of 2I: U_{2j}(cos(alpha/2))."""
    if two_j < 0:
        raise ValueError("two_j must be a non-negative integer")
    return tuple(su2_character(two_j, w) for w in class_cos_half())


def su2_branching(two_j: int) -> Dict[str, int]:
    """SU(2) -> 2I branching of D^j, j = two_j/2: {irrep of 2I: multiplicity}, nonzero
    entries only, in the order A, T1, T2, G, H, 2, 2', 4s, 6.

    two_j odd reproduces Eq. (12) (j = 1/2: 2; 3/2: 4s; 5/2: 6; 7/2: 2' + 6; ...);
    two_j = 2l even reproduces the SO(3) -> I branching of Table 3."""
    return character_table_2I().decompose(su2_characters(two_j))


def dimension(dec: Dict[str, int]) -> int:
    """sum_Gamma m_Gamma dim(Gamma) of a decomposition into irreps of 2I."""
    T = character_table_2I()
    return sum(m * T.dim(k) for k, m in dec.items())


def is_vector(dec: Dict[str, int]) -> bool:
    """True if only vector irreps (A, T1, T2, G, H; those on which -1 acts trivially) occur."""
    return all(k in IRREPS_I for k in dec)


# --------------------------------------------------------------------------- pairs
def pair_character(irrep: str, part: str) -> Tuple[sp.Expr, ...]:
    """Character of the antisymmetric ('antisym', Alt^2) or symmetric ('sym', Sym^2)
    square of an irrep of 2I on the nine classes: (chi(g)^2 -+ chi(g^2)) / 2."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    if irrep not in IRREPS_2I:
        raise ValueError(f"unknown irrep of 2I: {irrep!r}")
    chi = character_table_2I().chars[irrep]
    sq = square_class_map()
    sign = _SIGN[part]
    return tuple(canon((chi[i] ** 2 + sign * chi[sq[i]]) / 2) for i in range(len(chi)))


def pair_decomposition(irrep: str, part: str) -> Dict[str, int]:
    """Decomposition of the antisymmetric ('antisym', singlet-like) or symmetric
    ('sym', triplet-like) square of an irrep of 2I into irreps of 2I (Eq. 13 for the
    spinor irreps 2, 2', 4s, 6).  For the vector irreps T1, T2, G, H and 'sym' this is
    the Sym^2 decomposition that counts the quartic GL invariants (Section 5).

    For a spinor irrep the element -1 acts as (-1)^2 = +1 on both squares, so only
    vector irreps appear (checked; see ``is_vector``)."""
    dec = character_table_2I().decompose(pair_character(irrep, part))
    if irrep in SPINOR_IRREPS and not is_vector(dec):
        raise AssertionError(f"non-vector irrep in {irrep} x {irrep} ({part}): {dec}")
    return dec


def pair_decompositions(irrep: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(antisymmetric, symmetric) squares of an irrep, see ``pair_decomposition``."""
    return pair_decomposition(irrep, "antisym"), pair_decomposition(irrep, "sym")


# --------------------------------------------------------------------------- formatting
def spin_label(two_j: int) -> str:
    """'1/2', '3/2', ... for odd two_j; '0', '1', ... for even two_j."""
    j = Rational(two_j, 2)
    return str(j)


def format_decomposition(dec: Dict[str, int], times: str = "*", plus: str = " + ") -> str:
    """'2 + 4s + 6', '4s + 2*6', 'A + G + 2*H' (multiplicities > 1 written as m*Gamma)."""
    if not dec:
        return "0"
    return plus.join(f"{m}{times}{k}" if m != 1 else k for k, m in dec.items() if m)


def branching_table(two_j_values: Sequence[int] = HALF_INTEGER_TWO_J,
                    times: str = "*") -> List[Dict[str, object]]:
    """Rows of Eq. (12) as plain data for a LaTeX emitter:
    {'two_j': 11, 'j': '11/2', 'dim': 12, 'irreps': {'2': 1, '4s': 1, '6': 1},
     'text': '2 + 4s + 6'}."""
    rows = []
    for tj in two_j_values:
        dec = su2_branching(tj)
        rows.append({"two_j": int(tj), "j": spin_label(tj), "dim": tj + 1,
                     "irreps": dict(dec), "text": format_decomposition(dec, times)})
    return rows


def pair_table(irreps: Sequence[str] = SPINOR_IRREPS, times: str = "*") -> List[Dict[str, object]]:
    """Rows of Eq. (13) as plain data:
    {'irrep': '6', 'dim': 6, 'antisym': {'A': 1, 'G': 1, 'H': 2}, 'sym': {...},
     'antisym_text': 'A + G + 2*H', 'sym_text': '2*T1 + 2*T2 + G + H',
     'text': '6 x 6 = (A + G + 2*H)_a + (2*T1 + 2*T2 + G + H)_s'}."""
    T = character_table_2I()
    rows = []
    for irrep in irreps:
        a, s = pair_decompositions(irrep)
        a_text, s_text = format_decomposition(a, times), format_decomposition(s, times)
        rows.append({"irrep": irrep, "dim": T.dim(irrep), "antisym": dict(a), "sym": dict(s),
                     "antisym_text": a_text, "sym_text": s_text,
                     "text": f"{irrep} x {irrep} = ({a_text})_a + ({s_text})_s"})
    return rows


def eq12_lines(times: str = "*") -> List[str]:
    """Eq. (12) as text lines 'j = 1/2: 2', ..., 'j = 15/2: 4s + 2*6'."""
    return [f"j = {r['j']}: {r['text']}" for r in branching_table(times=times)]


def eq13_lines(times: str = "*") -> List[str]:
    """Eq. (13) as text lines '2 x 2 = (A)_a + (T1)_s', ..."""
    return [r["text"] for r in pair_table(times=times)]


def summary(times: str = "*") -> Dict[str, object]:
    """Everything of Section 3.5 as plain Python data: {'eq12': [...rows...],
    'eq13': [...rows...], 'square_class_map': [...], 'cos_half': [...str...]}."""
    return {"eq12": branching_table(times=times), "eq13": pair_table(times=times),
            "square_class_map": list(square_class_map()),
            "cos_half": [str(w) for w in class_cos_half()]}
