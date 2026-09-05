"""The icosahedral groups: 2I, I and I_h, their classes and character tables.

Conventions (Section 2 of the paper)
------------------------------------
* The binary icosahedral group 2I is the set of 120 unit quaternions
  {+-1, +-i, +-j, +-k, (+-1 +-i +-j +-k)/2} together with the 96 even permutations of
  (+-phi, +-1, +-1/phi, 0)/2.  The rotation group I = 2I/{+-1} (60 exact matrices)
  then has its two-fold axes along x, y, z, its three-fold axes along (+-1, +-1, +-1)
  and its six five-fold axes along the cyclic permutations of (+-phi, +-1, 0).
* Classes of I: E, 12 C5, 12 C5^2, 20 C3, 15 C2 (in this order).
* Classes of 2I are labelled by the SU(2) rotation angle 0, 72, 120, 144, 180, 216,
  240, 288, 360 degrees (the element -1 is 360 degrees).
* Irreps of I: A, T1, T2, G, H.  Spinor irreps of 2I: 2, 2', 4s, 6.
* I_h = I x {E, i}; every irrep Gamma lifts to Gamma_g and Gamma_u.

All arithmetic is exact over Q(sqrt5) (see ``_exact``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import sympy as sp
from sympy import Matrix, Rational, S

from ._exact import (K, K_ONE, K_ZERO, PHI, SQRT5, ab, as_float, canon, from_ab, galois,
                     to_K, to_sympy)

IRREPS_I = ("A", "T1", "T2", "G", "H")
CLASSES_I = ("E", "C5", "C5^2", "C3", "C2")
CLASS_SIZES_I = (1, 12, 12, 20, 15)
SPINOR_IRREPS = ("2", "2'", "4s", "6")
IRREPS_2I = IRREPS_I + SPINOR_IRREPS
ANGLES_2I = (0, 72, 120, 144, 180, 216, 240, 288, 360)
CLASS_SIZES_2I = (1, 12, 20, 12, 30, 12, 20, 12, 1)

# cos(theta/2) of the rotation classes of I (theta = rotation angle)
_COS_HALF_I = {"E": S(1), "C5": PHI / 2, "C5^2": (PHI - 1) / 2, "C3": Rational(1, 2), "C2": S(0)}
# cos(alpha/2) for the SU(2) classes of 2I in the order of ANGLES_2I
_COS_HALF_2I = (S(1), PHI / 2, Rational(1, 2), (PHI - 1) / 2, S(0),
                -(PHI - 1) / 2, Rational(-1, 2), -PHI / 2, S(-1))


# --------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Rotation:
    """One element of I: exact matrix, a lift to 2I, class label, order and axis."""
    index: int
    matrix: Matrix                    # exact 3x3 rotation matrix over Q(sqrt5)
    quaternion: Tuple[sp.Expr, ...]   # (w, x, y, z), the lift with cos(alpha/2) >= 0
    cls: str                          # 'E', 'C5', 'C5^2', 'C3', 'C2'
    order: int
    axis: Optional[Matrix]            # exact axis direction (None for E), see axis_canonical


@dataclass(frozen=True)
class CharacterTable:
    """A character table: class labels and sizes, irreps as tuples of characters."""
    name: str
    classes: Tuple[str, ...]
    sizes: Tuple[int, ...]
    chars: Dict[str, Tuple[sp.Expr, ...]]

    @property
    def order(self) -> int:
        return sum(self.sizes)

    @property
    def irreps(self) -> Tuple[str, ...]:
        return tuple(self.chars)

    def dim(self, irrep: str) -> int:
        return int(self.chars[irrep][0])

    def inner(self, u: Sequence, v: Sequence) -> sp.Expr:
        """<u, v> = (1/|G|) sum_C |C| conj(u_C) v_C (characters here are real)."""
        return canon(sum(s * sp.conjugate(a) * b for s, a, b in zip(self.sizes, u, v)) / self.order)

    def is_orthonormal(self) -> bool:
        return all(self.inner(self.chars[a], self.chars[b]) == (1 if a == b else 0)
                   for a in self.chars for b in self.chars)

    def decompose(self, chars: Sequence) -> Dict[str, int]:
        """Multiplicities of the irreps in a (virtual) character; keys with m != 0 only."""
        out = {}
        for name, row in self.chars.items():
            m = self.inner(row, chars)
            if m != 0:
                if not m.is_Integer:
                    raise ValueError(f"non-integral multiplicity {m} for {name}")
                out[name] = int(m)
        return out

    def format(self, dec: Dict[str, int], plus: str = "+") -> str:
        return f" {plus} ".join(f"{m if m > 1 else ''}{k}" for k, m in dec.items())


@dataclass(frozen=True)
class IGroup:
    """The rotation group I with its multiplication table and class structure."""
    rotations: Tuple[Rotation, ...]
    quaternions: Tuple[Tuple[sp.Expr, ...], ...]      # the 120 elements of 2I
    mult: Tuple[Tuple[int, ...], ...]                  # mult[i][j] = index of R_i R_j
    inverse: Tuple[int, ...]
    classes: Dict[str, Tuple[int, ...]]                # class label -> indices
    kmatrices: Tuple[Tuple[Tuple[object, ...], ...], ...] = ()   # the same matrices as field elements

    @property
    def order(self) -> int:
        return len(self.rotations)

    @property
    def matrices(self) -> Tuple[Matrix, ...]:
        return tuple(r.matrix for r in self.rotations)

    def class_of(self, i: int) -> str:
        return self.rotations[i].cls

    def character(self, irrep: str, i: int) -> sp.Expr:
        """chi_irrep(R_i)."""
        return CHAR_I.chars[irrep][CLASSES_I.index(self.rotations[i].cls)]

    def index_of(self, M: Matrix) -> int:
        return _matrix_index()[_key(M)]

    def as_numpy(self):
        """The 60 rotation matrices as a float array of shape (60, 3, 3)."""
        import numpy as np

        return np.array([[[as_float(e) for e in row] for row in r.matrix.tolist()]
                         for r in self.rotations], dtype=float)


# --------------------------------------------------------------------------- 2I
def _even_perms(v):
    out = []
    for p in itertools.permutations(range(4)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            out.append(tuple(v[p[i]] for i in range(4)))
    return out


def _sort_key(q):
    return tuple((Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q))) for a, b in (ab(c) for c in q))


@lru_cache(maxsize=None)
def _kquats() -> Tuple[Tuple, ...]:
    """The 120 unit quaternions of 2I as tuples of field elements, in a fixed order."""
    half = from_ab(Fraction(1, 2))
    phi = from_ab(Fraction(1, 2), Fraction(1, 2))
    phi_inv = from_ab(Fraction(-1, 2), Fraction(1, 2))
    one, zero = K_ONE, K_ZERO
    quats = set()
    for i in range(4):
        for s in (1, -1):
            q = [zero] * 4
            q[i] = one if s == 1 else -one
            quats.add(tuple(q))
    for signs in itertools.product((1, -1), repeat=4):
        quats.add(tuple(half if s == 1 else -half for s in signs))
    base = [phi, one, phi_inv, zero]
    for signs in itertools.product((1, -1), repeat=3):
        v = [base[0] if signs[0] == 1 else -base[0],
             base[1] if signs[1] == 1 else -base[1],
             base[2] if signs[2] == 1 else -base[2], zero]
        for p in _even_perms(v):
            quats.add(tuple(c * half for c in p))
    quats = sorted(quats, key=_sort_key)
    assert len(quats) == 120
    return tuple(quats)


def quat_mult(a, b):
    """Hamilton product of two quaternions (tuples of field elements or sympy numbers)."""
    a0, a1, a2, a3 = (to_K(c) for c in a)
    b0, b1, b2, b3 = (to_K(c) for c in b)
    return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


def binary_icosahedral() -> Tuple[Tuple[sp.Expr, ...], ...]:
    """The 120 unit quaternions (w, x, y, z) of 2I as tuples of exact sympy numbers."""
    return tuple(tuple(to_sympy(c) for c in q) for q in _kquats())


def check_closure_2I() -> bool:
    """Full closure check: all 120 x 120 products lie in the set."""
    qs = _kquats()
    qset = set(qs)
    return all(quat_mult(a, b) in qset for a in qs for b in qs)


def _rot_k(q):
    """Rotation matrix (as a 3x3 list of field elements) of the unit quaternion q."""
    w, a, b, c = q
    two = from_ab(2)
    return [[w * w + a * a - b * b - c * c, two * (a * b - w * c), two * (a * c + w * b)],
            [two * (a * b + w * c), w * w - a * a + b * b - c * c, two * (b * c - w * a)],
            [two * (a * c - w * b), two * (b * c + w * a), w * w - a * a - b * b + c * c]]


def _key(M) -> tuple:
    if isinstance(M, Matrix):
        return tuple(to_K(e) for e in M)
    return tuple(e for row in M for e in row)


def _positive_lift(q):
    """Of q and -q, the one with w > 0 (or w == 0 and first nonzero component > 0)."""
    for c in q:
        if c != K_ZERO:
            return q if as_float(c) > 0 else tuple(-x for x in q)
    raise ValueError


def _class_of_w(w) -> str:
    """Class of I of a rotation whose lift has real part w = cos(alpha/2)."""
    a, b = ab(w)
    val = (abs(Fraction(int(a.p), int(a.q))), Fraction(int(b.p), int(b.q)) * (1 if a >= 0 else -1))
    table = {(Fraction(1), Fraction(0)): "E",
             (Fraction(1, 4), Fraction(1, 4)): "C5",       # phi/2 = 1/4 + sqrt5/4
             (Fraction(1, 4), Fraction(-1, 4)): "C5^2",    # (phi-1)/2 = -1/4 + sqrt5/4
             (Fraction(1, 2), Fraction(0)): "C3",
             (Fraction(0), Fraction(0)): "C2"}
    # a == 0 for C2, handle sign convention
    if a == 0 and b == 0:
        return "C2"
    if a < 0:
        val = (abs(Fraction(int(a.p), int(a.q))), -Fraction(int(b.p), int(b.q)))
    else:
        val = (Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q)))
    return table[val]


def axis_canonical(v: Sequence) -> Matrix:
    """Scale an axis direction so its smallest nonzero |component| is 1 and its first nonzero
    component is positive.  Five-fold axes then read (phi, +-1, 0) and cyclic permutations,
    three-fold (1, +-1, +-1), coordinate two-fold (1, 0, 0) etc."""
    comps = [to_sympy(c) for c in v]
    nz = [c for c in comps if c != 0]
    smallest = min(nz, key=lambda c: abs(as_float(c)))
    scaled = [canon(c / smallest) for c in comps]
    first = next(c for c in scaled if c != 0)
    if as_float(first) < 0:
        scaled = [canon(-c) for c in scaled]
    return Matrix(scaled)


@lru_cache(maxsize=None)
def icosahedral_group() -> IGroup:
    """The rotation group I: 60 exact matrices, classes, multiplication table, axes."""
    qs = _kquats()
    qindex = {q: i for i, q in enumerate(qs)}
    rot_index: Dict[tuple, int] = {}
    rotations = []
    rep_quat = []
    kmats = []
    for q in qs:
        R = _rot_k(q)
        key = _key(R)
        if key in rot_index:
            continue
        idx = len(rotations)
        rot_index[key] = idx
        kmats.append(tuple(tuple(row) for row in R))
        lift = _positive_lift(q)
        w = lift[0]
        cls = _class_of_w(w)
        order = {"E": 1, "C5": 5, "C5^2": 5, "C3": 3, "C2": 2}[cls]
        axis = None if cls == "E" else axis_canonical(lift[1:])
        M = Matrix(3, 3, [to_sympy(e) for row in R for e in row])
        rotations.append(Rotation(idx, M, tuple(to_sympy(c) for c in lift), cls, order, axis))
        rep_quat.append(lift)
    assert len(rotations) == 60
    # multiplication table via quaternions (exact); q_i q_j = +- q_k
    quat_to_rot = {}
    for q in qs:
        quat_to_rot[q] = rot_index[_key(_rot_k(q))]
    mult = []
    for qi in rep_quat:
        row = []
        for qj in rep_quat:
            row.append(quat_to_rot[quat_mult(qi, qj)])
        mult.append(tuple(row))
    mult = tuple(mult)
    inverse = tuple(next(j for j in range(60) if mult[i][j] == 0) for i in range(60))
    classes = {c: tuple(r.index for r in rotations if r.cls == c) for c in CLASSES_I}
    return IGroup(tuple(rotations), tuple(tuple(to_sympy(c) for c in q) for q in qs),
                  mult, inverse, classes, tuple(kmats))


@lru_cache(maxsize=None)
def _matrix_index() -> Dict[tuple, int]:
    return {_key(r.matrix): r.index for r in icosahedral_group().rotations}


def icosahedral_rotations() -> Tuple[Matrix, ...]:
    """The 60 exact rotation matrices of I."""
    return icosahedral_group().matrices


def class_sizes_I() -> Tuple[int, ...]:
    G = icosahedral_group()
    return tuple(len(G.classes[c]) for c in CLASSES_I)


# --------------------------------------------------------------------------- characters
def su2_character(two_j: int, w) -> sp.Expr:
    """Character of the SU(2) irrep of spin j = two_j/2 on the class with cos(alpha/2) = w:
    sin((2j+1) alpha/2)/sin(alpha/2) = U_{2j}(w), the Chebyshev polynomial of the second kind."""
    return canon(sp.chebyshevu(two_j, w))


def so3_character(l: int, cls: str) -> sp.Expr:
    """Character of the SO(3) irrep of degree l on the class cls of I (theta the rotation
    angle): sin((l+1/2) theta)/sin(theta/2) = U_{2l}(cos(theta/2))."""
    return su2_character(2 * l, _COS_HALF_I[cls])


def so3_characters(l: int) -> Tuple[sp.Expr, ...]:
    return tuple(so3_character(l, c) for c in CLASSES_I)


CHAR_I = CharacterTable(
    "I", CLASSES_I, CLASS_SIZES_I,
    {"A": (S(1), S(1), S(1), S(1), S(1)),
     "T1": (S(3), canon(PHI), canon(1 - PHI), S(0), S(-1)),
     "T2": (S(3), canon(1 - PHI), canon(PHI), S(0), S(-1)),
     "G": (S(4), S(-1), S(-1), S(1), S(0)),
     "H": (S(5), S(0), S(0), S(-1), S(1))})


def character_table_I() -> CharacterTable:
    """Character table of I (Table 1 of the paper)."""
    return CHAR_I


@lru_cache(maxsize=None)
def two_i_classes() -> Tuple[Tuple[int, sp.Expr, Tuple[int, ...]], ...]:
    """Classes of 2I: (angle in degrees, w = cos(alpha/2), indices into binary_icosahedral())."""
    qs = _kquats()
    out = []
    for angle, w in zip(ANGLES_2I, _COS_HALF_2I):
        wk = to_K(w)
        idx = tuple(i for i, q in enumerate(qs) if q[0] == wk)
        out.append((angle, canon(w), idx))
    assert sum(len(t[2]) for t in out) == 120
    return tuple(out)


@lru_cache(maxsize=None)
def character_table_2I() -> CharacterTable:
    """Character table of 2I (Table 2 of the paper), classes labelled by the SU(2) angle.

    Built as in the legacy code: the SU(2) characters U_{2j}(w) for 2j = 0..5 give the
    irreps 1 (=A), 2, T1, 4s, H, 6; the Galois conjugates of 2 and T1 give 2' and T2; and
    G = 2 x 2'."""
    ws = _COS_HALF_2I
    rows = {}
    rows["A"] = tuple(su2_character(0, w) for w in ws)
    rows["2"] = tuple(su2_character(1, w) for w in ws)
    rows["T1"] = tuple(su2_character(2, w) for w in ws)
    rows["4s"] = tuple(su2_character(3, w) for w in ws)
    rows["H"] = tuple(su2_character(4, w) for w in ws)
    rows["6"] = tuple(su2_character(5, w) for w in ws)
    rows["2'"] = tuple(galois(c) for c in rows["2"])
    rows["T2"] = tuple(galois(c) for c in rows["T1"])
    rows["G"] = tuple(canon(a * b) for a, b in zip(rows["2"], rows["2'"]))
    chars = {name: rows[name] for name in IRREPS_2I}
    return CharacterTable("2I", tuple(f"{a}" for a in ANGLES_2I), CLASS_SIZES_2I, chars)


# improper classes of I_h in the order (-E, -C5, -C5^2, -C3, -C2): the inversion i, then
# i C5 = S10^7 (class S10^3), i C5^2 = S10^9 (class S10), i C3 = S6^5 (class S6), i C2 = sigma
CLASSES_IH = CLASSES_I + ("i", "S10^3", "S10", "S6", "sigma")


@lru_cache(maxsize=None)
def character_table_Ih() -> CharacterTable:
    """Character table of I_h = I x {E, i}: Gamma_g(iR) = chi(R), Gamma_u(iR) = -chi(R).
    The improper classes are listed as -R for R in E, C5, C5^2, C3, C2 (labels i, S10^3,
    S10, S6, sigma in Schoenflies notation)."""
    chars = {}
    for name, row in CHAR_I.chars.items():
        chars[name + "g"] = row + row
        chars[name + "u"] = row + tuple(-c for c in row)
    return CharacterTable("Ih", CLASSES_IH, CLASS_SIZES_I + CLASS_SIZES_I, chars)


# --------------------------------------------------------------------------- axes
def _axes_of_class(cls: str) -> Tuple[Matrix, ...]:
    G = icosahedral_group()
    seen = []
    for i in G.classes[cls]:
        a = G.rotations[i].axis
        if not any(a == b for b in seen):
            seen.append(a)
    return tuple(seen)


def fivefold_axes() -> Tuple[Matrix, ...]:
    """The six five-fold axes (up to sign): cyclic permutations of (phi, +-1, 0)."""
    return _axes_of_class("C5")


def threefold_axes() -> Tuple[Matrix, ...]:
    """The ten three-fold axes (up to sign): (1, +-1, +-1)."""
    return _axes_of_class("C3")


def twofold_axes() -> Tuple[Matrix, ...]:
    """The fifteen two-fold axes (up to sign); x, y, z among them."""
    return _axes_of_class("C2")


def rotation_about(axis: Sequence, cls: str) -> Rotation:
    """A rotation of the given class about the given axis (exact match of the direction).

    For 'C5' / 'C5^2' the sense is fixed by the lift with cos(alpha/2) > 0, i.e. the
    rotation by +72 / +144 degrees about the oriented axis."""
    target = axis_canonical(axis)
    G = icosahedral_group()
    for i in G.classes[cls]:
        r = G.rotations[i]
        if r.axis == target:
            # orientation: the quaternion vector part is along +axis or -axis
            v = Matrix([to_sympy(c) for c in r.quaternion[1:]])
            if as_float((v.T * target)[0]) > 0:
                return r
    for i in G.classes[cls]:
        r = G.rotations[i]
        if r.axis == target:
            return r
    raise ValueError(f"no {cls} rotation about {list(axis)}")


def generate_subgroup(generators: Sequence[int]) -> Tuple[int, ...]:
    """Indices of the subgroup of I generated by the given rotation indices."""
    G = icosahedral_group()
    elems = {0}
    frontier = [0]
    while frontier:
        new = []
        for a in frontier:
            for g in generators:
                b = G.mult[a][g]
                if b not in elems:
                    elems.add(b)
                    new.append(b)
        frontier = new
    return tuple(sorted(elems))
