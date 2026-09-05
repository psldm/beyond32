"""Permutation representations of I and I_h on the icosahedral orbit shells (Table 4).

Physics
-------
A shell of atoms (or of "hot spots" on the pseudo-Fermi surface) of an icosahedral
quasicrystal is an orbit of I_h = I x {E, i} acting on directions.  The symmetric
special directions give the four orbit types of Table 4 of the paper:

    orbit   seed direction   polyhedron                stabiliser in I / in I_h
    12      (phi, 1, 0)      icosahedron vertices      C5 / C5v
    20      (1, 1, 1)        dodecahedron vertices     C3 / C3v
    30      (0, 0, 1)        icosidodecahedron         C2 / C2v
    60      (0, 1, 3 phi)    truncated icosahedron     C1 / Cs

The permutation representation of a group on an orbit has the character
chi_perm(g) = #{p in orbit : g p = p}.  The improper element i R of I_h acts on a
direction as -R, so its fixed points are the p with R p = -p.  By Frobenius
reciprocity the multiplicity of an irrep Gamma in the permutation representation is
the number of trivial characters of the stabiliser contained in Gamma restricted to
the stabiliser.  G restricted to C5 contains no trivial character, so G (and Gg, Gu)
is absent from the 12-orbit: a G-channel gap function must vanish on all twelve
five-fold directions -- the enforced-node theorem of the paper in its real-space guise.

Everything is exact over Q(sqrt5).  Orbit points are tuples of exact numbers (direction
vectors, not normalised: sqrt(phi^2 + 1) is not in Q(sqrt5)), fixed points are decided by
exact comparison, and multiplicities come from ``CharacterTable.decompose`` with the
character tables of ``groups``.  The class order of I_h is that of
``groups.character_table_Ih()``: the five proper classes E, C5, C5^2, C3, C2 followed by
the improper classes {-R : R in E}, {-R : R in C5}, ... (labelled i, S10^3, S10, S6, sigma: -C5 is the rotoreflection S10^7, in the class of S10^3).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import sympy as sp
from sympy import S
from sympy.polys.polyclasses import ANP

from ._exact import K_ZERO, PHI, SQRT5, VARPHI, canon, from_ab, phi_form, to_K, to_sympy
from .groups import (CHAR_I, CLASSES_I, IRREPS_I, CharacterTable, IGroup, character_table_Ih,
                     icosahedral_group)

__all__ = ["SHELL_NAMES", "SEEDS", "POLYHEDRA", "IRREPS_IH_ORDER", "Shell", "orbit", "orbit_Ih",
           "is_centrosymmetric", "fixed_point_counts", "permutation_character",
           "permutation_character_Ih", "stabiliser_I", "stabiliser_Ih", "stabiliser_labels",
           "decomposition_I", "decomposition_Ih", "decomposition_by_frobenius", "shell",
           "shells_table", "format_decomposition", "format_shells_table"]

Point = Tuple[sp.Expr, sp.Expr, sp.Expr]

SHELL_NAMES = ("12", "20", "30", "60")

#: Seed directions of the four orbit types (exact, not normalised).
SEEDS: Dict[str, Point] = {
    "12": (canon(PHI), S(1), S(0)),          # a five-fold axis
    "20": (S(1), S(1), S(1)),                # a three-fold axis
    "30": (S(0), S(0), S(1)),                # a two-fold axis
    "60": (S(0), S(1), canon(3 * PHI)),      # a vertex of the truncated icosahedron
}

POLYHEDRA = {"12": "icosahedron", "20": "dodecahedron", "30": "icosidodecahedron",
             "60": "truncated icosahedron"}

#: Irreps of I_h in the order used by the paper's tables: gerade A, T1, T2, G, H, then ungerade.
IRREPS_IH_ORDER = tuple(n + "g" for n in IRREPS_I) + tuple(n + "u" for n in IRREPS_I)


# --------------------------------------------------------------------------- exact points
def _to_field(e):
    """Exact conversion of a number in Q(sqrt5) to the field.  Canonical sympy numbers
    a + b*sqrt5 (as produced by ``_exact.canon``) are read off directly, which avoids the
    slow minimal-polynomial route of ``K.from_sympy``; anything else goes through ``to_K``."""
    if isinstance(e, (int, Fraction)):
        return from_ab(e)
    e = sp.sympify(e)
    if e.is_Rational:
        return from_ab(Fraction(int(e.p), int(e.q)))
    b = e.coeff(SQRT5)
    a = sp.expand(e - b * SQRT5)
    if a.is_Rational and b.is_Rational:
        return from_ab(Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q)))
    return to_K(e)


@lru_cache(maxsize=None)
def _kpoint_cached(p: tuple) -> tuple:
    return tuple(_to_field(c) for c in p)


def _kpoint(p: Sequence) -> tuple:
    """A direction as a tuple of three field elements (hashable, exact)."""
    if len(p) != 3:
        raise ValueError("a direction has three components")
    if all(isinstance(c, ANP) for c in p):          # already field elements
        return tuple(p)
    return _kpoint_cached(tuple(p))


def _kpoints(points: Sequence[Sequence]) -> Tuple[tuple, ...]:
    kpts = tuple(_kpoint(p) for p in points)
    if len(set(kpts)) != len(kpts):
        raise ValueError("repeated points")
    return kpts


def _spoint(kp) -> Point:
    """Field-element tuple -> tuple of canonical sympy numbers (hashable, exact)."""
    return tuple(to_sympy(c) for c in kp)


def _apply(kmat, kp):
    """R p for a rotation given as a 3x3 tuple of field elements."""
    return tuple(kmat[i][0] * kp[0] + kmat[i][1] * kp[1] + kmat[i][2] * kp[2] for i in range(3))


def _neg(kp):
    return tuple(-c for c in kp)


def _phi_str(e) -> str:
    """Print an element of Q(sqrt5) as p + q*phi (e.g. 'phi', '3*phi', '1')."""
    return sp.sstr(phi_form(e).subs(VARPHI, sp.Symbol("phi")))


@lru_cache(maxsize=None)
def _orbit_k(kseed: tuple, with_inversion: bool = False) -> Tuple[tuple, ...]:
    """The orbit of a field-element direction under the 60 rotations of I (and, if
    requested, under -1 as well), in the order in which the points are first produced."""
    G = icosahedral_group()
    if all(c == K_ZERO for c in kseed):
        raise ValueError("the seed must be a nonzero direction")
    points: List[tuple] = []
    seen = set()
    for kmat in G.kmatrices:
        for q in ((_apply(kmat, kseed),) if not with_inversion
                  else (_apply(kmat, kseed), _neg(_apply(kmat, kseed)))):
            if q not in seen:
                seen.add(q)
                points.append(q)
    return tuple(points)


def orbit(seed: Sequence) -> Tuple[Point, ...]:
    """The orbit {R p : R in I} of the direction ``seed`` under the 60 exact rotations.

    Points are tuples of canonical sympy numbers in Q(sqrt5) (hashable; exact equality),
    in a fixed deterministic order.  For the seeds of ``SEEDS`` the orbit sizes are
    12, 20, 30 and 60 (Table 4)."""
    return tuple(_spoint(q) for q in _orbit_k(_kpoint(seed)))


def orbit_Ih(seed: Sequence) -> Tuple[Point, ...]:
    """The orbit under I_h = I x {E, i}, i.e. under +-R.  For the four seeds of Table 4 it
    coincides with ``orbit(seed)`` (the I-orbits are centrosymmetric)."""
    return tuple(_spoint(q) for q in _orbit_k(_kpoint(seed), True))


def is_centrosymmetric(points: Sequence[Sequence]) -> bool:
    """True if the point set is mapped to itself by the inversion p -> -p."""
    kpts = set(_kpoints(points))
    return all(_neg(p) in kpts for p in kpts)


# --------------------------------------------------------------------------- characters
@lru_cache(maxsize=None)
def _fixed_counts(kpts: Tuple[tuple, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """(proper, improper) fixed-point counts for the 60 elements of I: for each rotation R
    the number of points with R p = p and the number with R p = -p (i.e. -R p = p).  Every
    image R p is computed once, exactly, and compared exactly."""
    G = icosahedral_group()
    proper, improper = [], []
    for kmat in G.kmatrices:
        n_fix = n_inv = 0
        for p in kpts:
            q = _apply(kmat, p)
            if q == p:
                n_fix += 1
            elif _neg(q) == p:
                n_inv += 1
        proper.append(n_fix)
        improper.append(n_inv)
    return tuple(proper), tuple(improper)


def fixed_point_counts(points: Sequence[Sequence], improper: bool = False) -> Tuple[int, ...]:
    """#{p in points : g p = p} for every element g of I (60 entries in group order).

    With ``improper=True`` the element is i R, acting as -R: the count is the number of
    points with R p = -p.  Comparison is exact."""
    return _fixed_counts(_kpoints(points))[1 if improper else 0]


def _class_function(G: IGroup, counts: Sequence[int]) -> Tuple[sp.Integer, ...]:
    chars = []
    for cls in CLASSES_I:
        vals = {counts[i] for i in G.classes[cls]}
        if len(vals) != 1:
            raise ValueError(f"fixed-point count is not constant on the class {cls}: {vals}")
        chars.append(S(vals.pop()))
    return tuple(chars)


def permutation_character(points: Sequence[Sequence], improper: bool = False) -> Tuple[sp.Integer, ...]:
    """Character of the permutation representation on ``points``: the number of fixed
    points of one representative of each class E, C5, C5^2, C3, C2 of I (order of
    ``groups.CLASSES_I``); it is checked to be constant on each class.

    ``improper=True`` gives the values on the improper classes i, S10^3, S10, S6, sigma
    of I_h, i.e. the fixed points of -R for R in E, C5, C5^2, C3, C2.  The point set must
    then be centrosymmetric, otherwise I_h does not act on it."""
    if improper and not is_centrosymmetric(points):
        raise ValueError("the point set is not closed under inversion; I_h does not act on it")
    return _class_function(icosahedral_group(), fixed_point_counts(points, improper))


def permutation_character_Ih(points: Sequence[Sequence]) -> Tuple[sp.Integer, ...]:
    """The 10-class character of I_h on a centrosymmetric point set, in the class order of
    ``groups.character_table_Ih()`` (proper classes first, then the improper ones)."""
    return permutation_character(points) + permutation_character(points, improper=True)


# --------------------------------------------------------------------------- stabilisers
def stabiliser_I(point: Sequence) -> Tuple[int, ...]:
    """Indices (into ``icosahedral_group().rotations``) of the rotations fixing ``point``."""
    G = icosahedral_group()
    kp = _kpoint(point)
    return tuple(i for i, kmat in enumerate(G.kmatrices) if _apply(kmat, kp) == kp)


def stabiliser_Ih(point: Sequence) -> Tuple[Tuple[int, int], ...]:
    """The stabiliser of ``point`` in I_h as pairs (index, sign): sign +1 is the rotation
    R_index, sign -1 the improper element i R_index = -R_index (which fixes p iff R p = -p)."""
    G = icosahedral_group()
    kp = _kpoint(point)
    out = []
    for i, kmat in enumerate(G.kmatrices):
        q = _apply(kmat, kp)
        if q == kp:
            out.append((i, 1))
        if _neg(q) == kp:
            out.append((i, -1))
    return tuple(out)


def stabiliser_labels(point: Sequence) -> Tuple[str, str]:
    """Schoenflies labels (in I, in I_h) of the stabiliser of a direction, read off from
    the group data: C_n from the n rotations fixing the point; C_nv (n > 1) or C_s (n = 1)
    when the n further improper elements are all reflections -C2 in planes through the
    point; plain C_n when there is no improper element."""
    G = icosahedral_group()
    stab = stabiliser_Ih(point)
    proper = [i for i, s in stab if s == 1]
    improper = [i for i, s in stab if s == -1]
    n = len(proper)
    label_I = f"C{n}"
    if not improper:
        return label_I, label_I
    reflections = [i for i in improper if G.rotations[i].cls == "C2"]
    if len(improper) == n and len(reflections) == n:
        return label_I, ("Cs" if n == 1 else f"C{n}v")
    raise ValueError(f"stabiliser of {tuple(point)} is not of type C_n, C_nv or C_s")


# --------------------------------------------------------------------------- decompositions
def _ordered(dec: Dict[str, int], order: Sequence[str]) -> Dict[str, int]:
    unknown = set(dec) - set(order)
    if unknown:
        raise ValueError(f"unknown irreps {sorted(unknown)}")
    return {k: dec[k] for k in order if dec.get(k, 0)}


def _seed_of(name) -> Point:
    key = str(name)
    if key not in SEEDS:
        raise KeyError(f"unknown shell {name!r}; expected one of {SHELL_NAMES}")
    return SEEDS[key]


def _check_dimension(table: CharacterTable, dec: Dict[str, int], size: int) -> None:
    dim = sum(m * table.dim(k) for k, m in dec.items())
    if dim != size:
        raise ValueError(f"decomposition has dimension {dim}, orbit has {size} points")


def decomposition_I(name) -> Dict[str, int]:
    """Irrep content of the permutation representation of I on the orbit ``name``
    ('12', '20', '30' or '60'), e.g. {'A': 1, 'T1': 1, 'T2': 1, 'H': 1} for the 12-orbit.
    Keys in the order A, T1, T2, G, H; irreps with multiplicity 0 are omitted."""
    return dict(shell(name).decomposition_I)


def decomposition_Ih(name) -> Dict[str, int]:
    """Irrep content of the permutation representation of I_h on the orbit ``name``
    (Table 4 of the paper), e.g. {'Ag': 1, 'Hg': 1, 'T1u': 1, 'T2u': 1} for the 12-orbit.
    Keys in the order Ag, T1g, T2g, Gg, Hg, Au, T1u, T2u, Gu, Hu; zeros omitted."""
    return dict(shell(name).decomposition_Ih)


def decomposition_by_frobenius(name, group: str = "I") -> Dict[str, int]:
    """Independent cross-check by Frobenius reciprocity: the multiplicity of Gamma in the
    permutation representation Ind_S^G 1 is the average of chi_Gamma over the stabiliser S
    of the seed.  ``group`` is 'I' or 'Ih'; for I_h the character of Gamma_g / Gamma_u on an
    improper element -R is +chi(R) / -chi(R)."""
    G = icosahedral_group()
    seed = _seed_of(name)
    out: Dict[str, int] = {}
    if group == "I":
        stab = stabiliser_I(seed)
        for irr in IRREPS_I:
            m = canon(sum(G.character(irr, i) for i in stab) / len(stab))
            if m != 0:
                out[irr] = int(m)
        return _ordered(out, IRREPS_I)
    if group == "Ih":
        stab = stabiliser_Ih(seed)
        for irr in IRREPS_I:
            for parity, sgn in (("g", 1), ("u", -1)):
                m = canon(sum(G.character(irr, i) * (1 if s == 1 else sgn) for i, s in stab)
                          / len(stab))
                if m != 0:
                    out[irr + parity] = int(m)
        return _ordered(out, IRREPS_IH_ORDER)
    raise ValueError("group must be 'I' or 'Ih'")


# --------------------------------------------------------------------------- Table 4
@dataclass(frozen=True)
class Shell:
    """One row of Table 4: an orbit type of I_h with its permutation characters and
    their irrep content.  ``character_I`` is the 5-class character (E, C5, C5^2, C3, C2),
    ``character_Ih`` the 10-class character in the class order of ``character_table_Ih()``.
    Decompositions are tuples of (irrep, multiplicity) pairs in the paper's order."""
    name: str
    seed: Point
    polyhedron: str
    points: Tuple[Point, ...]
    stabiliser_I: str
    stabiliser_Ih: str
    character_I: Tuple[sp.Integer, ...]
    character_Ih: Tuple[sp.Integer, ...]
    decomposition_I: Tuple[Tuple[str, int], ...]
    decomposition_Ih: Tuple[Tuple[str, int], ...]

    @property
    def size(self) -> int:
        return len(self.points)


@lru_cache(maxsize=None)
def shell(name) -> Shell:
    """Compute the row of Table 4 for the orbit ``name`` ('12', '20', '30' or '60')."""
    key = str(name)
    seed = _seed_of(key)
    pts = orbit(seed)
    if len(pts) != int(key):
        raise ValueError(f"orbit of {seed} has {len(pts)} points, expected {key}")
    if not is_centrosymmetric(pts):
        raise ValueError(f"orbit of {seed} is not centrosymmetric")
    chi_I = permutation_character(pts)
    chi_Ih = chi_I + permutation_character(pts, improper=True)
    dec_I = _ordered(CHAR_I.decompose(chi_I), IRREPS_I)
    table_Ih = character_table_Ih()
    dec_Ih = _ordered(table_Ih.decompose(chi_Ih), IRREPS_IH_ORDER)
    _check_dimension(CHAR_I, dec_I, len(pts))
    _check_dimension(table_Ih, dec_Ih, len(pts))
    lab_I, lab_Ih = stabiliser_labels(seed)
    return Shell(key, seed, POLYHEDRA[key], pts, lab_I, lab_Ih, chi_I, chi_Ih,
                 tuple(dec_I.items()), tuple(dec_Ih.items()))


def shells_table() -> List[Dict[str, object]]:
    """The data of Table 4 as plain Python: one dict per orbit type with keys
    'orbit' (str), 'size' (int), 'seed' (str, phi-form), 'polyhedron', 'stabiliser_I',
    'stabiliser_Ih', 'character_I' (list of int, classes E, C5, C5^2, C3, C2),
    'character_Ih' (list of int, 10 classes), 'decomposition_I' and 'decomposition_Ih'
    (dicts irrep -> multiplicity in the paper's order)."""
    rows = []
    for name in SHELL_NAMES:
        s = shell(name)
        rows.append({
            "orbit": s.name,
            "size": s.size,
            "seed": "(" + ", ".join(_phi_str(c) for c in s.seed) + ")",
            "polyhedron": s.polyhedron,
            "stabiliser_I": s.stabiliser_I,
            "stabiliser_Ih": s.stabiliser_Ih,
            "character_I": [int(c) for c in s.character_I],
            "character_Ih": [int(c) for c in s.character_Ih],
            "decomposition_I": dict(s.decomposition_I),
            "decomposition_Ih": dict(s.decomposition_Ih),
        })
    return rows


def format_decomposition(dec: Dict[str, int], plus: str = "+") -> str:
    """'Ag + Hg + T1u + T2u' style string; irreps of I in the order A, T1, T2, G, H, irreps
    of I_h gerade first (A, T1, T2, G, H) then ungerade; '2Gg' for multiplicity 2."""
    order = IRREPS_IH_ORDER if any(k in IRREPS_IH_ORDER for k in dec) else IRREPS_I
    ordered = _ordered(dict(dec), order)
    return f" {plus} ".join(f"{m if m > 1 else ''}{k}" for k, m in ordered.items())


def format_shells_table(plus: str = "+") -> List[Dict[str, str]]:
    """Table 4 as strings, ready for a LaTeX emitter: per orbit the keys 'orbit',
    'size', 'seed', 'polyhedron', 'stabiliser' (in I_h), 'stabiliser_I', 'I' and 'Ih'
    (the decompositions as 'A + T1 + T2 + H' / 'Ag + Hg + T1u + T2u')."""
    rows = []
    for r in shells_table():
        rows.append({"orbit": r["orbit"], "size": str(r["size"]), "seed": r["seed"],
                     "polyhedron": r["polyhedron"], "stabiliser": r["stabiliser_Ih"],
                     "stabiliser_I": r["stabiliser_I"],
                     "I": format_decomposition(r["decomposition_I"], plus),
                     "Ih": format_decomposition(r["decomposition_Ih"], plus)})
    return rows
