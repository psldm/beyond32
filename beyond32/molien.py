"""Molien series of I on harmonic polynomials and the S^3/2I (Poincare) dictionary.

Section 4 of the paper, Eqs. (14)-(16)
--------------------------------------
Functions on S^3 = SU(2) decompose under SU(2)_L x SU(2)_R as the sum over k >= 0 of
V_{k/2} (x) V_{k/2}, Eq. (14), where k labels the Laplace eigenvalue k(k+2).  The
eigenmodes of the Poincare dodecahedral space S^3/2I at wavenumber k are the vectors of
V_{k/2} (x) V_{k/2} invariant under 2I acting on one factor.  For odd k the element -1 of
2I acts as -1 and there are no invariants; for even k the number of invariants in
V_{k/2} is the number m_A(k/2) of I-invariants in the spherical harmonics of degree
l = k/2, i.e. the A line of Table 3 continued to all l.  Hence (Eq. 15)

    mult(k) = (k + 1) m_A(k/2)      (k even),

    sum_{l >= 0} m_A(l) t^l = (1 + t^15) / ((1 - t^6)(1 - t^10)),

the generating function being the Molien series of I on harmonic polynomials: the
denominator degrees 6 and 10 are the degrees of the primary icosahedral invariants and
the numerator records the secondary invariant of degree 15.  The resulting spectrum,
Eq. (16), starts (0,1), (12,13), (20,21), (24,25), (30,31), (32,33), (36,37), (40,41),
(42,43), (44,45), (48,49), (50,51), ..., (60,122), ...; k = 60 is the first wavenumber
with m_A = 2.

What this module computes
-------------------------
* ``m_A(l)``: the multiplicity of the trivial irrep A in the degree-l harmonics from the
  character formula (3) of the paper, <chi_A, chi_l>_I, exactly over Q(sqrt5).
* ``molien_series_coefficients(lmax)``: the Taylor coefficients of the closed form of
  Eq. (15), by exact power-series division of integer polynomials; ``check_molien``
  compares the two lists (the "verified for l <= 30" of the paper).
* ``molien_series_from_group()``: the same generating function obtained directly from
  the 60 rotation matrices by Molien's theorem, (1 - t^2)/|I| sum_g 1/det(1 - t R_g)
  (the factor 1 - t^2 removes the multiples of r^2, leaving the harmonic polynomials);
  ``check_molien_closed_form`` verifies that it equals (1 + t^15)/((1 - t^6)(1 - t^10))
  exactly.  This route is independent of the character table.
* ``poincare_spectrum(kmax)``: the pairs (k, mult(k)) of Eq. (16); ``twisted_spectrum``
  gives the analogous multiplicities (k + 1) m_Gamma(k/2) of the sectors twisted by a
  non-trivial irrep Gamma (last paragraph of Section 4).
* ``paper_dictionary`` / ``format_molien``: the data of Eqs. (15)-(16) as plain Python
  data (ints, strings, lists) for the LaTeX emitter.

All group-theoretic arithmetic is exact over Q(sqrt5) (see ``_exact``); the series
coefficients are exact integers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import sympy as sp
from sympy import Poly, Symbol

from ._exact import K
from .groups import CHAR_I, IRREPS_I, icosahedral_group, so3_characters

#: formal variable of the generating functions
t = Symbol("t")

#: degrees of the primary icosahedral invariants (denominator of Eq. 15) and of the
#: secondary invariant (numerator of Eq. 15)
MOLIEN_PRIMARY_DEGREES: Tuple[int, int] = (6, 10)
MOLIEN_SECONDARY_DEGREE: int = 15


# --------------------------------------------------------------------------- multiplicities
@lru_cache(maxsize=None)
def m_Gamma(l: int, irrep: str) -> int:
    """Multiplicity m_Gamma(l) of the irrep Gamma of I in the spherical harmonics of
    degree l, from the character formula (3): <chi_Gamma, chi_l>_I with chi_l the SO(3)
    character sin((l + 1/2) theta)/sin(theta/2) on the classes E, C5, C5^2, C3, C2.
    Exact over Q(sqrt5); the result is an integer (Table 3 continued to all l)."""
    if l < 0:
        raise ValueError("l must be non-negative")
    if irrep not in IRREPS_I:
        raise ValueError(f"unknown irrep {irrep!r} of I")
    m = CHAR_I.inner(CHAR_I.chars[irrep], so3_characters(l))
    if not m.is_Integer:
        raise ValueError(f"non-integral multiplicity {m} for {irrep} at l = {l}")
    return int(m)


def m_A(l: int) -> int:
    """Number m_A(l) of I-invariant spherical harmonics of degree l (the A line of
    Table 3), from the character formula (3).  This is also the number of 2I-invariants
    in the SU(2) irrep V_l, i.e. the Laplace multiplicity of S^3/2I at k = 2l divided by
    k + 1 (Eq. 15)."""
    return m_Gamma(l, "A")


def m_A_table(lmax: int = 30) -> List[int]:
    """[m_A(0), ..., m_A(lmax)] from the character formula."""
    return [m_A(l) for l in range(lmax + 1)]


def nontrivial_degrees(lmax: int = 30) -> List[int]:
    """The degrees l >= 1 with m_A(l) > 0 (paper: l = 6, 10, 12, 15, 16, 18, 20, 21, 22,
    24, ...): the degrees at which an "extended-s" A_g gap function of degree l exists."""
    return [l for l in range(1, lmax + 1) if m_A(l) > 0]


# --------------------------------------------------------------------------- Molien series, closed form
def molien_series() -> sp.Expr:
    """The closed form of Eq. (15), (1 + t^15)/((1 - t^6)(1 - t^10)), as a sympy
    expression in the symbol ``molien.t``."""
    d1, d2 = MOLIEN_PRIMARY_DEGREES
    return (1 + t**MOLIEN_SECONDARY_DEGREE) / ((1 - t**d1) * (1 - t**d2))


def molien_series_string() -> str:
    """The closed form of Eq. (15) as a plain string, e.g. for a table caption."""
    d1, d2 = MOLIEN_PRIMARY_DEGREES
    return f"(1 + t^{MOLIEN_SECONDARY_DEGREE})/((1 - t^{d1})(1 - t^{d2}))"


def _int_coeffs(p: sp.Expr) -> List[int]:
    """Ascending integer coefficient list [c_0, c_1, ...] of a polynomial in t."""
    P = Poly(sp.expand(p), t)
    coeffs = P.all_coeffs()[::-1]
    if not all(c.is_Integer for c in coeffs):
        raise ValueError("non-integer polynomial coefficients")
    return [int(c) for c in coeffs]


def _series_quotient(num: Sequence[int], den: Sequence[int], nmax: int) -> List[int]:
    """Coefficients c_0..c_nmax of the formal power series num(t)/den(t) for integer
    polynomials with den_0 = 1 (exact long division: den * c = num order by order)."""
    if den[0] != 1:
        raise ValueError("denominator must have constant term 1")
    out: List[int] = []
    for n in range(nmax + 1):
        c = num[n] if n < len(num) else 0
        for k in range(1, min(n, len(den) - 1) + 1):
            c -= den[k] * out[n - k]
        out.append(c)
    return out


def molien_series_coefficients(lmax: int = 30) -> List[int]:
    """Taylor coefficients of (1 + t^15)/((1 - t^6)(1 - t^10)) up to t^lmax, by exact
    power-series division of the integer numerator and denominator polynomials.
    Combinatorially, coefficient l counts the pairs (a, b) >= 0 with 6a + 10b = l plus
    those with 6a + 10b = l - 15 (Eq. 15, right-hand side)."""
    num, den = sp.fraction(sp.together(molien_series()))
    return _series_quotient(_int_coeffs(num), _int_coeffs(den), lmax)


def check_molien(lmax: int = 30) -> bool:
    """True if the character-formula multiplicities m_A(l) agree with the Molien
    coefficients of Eq. (15) for all l <= lmax (the paper's check for l <= 30)."""
    return m_A_table(lmax) == molien_series_coefficients(lmax)


# --------------------------------------------------------------------------- Molien's theorem from the matrices
@lru_cache(maxsize=None)
def _molien_fraction() -> Tuple[Poly, Poly]:
    """(numerator, denominator) polynomials over Q(sqrt5) of the harmonic Molien series

        M(t) = (1 - t^2)/|I| sum_{g in I} 1/det(1 - t R_g),

    computed from the 60 exact rotation matrices (Molien's theorem; the factor 1 - t^2
    passes from all polynomials to harmonic ones since P_l = H_l + r^2 P_{l-2})."""
    G = icosahedral_group()
    dets: Dict[Poly, int] = {}
    for r in G.rotations:
        D = Poly(sp.expand((sp.eye(3) - t * r.matrix).det()), t, domain=K)
        dets[D] = dets.get(D, 0) + 1          # one determinant per class of I
    one = Poly(1, t, domain=K)
    den = one
    for D in dets:
        den = den * D
    num = Poly(0, t, domain=K)
    for D, count in dets.items():
        term = one
        for D2 in dets:
            if D2 != D:
                term = term * D2
        num = num + term * count
    num = num * Poly(1 - t**2, t, domain=K)
    den = den * G.order
    return num, den


def molien_series_from_group() -> sp.Expr:
    """The harmonic Molien series of I obtained from the rotation matrices by Molien's
    theorem, (1 - t^2)/60 sum_g 1/det(1 - t R_g), as a sympy rational function of
    ``molien.t`` in lowest terms (``sp.cancel``); it equals ``molien_series()``, see
    ``check_molien_closed_form``."""
    num, den = _molien_fraction()
    return sp.cancel(num.as_expr() / den.as_expr())


def check_molien_closed_form() -> bool:
    """Exact verification of the closed form of Eq. (15) from the group data alone:
    (1 - t^2)/60 sum_g 1/det(1 - t R_g) == (1 + t^15)/((1 - t^6)(1 - t^10)), checked by
    cross-multiplication of polynomials over Q(sqrt5).  Independent of the character
    table (used by ``m_A``)."""
    num, den = _molien_fraction()
    cnum, cden = sp.fraction(sp.together(molien_series()))
    lhs = num * Poly(cden, t, domain=K)
    rhs = den * Poly(cnum, t, domain=K)
    return (lhs - rhs).is_zero


# --------------------------------------------------------------------------- S^3/2I spectrum
def poincare_multiplicity(k: int) -> int:
    """Laplace multiplicity mult(k) = (k + 1) m_A(k/2) of S^3/2I at wavenumber k
    (eigenvalue k(k + 2)), Eq. (15); zero for odd k (spinor sector has no invariants)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k % 2:
        return 0
    return (k + 1) * m_A(k // 2)


def poincare_spectrum(kmax: int = 60) -> List[Tuple[int, int]]:
    """The pairs (k, mult(k)) of Eq. (16) for even k <= kmax with mult(k) > 0."""
    out = []
    for k in range(0, kmax + 1, 2):
        m = poincare_multiplicity(k)
        if m > 0:
            out.append((k, m))
    return out


def first_k_with_m_A(m: int = 2, kmax: int = 600) -> int:
    """The smallest even wavenumber k with m_A(k/2) == m (k = 60 for m = 2: the first
    degree with two independent I-invariant harmonics is l = 30)."""
    for k in range(0, kmax + 1, 2):
        if m_A(k // 2) == m:
            return k
    raise ValueError(f"no even k <= {kmax} with m_A(k/2) == {m}")


def twisted_spectrum(irrep: str, kmax: int = 60) -> List[Tuple[int, int]]:
    """(k, (k + 1) m_Gamma(k/2)) for even k <= kmax with non-zero entry: the spectral
    multiplicities of S^3/2I for fields twisted by the irrep Gamma of the fundamental
    group 2I (Section 4, last paragraph); Gamma = 'A' gives ``poincare_spectrum``."""
    out = []
    for k in range(0, kmax + 1, 2):
        m = m_Gamma(k // 2, irrep)
        if m > 0:
            out.append((k, (k + 1) * m))
    return out


# --------------------------------------------------------------------------- paper data / table output
def paper_dictionary(lmax: int = 30, kmax: int = 60) -> Dict[str, object]:
    """The data quoted in Section 4, Eqs. (15)-(16), as plain Python data:

    ``m_A``                 [m_A(0), ..., m_A(lmax)] (character formula)
    ``molien_coefficients`` the same range of Taylor coefficients of Eq. (15)
    ``molien_series``       the closed form as a string
    ``nontrivial_l``        degrees 1 <= l <= lmax with m_A(l) > 0
    ``spectrum``            [(k, mult(k)), ...] for even k <= kmax, Eq. (16)
    ``first_k_with_m_A_2``  first wavenumber with m_A = 2
    ``checks``              {'character_formula_vs_series': bool, 'closed_form_from_group': bool}
    """
    return {
        "m_A": m_A_table(lmax),
        "molien_coefficients": molien_series_coefficients(lmax),
        "molien_series": molien_series_string(),
        "primary_degrees": list(MOLIEN_PRIMARY_DEGREES),
        "secondary_degree": MOLIEN_SECONDARY_DEGREE,
        "nontrivial_l": nontrivial_degrees(lmax),
        "spectrum": poincare_spectrum(kmax),
        "first_k_with_m_A_2": first_k_with_m_A(2),
        "checks": {"character_formula_vs_series": check_molien(lmax),
                   "closed_form_from_group": check_molien_closed_form()},
    }


def format_molien(lmax: int = 30, kmax: int = 60) -> Dict[str, object]:
    """Table-friendly version of Eqs. (15)-(16) for the LaTeX emitter (tab_molien.tex).

    Returns a dict of strings, ints and lists only:
    ``series``          closed form of Eq. (15) as a string
    ``columns``         ['l', 'm_A(l)', 'k = 2l', 'mult(k)']
    ``rows``            one row [l, m_A(l), 2l, (2l+1) m_A(l)] per l <= lmax with m_A(l) > 0
    ``m_A_row``         [m_A(0), ..., m_A(lmax)] for a one-line table
    ``nontrivial_l``    '6, 10, 12, 15, ...' as a string
    ``spectrum``        '(0, 1), (12, 13), ...' as a string, even k <= kmax
    ``first_k_with_m_A_2``  60
    """
    table = m_A_table(lmax)
    rows = [[l, m, 2 * l, (2 * l + 1) * m] for l, m in enumerate(table) if m > 0]
    return {
        "series": molien_series_string(),
        "columns": ["l", "m_A(l)", "k = 2l", "mult(k)"],
        "rows": rows,
        "m_A_row": list(table),
        "nontrivial_l": ", ".join(str(l) for l in nontrivial_degrees(lmax)),
        "spectrum": ", ".join(f"({k}, {m})" for k, m in poincare_spectrum(kmax)),
        "first_k_with_m_A_2": first_k_with_m_A(2),
    }
