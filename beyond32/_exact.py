"""Exact arithmetic over Q(sqrt 5).

Every group-theoretic quantity in this package lives in the number field
Q(sqrt 5) = {a + b sqrt5 : a, b rational}.  This module is the single place where
that field is set up.  Arithmetic is done with sympy's algebraic-field domain ``K``
(elements are ``ANP`` objects) and linear algebra with ``DomainMatrix`` over ``K``;
both are exact.  Conversion to ordinary sympy expressions happens only at the
boundary (printing, tests, LaTeX).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

import sympy as sp
from sympy import Matrix, Rational, sqrt
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyclasses import ANP

SQRT5 = sqrt(5)
PHI = (1 + SQRT5) / 2            # golden ratio
PHI_INV = (SQRT5 - 1) / 2        # 1/phi = phi - 1
VARPHI = sp.Symbol("varphi")     # symbol used for the printed "phi-form"

K = QQ.algebraic_field(SQRT5)    # the field Q(sqrt 5)
K_ZERO = K.zero
K_ONE = K.one
K_SQRT5 = K.from_sympy(SQRT5)


# --------------------------------------------------------------------------- scalars
def from_ab(a, b=0) -> ANP:
    """The field element a + b*sqrt5 from two rationals."""
    return K.convert(QQ(Fraction(a))) + K.convert(QQ(Fraction(b))) * K_SQRT5


def to_K(e) -> ANP:
    """Convert a sympy expression / number that lies in Q(sqrt5) to the field."""
    if isinstance(e, ANP):
        return e
    if isinstance(e, (int, Fraction)):
        return K.convert(QQ(Fraction(e)))
    e = sp.sympify(e)
    if e.is_Rational:
        return K.convert(QQ(int(e.p), int(e.q)))
    return K.from_sympy(sp.radsimp(sp.expand(e)))


def to_sympy(e) -> sp.Expr:
    """Canonical sympy form a + b*sqrt5 of a field element (or of a sympy expression)."""
    if isinstance(e, ANP):
        return K.to_sympy(e)
    return K.to_sympy(to_K(e))


def ab(e) -> tuple[Rational, Rational]:
    """The rational pair (a, b) with e = a + b*sqrt5."""
    s = to_sympy(e)
    b = s.coeff(SQRT5)
    a = sp.expand(s - b * SQRT5)
    return Rational(a), Rational(b)


def phi_form(e) -> sp.Expr:
    """Write an element of Q(sqrt5) as p + q*varphi with rational p, q."""
    a, b = ab(e)
    return a - b + 2 * b * VARPHI      # sqrt5 = 2 phi - 1


def galois(e):
    """The Galois conjugate sqrt5 -> -sqrt5 (phi -> 1 - phi)."""
    a, b = ab(e)
    return to_sympy(from_ab(a, -b)) if not isinstance(e, ANP) else from_ab(a, -b)


def canon(e) -> sp.Expr:
    """Canonicalise a sympy expression in Q(sqrt5) (expand, rationalise denominators)."""
    return to_sympy(e)


def is_zero(e) -> bool:
    return to_K(e) == K_ZERO


def as_float(e) -> float:
    a, b = ab(e)
    return float(a) + float(b) * 5 ** 0.5


# --------------------------------------------------------------------------- matrices
def dm(rows: Sequence[Sequence[ANP]]) -> DomainMatrix:
    """DomainMatrix over K from a list of rows of field elements."""
    rows = [list(r) for r in rows]
    return DomainMatrix(rows, (len(rows), len(rows[0]) if rows else 0), K)


def dm_from_sympy(M: Matrix) -> DomainMatrix:
    """Convert a sympy Matrix with entries in Q(sqrt5) to a DomainMatrix over K."""
    rows = [[to_K(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]
    return DomainMatrix(rows, (M.rows, M.cols), K)


def dm_zeros(n: int, m: int) -> DomainMatrix:
    return DomainMatrix.zeros((n, m), K)


def dm_eye(n: int) -> DomainMatrix:
    return DomainMatrix.eye(n, K)


def dm_col(entries: Iterable[ANP]) -> DomainMatrix:
    rows = [[e] for e in entries]
    return DomainMatrix(rows, (len(rows), 1), K)


def dm_scale(D: DomainMatrix, c) -> DomainMatrix:
    return D * to_K(c)


def to_matrix(D: DomainMatrix) -> Matrix:
    """DomainMatrix over K -> sympy Matrix with canonical entries."""
    return Matrix(D.shape[0], D.shape[1], [K.to_sympy(e) for row in D.to_list() for e in row])


def to_float_array(D: DomainMatrix):
    """DomainMatrix over K -> numpy float array (numerics only; never fed back into exact code)."""
    import numpy as np

    return np.array([[as_float(e) for e in row] for row in D.to_list()], dtype=float)


def nullspace_columns(D: DomainMatrix) -> list[DomainMatrix]:
    """Basis of the right null space of D as a list of column DomainMatrices."""
    N = D.nullspace()                       # rows are basis vectors
    return [N[i, :].transpose() for i in range(N.shape[0])]


def rref_rows(D: DomainMatrix) -> list[DomainMatrix]:
    """The nonzero rows of the reduced row echelon form of D (as row DomainMatrices)."""
    R, pivots = D.rref()
    return [R[i, :] for i in range(len(pivots))]


def rank(D: DomainMatrix) -> int:
    return D.rank()


def solve(A: DomainMatrix, b: DomainMatrix) -> DomainMatrix:
    """Solve A x = b exactly (A need not be square; the system must be consistent)."""
    n = A.shape[1]
    aug = DomainMatrix.hstack(A, b)
    R, pivots = aug.rref()
    if n in pivots:
        raise ValueError("inconsistent linear system")
    rows = [[K_ZERO] * b.shape[1] for _ in range(n)]
    for r, p in enumerate(pivots):
        rows[p] = [R[r, n + j].element for j in range(b.shape[1])]
    return DomainMatrix(rows, (n, b.shape[1]), K)


def entries(D: DomainMatrix) -> list[list[ANP]]:
    return D.to_list()


def dm_equal(A: DomainMatrix, B: DomainMatrix) -> bool:
    return A.shape == B.shape and A.to_list() == B.to_list()
