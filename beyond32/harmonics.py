"""Harmonic polynomials on the pseudo-Fermi sphere and the action of I on them.

* ``rep_matrices(l)``: the 60 matrices of I on homogeneous polynomials of degree l
  (monomial basis), (rho(R) f)(r) = f(R^-1 r).
* ``harmonic_subspace(l)``: the (2l+1)-dimensional kernel of the Laplacian.
* ``projector(l, irrep)``: the isotypic projector P_Gamma = (d/60) sum chi(g)* rho(g).
* ``branching(l)`` / ``isotypic_basis(l, irrep)``: SO(3) -> I branching by explicit
  projection (Table 3), cross-checked by ``branching_by_characters`` (Eq. 3).
* ``paper_basis_functions()``: the phi-form basis functions of Eqs. (4)-(9), obtained by
  applying the projectors to symmetric seed polynomials; ``invariant_P6()`` and
  ``hexad_identity()`` for Eqs. (10)-(11).
* ``sphere_inner(f, g)``: exact inner product int f g dOmega / 4pi.

All arithmetic is exact over Q(sqrt5).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import sympy as sp
from sympy import Matrix, Poly, Rational, S, factorial2

from ._exact import (K, K_ONE, K_ZERO, canon, dm, dm_col, from_ab, phi_form, rref_rows, to_K,
                     to_matrix, to_sympy)
from .groups import CHAR_I, CLASSES_I, IRREPS_I, icosahedral_group, so3_characters

X, Y, Z = sp.symbols("x y z", real=True)
XYZ = (X, Y, Z)
R2 = X**2 + Y**2 + Z**2


# --------------------------------------------------------------------------- monomials
def exponents(l: int) -> Tuple[Tuple[int, int, int], ...]:
    """Exponent triples (a, b, c) of the monomials x^a y^b z^c of degree l (fixed order)."""
    return tuple((a, b, l - a - b) for a in range(l + 1) for b in range(l + 1 - a))


def monomials(l: int) -> Tuple[sp.Expr, ...]:
    return tuple(X**a * Y**b * Z**c for a, b, c in exponents(l))


def n_monomials(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def degree(p) -> int:
    return Poly(sp.expand(p), X, Y, Z).total_degree()


def coeff_dict(p) -> Dict[Tuple[int, int, int], sp.Expr]:
    P = Poly(sp.expand(p), X, Y, Z)
    return dict(zip(P.monoms(), P.coeffs()))


def coeff_vector(p, l: int):
    """Coefficient vector (column DomainMatrix over K) of a degree-l polynomial with
    coefficients in Q(sqrt5)."""
    d = coeff_dict(p)
    return dm_col(to_K(d.get(e, 0)) for e in exponents(l))


def poly_from_vector(v, l: int) -> sp.Expr:
    """Polynomial from a coefficient vector (DomainMatrix column, list of field elements,
    or sympy Matrix)."""
    if hasattr(v, "to_list"):
        coeffs = [row[0] for row in v.to_list()]
        coeffs = [to_sympy(c) for c in coeffs]
    else:
        coeffs = [to_sympy(c) if not isinstance(c, sp.Basic) or c.free_symbols == set() else c
                  for c in v]
    return sp.expand(sum(c * m for c, m in zip(coeffs, monomials(l))))


def cyclic(f) -> sp.Expr:
    """The image of f under x -> y -> z -> x."""
    return sp.expand(f.subs({X: Y, Y: Z, Z: X}, simultaneous=True))


def act(R: Matrix, f) -> sp.Expr:
    """(rho(R) f)(r) = f(R^-1 r) = f(R^T r) for a rotation matrix R."""
    Rt = R.T
    sub = {X: Rt[0, 0] * X + Rt[0, 1] * Y + Rt[0, 2] * Z,
           Y: Rt[1, 0] * X + Rt[1, 1] * Y + Rt[1, 2] * Z,
           Z: Rt[2, 0] * X + Rt[2, 1] * Y + Rt[2, 2] * Z}
    return sp.expand(f.subs(sub, simultaneous=True))


def laplacian(f) -> sp.Expr:
    return sp.expand(sp.diff(f, X, 2) + sp.diff(f, Y, 2) + sp.diff(f, Z, 2))


def is_harmonic(f) -> bool:
    return laplacian(f) == 0


def phi_form_poly(p) -> sp.Expr:
    """Rewrite the coefficients of p (in Q(sqrt5)) as p + q*varphi."""
    return sp.Add(*[phi_form(c) * X**e[0] * Y**e[1] * Z**e[2] for e, c in coeff_dict(p).items()])


# --------------------------------------------------------------------------- representation
def _linear_form(row) -> Poly:
    return Poly.from_dict({(1, 0, 0): row[0], (0, 1, 0): row[1], (0, 0, 1): row[2]},
                          X, Y, Z, domain=K)


@lru_cache(maxsize=None)
def rep_matrices(l: int) -> Tuple:
    """The 60 representation matrices of I on degree-l polynomials, monomial basis,
    as DomainMatrices over Q(sqrt5).  Column j is the image of monomial j."""
    G = icosahedral_group()
    exps = exponents(l)
    index = {e: i for i, e in enumerate(exps)}
    n = len(exps)
    out = []
    for kmat in G.kmatrices:
        # (rho(R) f)(r) = f(R^T r): x -> row 0 of R^T = column 0 of R, etc.
        Lx = _linear_form([kmat[0][0], kmat[1][0], kmat[2][0]])
        Ly = _linear_form([kmat[0][1], kmat[1][1], kmat[2][1]])
        Lz = _linear_form([kmat[0][2], kmat[1][2], kmat[2][2]])
        one = Poly.from_dict({(0, 0, 0): K_ONE}, X, Y, Z, domain=K)
        px = [one]
        py = [one]
        pz = [one]
        for _ in range(l):
            px.append(px[-1] * Lx)
            py.append(py[-1] * Ly)
            pz.append(pz[-1] * Lz)
        cols = []
        for (a, b, c) in exps:
            img = px[a] * py[b] * pz[c]
            col = [K_ZERO] * n
            for e, coeff in img.as_dict(native=True).items():
                col[index[e]] = coeff
            cols.append(col)
        rows = [[cols[j][i] for j in range(n)] for i in range(n)]
        out.append(dm(rows))
    return tuple(out)


@lru_cache(maxsize=None)
def class_sums(l: int) -> Dict[str, object]:
    """sum_{g in C} rho(g) for each class C of I (DomainMatrices)."""
    G = icosahedral_group()
    reps = rep_matrices(l)
    out = {}
    for cls in CLASSES_I:
        acc = None
        for i in G.classes[cls]:
            acc = reps[i] if acc is None else acc + reps[i]
        out[cls] = acc
    return out


@lru_cache(maxsize=None)
def projector(l: int, irrep: str):
    """Isotypic projector P_Gamma = (d_Gamma/60) sum_g chi_Gamma(g) rho(g) on degree-l
    polynomials (DomainMatrix over Q(sqrt5))."""
    sums = class_sums(l)
    chars = CHAR_I.chars[irrep]
    d = CHAR_I.dim(irrep)
    acc = None
    for cls, chi in zip(CLASSES_I, chars):
        if chi == 0:
            continue
        term = sums[cls] * to_K(chi)
        acc = term if acc is None else acc + term
    return acc * to_K(Rational(d, 60))


# --------------------------------------------------------------------------- harmonics
@lru_cache(maxsize=None)
def laplacian_matrix(l: int):
    """Matrix of the Laplacian from degree-l to degree-(l-2) polynomials (monomial bases)."""
    if l < 2:
        return dm([[K_ZERO] * n_monomials(l)])
    src = exponents(l)
    dst = {e: i for i, e in enumerate(exponents(l - 2))}
    rows = [[K_ZERO] * len(src) for _ in dst]
    for j, (a, b, c) in enumerate(src):
        if a >= 2:
            rows[dst[(a - 2, b, c)]][j] += from_ab(a * (a - 1))
        if b >= 2:
            rows[dst[(a, b - 2, c)]][j] += from_ab(b * (b - 1))
        if c >= 2:
            rows[dst[(a, b, c - 2)]][j] += from_ab(c * (c - 1))
    return dm(rows)


@lru_cache(maxsize=None)
def harmonic_subspace(l: int):
    """Columns: coefficient vectors of a basis of the harmonic polynomials of degree l
    (DomainMatrix of shape n_l x (2l+1))."""
    N = laplacian_matrix(l).nullspace()      # rows are basis vectors
    H = N.transpose()
    assert H.shape[1] == 2 * l + 1, (l, H.shape)
    return H


@lru_cache(maxsize=None)
def harmonic_projection_matrices(l: int):
    """Exact linear maps on degree-l coefficient vectors: v -> (harmonic part, quotient)
    with p = h + r^2 q.  Returns (Hproj, Qproj) as sympy Matrices with entries in Q(sqrt5);
    Hproj is n_l x n_l, Qproj is n_{l-2} x n_l (Qproj is empty for l < 2)."""
    from sympy.polys.matrices import DomainMatrix

    n = n_monomials(l)
    if l < 2:
        return sp.eye(n), sp.zeros(0, n)
    Hb = harmonic_subspace(l)
    exps_l2 = exponents(l - 2)
    index = {e: i for i, e in enumerate(exponents(l))}
    qcols = []
    for (a, b, c) in exps_l2:
        col = [K_ZERO] * n
        col[index[(a + 2, b, c)]] += K_ONE
        col[index[(a, b + 2, c)]] += K_ONE
        col[index[(a, b, c + 2)]] += K_ONE
        qcols.append(col)
    Q = dm([[qcols[j][i] for j in range(len(qcols))] for i in range(n)])
    B = DomainMatrix.hstack(Hb, Q)                 # n x n, invertible
    Binv = B.inv()
    top = Binv[: 2 * l + 1, :]
    bottom = Binv[2 * l + 1:, :]
    return to_matrix(Hb * top), to_matrix(bottom)


def harmonic_part(p) -> sp.Expr:
    """The harmonic (top-l) component h of a homogeneous polynomial p = h + r^2 q."""
    p = sp.expand(p)
    if p == 0:
        return S(0)
    l = degree(p)
    Hproj, _ = harmonic_projection_matrices(l)
    v = Matrix([coeff_dict(p).get(e, 0) for e in exponents(l)])
    return sp.expand(sum(c * m for c, m in zip(Hproj * v, monomials(l))))


def harmonic_components(F) -> Dict[int, sp.Expr]:
    """Decompose a homogeneous polynomial F of degree n as sum_L h_L r^(n-L) with h_L
    harmonic; returns {L: h_L} (coefficients may be symbolic)."""
    F = sp.expand(F)
    if F == 0:
        return {}
    n = degree(F)
    comps = {}
    rest = F
    for l in range(n, -1, -2):
        Hproj, Qproj = harmonic_projection_matrices(l)
        v = Matrix([coeff_dict(rest).get(e, 0) for e in exponents(l)])
        h = sp.expand(sum(c * m for c, m in zip(Hproj * v, monomials(l))))
        comps[l] = h
        if l < 2:
            break
        rest = sp.expand(sum(c * m for c, m in zip(Qproj * v, monomials(l - 2))))
        if rest == 0:
            break
    return comps


# --------------------------------------------------------------------------- branching
def _nice(p) -> sp.Expr:
    """Normalise a polynomial: integer coefficients with gcd 1 if all rational, otherwise
    leading coefficient 1 (as in the legacy code)."""
    p = sp.expand(p)
    P = Poly(p, X, Y, Z)
    coeffs = [canon(c) for c in P.coeffs()]
    if all(c.is_Rational for c in coeffs):
        g = sp.gcd_list(coeffs)
        return sp.expand(p / g)
    lc = coeffs[0]
    return sp.expand(sum(canon(c / lc) * m for c, m in zip(coeffs, [sp.Mul(*[g**e for g, e in zip((X, Y, Z), mon)]) for mon in P.monoms()])))


@lru_cache(maxsize=None)
def _isotypic(l: int):
    """(branching dict, bases dict) for degree l by explicit projection of the harmonic
    subspace: for each irrep the rref basis of P_Gamma(harmonics)."""
    Hmat = harmonic_subspace(l)
    branching_ = {}
    bases = {}
    for irr in IRREPS_I:
        d = CHAR_I.dim(irr)
        img = projector(l, irr) * Hmat
        rows = rref_rows(img.transpose())
        assert len(rows) % d == 0, (l, irr, len(rows))
        m = len(rows) // d
        branching_[irr] = m
        if m:
            bases[irr] = tuple(_nice(poly_from_vector(r.transpose(), l)) for r in rows)
    dim = sum(branching_[irr] * CHAR_I.dim(irr) for irr in IRREPS_I)
    assert dim == 2 * l + 1, (l, dim)
    return branching_, bases


def branching(l: int) -> Dict[str, int]:
    """SO(3) -> I branching of the degree-l harmonics by explicit projection;
    keys with nonzero multiplicity only."""
    b, _ = _isotypic(l)
    return {k: m for k, m in b.items() if m}


def branching_by_characters(l: int) -> Dict[str, int]:
    """The same multiplicities from the character formula (3) of the paper."""
    return CHAR_I.decompose(so3_characters(l))


def isotypic_basis(l: int, irrep: str) -> Tuple[sp.Expr, ...]:
    """A basis (harmonic polynomials) of the Gamma-isotypic component of degree l."""
    _, bases = _isotypic(l)
    return bases.get(irrep, ())


def branching_table(lmax: int = 6) -> Dict[int, Dict[str, int]]:
    return {l: branching(l) for l in range(lmax + 1)}


def format_branching(dec: Dict[str, int]) -> str:
    return " + ".join(f"{m if m > 1 else ''}{k}" for k, m in dec.items() if m)


# --------------------------------------------------------------------------- seeds / paper bases
def project(l: int, irrep: str, p) -> sp.Expr:
    """P_Gamma applied to a degree-l polynomial with coefficients in Q(sqrt5)."""
    v = projector(l, irrep) * coeff_vector(p, l)
    return poly_from_vector(v, l)


def basis_from_seed(l: int, irrep: str, seed) -> sp.Expr:
    """P_Gamma[harm(seed)], normalised so that the coefficient of the seed's leading
    monomial is 1 (if it is nonzero; otherwise the leading coefficient), as in the legacy
    ``ih_seeds.py``.  Returns 0 if the projection vanishes."""
    out = project(l, irrep, harmonic_part(seed))
    if out == 0:
        return S(0)
    d = coeff_dict(out)
    sm = Poly(sp.expand(seed), X, Y, Z).monoms()[0]
    lead = d.get(sm, None)
    if lead is None or lead == 0:
        lead = Poly(out, X, Y, Z).coeffs()[0]
    return sp.expand(sum(canon(c / lead) * X**e[0] * Y**e[1] * Z**e[2] for e, c in d.items()))


SEED_TASKS = {1: {"T1": (X,)}, 2: {"H": (X * Y, X**2 - Y**2)},
              3: {"T2": (X**3,), "G": (X**3, X * Y * Z)},
              4: {"G": (X**4, X**2 * Y * Z), "H": (X**4, X**2 * Y * Z, X**3 * Y)}}


@lru_cache(maxsize=None)
def seed_basis_functions() -> Dict[Tuple[int, str, sp.Expr], sp.Expr]:
    """{(l, irrep, seed): P_Gamma[harm(seed)]} for the seeds of the legacy ``ih_seeds.py``."""
    out = {}
    for l, tasks in SEED_TASKS.items():
        for irr, seeds in tasks.items():
            for s in seeds:
                out[(l, irr, s)] = basis_from_seed(l, irr, s)
    return out


@lru_cache(maxsize=None)
def paper_basis_functions() -> Dict[Tuple[int, str], Dict[str, sp.Expr]]:
    """The named basis functions of Eqs. (4)-(9) of the paper, generated by the projectors
    ("+cyclic" = images under x -> y -> z -> x)."""
    out = {}
    # l = 1, T1
    fx = basis_from_seed(1, "T1", X)
    out[(1, "T1")] = {"x": fx, "y": cyclic(fx), "z": cyclic(cyclic(fx))}
    # l = 2, H: all of l = 2 is H
    xy = basis_from_seed(2, "H", X * Y)
    x2y2 = basis_from_seed(2, "H", X**2 - Y**2)
    out[(2, "H")] = {"xy": xy, "yz": cyclic(xy), "zx": cyclic(cyclic(xy)),
                     "x^2-y^2": x2y2,
                     "2z^2-x^2-y^2": sp.expand(cyclic(cyclic(x2y2)) - cyclic(x2y2))}
    # l = 3
    f = basis_from_seed(3, "T2", X**3)
    out[(3, "T2")] = {"f_x": f, "f_y": cyclic(f), "f_z": cyclic(cyclic(f))}
    gx = basis_from_seed(3, "G", X**3)
    out[(3, "G")] = {"g_x": gx, "g_y": cyclic(gx), "g_z": cyclic(cyclic(gx)),
                     "xyz": basis_from_seed(3, "G", X * Y * Z)}
    # l = 4
    u0 = basis_from_seed(4, "G", X**4)
    ux = basis_from_seed(4, "G", X**2 * Y * Z)
    out[(4, "G")] = {"u_0": u0, "u_x": ux, "u_y": cyclic(ux), "u_z": cyclic(cyclic(ux))}
    vx = basis_from_seed(4, "H", X**4)
    wx = basis_from_seed(4, "H", X**2 * Y * Z)
    out[(4, "H")] = {"v_x": vx, "v_y": cyclic(vx), "v_z": cyclic(cyclic(vx)),
                     "w_x": wx, "w_y": cyclic(wx), "w_z": cyclic(cyclic(wx))}
    return out


@lru_cache(maxsize=None)
def invariant_P6() -> sp.Expr:
    """The l = 6 icosahedral invariant P6 = P_A[harm(x^6)] with coefficient of x^6 equal 1
    (Eq. 11)."""
    return basis_from_seed(6, "A", X**6)


@lru_cache(maxsize=None)
def hexad_identity() -> Dict[str, sp.Expr]:
    """The hexad sum S = sum_i (a_i . n)^6 over the six unit five-fold axes, as a polynomial,
    and its exact decomposition S = c * P6 + e * r^6 (c = -2/35, e = 6/7 on the sphere)."""
    from .groups import fivefold_axes

    n = Matrix([X, Y, Z])
    S6 = S(0)
    for a in fivefold_axes():
        num = sp.expand((a.T * n)[0] ** 6)
        den = canon((a.T * a)[0] ** 3)
        S6 += sp.expand(num / den)
    S6 = sp.expand(S6)
    S6 = sp.expand(sum(canon(c) * X**e[0] * Y**e[1] * Z**e[2] for e, c in coeff_dict(S6).items()))
    P6 = invariant_P6()
    h6 = harmonic_part(S6)
    # h6 = c * P6
    d6, dP = coeff_dict(h6), coeff_dict(P6)
    c = canon(d6[(6, 0, 0)] / dP[(6, 0, 0)])
    assert sp.expand(h6 - c * P6) == 0
    rest = sp.expand(S6 - h6)
    e = canon(coeff_dict(rest).get((6, 0, 0), 0))
    assert sp.expand(rest - e * R2**3) == 0
    return {"hexad": S6, "c": c, "e": e, "P6": P6}


# --------------------------------------------------------------------------- sphere inner product
def mono_int(a: int, b: int, c: int) -> sp.Expr:
    """int x^a y^b z^c dOmega / 4pi over the unit sphere."""
    if a % 2 or b % 2 or c % 2:
        return S(0)
    return Rational(factorial2(a - 1) * factorial2(b - 1) * factorial2(c - 1), factorial2(a + b + c + 1))


def sphere_inner(f, g) -> sp.Expr:
    """Exact int f g dOmega / 4pi for polynomials with coefficients in Q(sqrt5)."""
    P = Poly(sp.expand(f * g), X, Y, Z)
    return canon(sum(c * mono_int(*m) for m, c in zip(P.monoms(), P.coeffs())))


def sphere_norm2(f) -> sp.Expr:
    return sphere_inner(f, f)


@lru_cache(maxsize=None)
def sphere_gram(l: int):
    """Gram matrix <m_i, m_j> of the degree-l monomials (DomainMatrix over Q(sqrt5))."""
    exps = exponents(l)
    rows = [[to_K(mono_int(a + a2, b + b2, c + c2)) for (a2, b2, c2) in exps] for (a, b, c) in exps]
    return dm(rows)
