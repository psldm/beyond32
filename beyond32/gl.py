"""Ginzburg-Landau theory of the icosahedral pairing channels (Section 6 of the paper).

Exact part (sympy over Q(sqrt5); the H basis function (2z^2-x^2-y^2)/(2 sqrt3) brings in
sqrt3, so a few intermediate matrix entries live in Q(sqrt5, sqrt3) -- still exact):

* ``channel(name)``: orthonormal (sphere inner product) basis functions of T1, T2, G, H and
  the 60 exact representation matrices D(g) in that basis.
* ``sym2_by_characters`` / ``sym2_projector_ranks``: Sym^2 Gamma decompositions (Table 6).
* ``quartic_invariants(name)``: harmonic content of Delta^2 and the invariants
  N_L = int |[Delta^2]_L|^2 dOmega/4pi; ``relations(name)``: N_L in terms of
  I1 = (eta.eta*)^2, I2 = |eta.eta|^2 (and N2 for G) and int |Delta|^4 (Eqs. 18-22).
* ``h_channel()``: the G/H split of [Delta^2]_4, the intertwiner J built by Schur averaging,
  its isometry scale, the cross terms Re C, Im C, and the independence of the six forms.

Numerical part (numpy/scipy, fixed seeds, as in the legacy scripts):

* ``minimise_ratio(name)``: weak-coupling ratio R = int|Delta|^4/(int|Delta|^2)^2 minimised
  over the order parameter (BFGS restarts, seed 0).
* ``g_ground_state``, ``g_stratum``, ``g_nodes``: the G ground state (kappa, phase, nodes).
* ``isotropy_table()``: isotropy subgroups (K, chi) with one-dimensional fixed space and the
  symmetry-fixed states (Table 7).
* ``h_candidates()``: the null-cone H states and their anisotropic invariants (Table 8).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy import I as IMAG
from sympy import Matrix, Poly, Rational, S, lambdify, sqrt
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from ._exact import K, PHI, ab, as_float, canon, solve as ksolve, to_K, to_matrix
from .groups import (CHAR_I, CLASSES_I, IRREPS_I, generate_subgroup, icosahedral_group,
                     rotation_about)
from .harmonics import (X, Y, Z, coeff_dict, exponents, laplacian_matrix, mono_int, monomials,
                        paper_basis_functions, projector, rep_matrices, sphere_gram, sphere_inner)

CHANNELS = ("T1", "T2", "G", "H")
CHANNEL_L = {"T1": 1, "H": 2, "T2": 3, "G": 3}
_TOL = 1e-8


# =========================================================================== channels
@lru_cache(maxsize=None)
def channel_basis(name: str) -> Tuple[sp.Expr, ...]:
    """The (orthogonal, not normalised) basis functions used for the irrep matrices:
    T1: x, y, z;  H: xy, yz, zx, (x^2-y^2)/2, (2z^2-x^2-y^2)/(2 sqrt3);
    T2: f_x, f_y, f_z;  G: g_x, g_y, g_z, xyz  (Eqs. 4-7 of the paper)."""
    B = paper_basis_functions()
    if name == "T1":
        return (X, Y, Z)
    if name == "H":
        return (X * Y, Y * Z, Z * X, (X**2 - Y**2) / 2, (2 * Z**2 - X**2 - Y**2) / (2 * sqrt(3)))
    if name == "T2":
        b = B[(3, "T2")]
        return (b["f_x"], b["f_y"], b["f_z"])
    if name == "G":
        b = B[(3, "G")]
        return (b["g_x"], b["g_y"], b["g_z"], b["xyz"])
    raise ValueError(name)


@dataclass(frozen=True)
class Channel:
    """A pairing channel: orthonormal basis functions and exact irrep matrices D(g)."""
    name: str
    l: int
    basis: Tuple[sp.Expr, ...]
    norms2: Tuple[sp.Expr, ...]        # int b_i^2 dOmega/4pi
    onb: Tuple[sp.Expr, ...]           # b_i / sqrt(norms2_i)
    matrices: Tuple[Matrix, ...]       # D(g)_{ji} = <b^_j, g b^_i>, g in I (60)

    @property
    def dim(self) -> int:
        return len(self.basis)

    def numeric_matrices(self) -> np.ndarray:
        return np.array([np.array(D.evalf(30).tolist(), dtype=float) for D in self.matrices])

    def numeric_basis(self):
        return [lambdify((X, Y, Z), b, "numpy") for b in self.onb]


def _coeff_matrix(funcs: Sequence[sp.Expr], l: int) -> Matrix:
    exps = exponents(l)
    return Matrix([[coeff_dict(f).get(e, 0) for f in funcs] for e in exps])


@lru_cache(maxsize=None)
def channel(name: str) -> Channel:
    """Orthonormal basis and exact representation matrices of a channel (legacy gl_inv.py).

    D(g)_{ji} = <b^_j, rho(g) b^_i> is computed from the monomial representation matrices
    and the exact Gram matrix of the monomials; the result is orthogonal, has trace
    chi_Gamma(g) and is a homomorphism (checked)."""
    l = CHANNEL_L[name]
    B = channel_basis(name)
    d = len(B)
    Gram = Matrix(d, d, lambda i, j: sphere_inner(B[i], B[j]))
    for i in range(d):
        for j in range(d):
            if i != j:
                assert Gram[i, j] == 0, (name, i, j, Gram[i, j])
    norms2 = tuple(Gram[i, i] for i in range(d))
    onb = tuple(sp.expand(B[i] / sqrt(norms2[i])) for i in range(d))
    Bc = _coeff_matrix(B, l)                               # n_l x d
    Gm = to_matrix(sphere_gram(l))                         # n_l x n_l
    A = (Bc.T * Gm)                                        # d x n_l
    scale = Matrix(d, d, lambda j, i: 1 / sqrt(norms2[j] * norms2[i]))
    G = icosahedral_group()
    Ds = []
    for i in range(60):
        rho = to_matrix(rep_matrices(l)[i])
        D = (A * rho * Bc)
        D = Matrix(d, d, lambda j, k: sp.expand(D[j, k] * scale[j, k]))
        Ds.append(D)
    # checks: orthogonal, character, homomorphism (sample as in the legacy code)
    for D in Ds:
        assert (D.T * D).applyfunc(sp.expand) == sp.eye(d), (name, "not orthogonal")
    for i, D in enumerate(Ds):
        assert sp.expand(D.trace() - G.character(name, i)) == 0, (name, "character mismatch")
    for a in range(0, 60, 17):
        for b in range(0, 60, 23):
            assert (Ds[a] * Ds[b] - Ds[G.mult[a][b]]).applyfunc(sp.expand) == sp.zeros(d), "not a homomorphism"
    return Channel(name, l, tuple(B), norms2, onb, tuple(Ds))


# =========================================================================== Sym^2
_SQUARE_CLASS = {"E": "E", "C5": "C5^2", "C5^2": "C5", "C3": "C3", "C2": "E"}


def sym2_by_characters(name: str) -> Dict[str, int]:
    """Sym^2 Gamma by characters, chi_s(g) = (chi(g)^2 + chi(g^2))/2 (Table 6)."""
    row = CHAR_I.chars[name]
    idx = {c: i for i, c in enumerate(CLASSES_I)}
    chars = [canon((row[i] ** 2 + row[idx[_SQUARE_CLASS[c]]]) / 2) for i, c in enumerate(CLASSES_I)]
    return CHAR_I.decompose(chars)


def antisym2_by_characters(name: str) -> Dict[str, int]:
    row = CHAR_I.chars[name]
    idx = {c: i for i, c in enumerate(CLASSES_I)}
    chars = [canon((row[i] ** 2 - row[idx[_SQUARE_CLASS[c]]]) / 2) for i, c in enumerate(CLASSES_I)]
    return CHAR_I.decompose(chars)


def hermitian_form_count(name: str) -> int:
    """dim Hom_I(Sym^2 Gamma, Sym^2 Gamma) = sum_lambda m_lambda^2 (Eq. 17)."""
    return sum(m * m for m in sym2_by_characters(name).values())


def tr_even_quartic_count(name: str) -> int:
    """Quartic terms allowed with F(eta*) = F(eta): sum_lambda m_lambda (m_lambda + 1)/2."""
    return sum(m * (m + 1) // 2 for m in sym2_by_characters(name).values())


def _rank_exact(M: Matrix) -> int:
    """Exact rank of a sympy matrix with entries in Q(sqrt3, sqrt5)."""
    K2 = _K2()
    rows = [[K2.from_sympy(sp.expand(M[i, j])) for j in range(M.cols)] for i in range(M.rows)]
    return DomainMatrix(rows, (M.rows, M.cols), K2).rank()


@lru_cache(maxsize=None)
def sym2_projector_ranks(name: str) -> Dict[str, int]:
    """Sym^2 Gamma by explicit isotypic projectors on symmetric tensors eta_i eta_j
    (legacy gl_inv.py): multiplicity = rank(P_lambda on Sym^2)/dim lambda."""
    ch = channel(name)
    d = ch.dim
    G = icosahedral_group()
    dd = d * d
    class_sums = {}
    for cls in CLASSES_I:
        acc = sp.zeros(dd)
        for i in G.classes[cls]:
            D = ch.matrices[i]
            acc += sp.kronecker_product(D, D)
        class_sums[cls] = acc.applyfunc(sp.expand)
    Ssym = []
    for i in range(d):
        for j in range(i, d):
            v = sp.zeros(dd, 1)
            v[i * d + j] += 1
            v[j * d + i] += 1
            Ssym.append(v)
    Smat = Matrix.hstack(*Ssym)
    out = {}
    for lam in IRREPS_I:
        dl = CHAR_I.dim(lam)
        P = sp.zeros(dd)
        for cls, chi in zip(CLASSES_I, CHAR_I.chars[lam]):
            if chi != 0:
                P += chi * class_sums[cls]
        P = (P * Rational(dl, 60)).applyfunc(sp.expand)
        rk = _rank_exact(P * Smat)
        assert rk % dl == 0, (name, lam, rk)
        if rk:
            out[lam] = rk // dl
    return out


# =========================================================================== quartic forms
def pair_index(d: int) -> List[Tuple[int, int]]:
    """The quadratic monomials eta_i eta_j, i <= j, in a fixed order."""
    return [(i, j) for i in range(d) for j in range(i, d)]


@lru_cache(maxsize=None)
def _K2():
    """The field Q(sqrt3, sqrt5) in which the H-channel forms live (sqrt3 from the
    normalisation of (x^2 - y^2)/2; all other channels stay inside Q(sqrt5))."""
    return QQ.algebraic_field(sqrt(3), sqrt(5))


@lru_cache(maxsize=None)
def _k2_sqrt5():
    return _K2().from_sympy(sqrt(5))


def _to_K2(e):
    """Fast conversion of a Q(sqrt5) field element / sympy number to Q(sqrt3, sqrt5)."""
    K2 = _K2()
    if hasattr(e, "rep") and not isinstance(e, sp.Basic):     # an element of K
        a, b = ab(e)
        return K2.convert(QQ(int(a.p), int(a.q))) + K2.convert(QQ(int(b.p), int(b.q))) * _k2_sqrt5()
    return K2.from_sympy(_c2(e))


def _dm2(rows) -> DomainMatrix:
    rows = [list(r) for r in rows]
    return DomainMatrix(rows, (len(rows), len(rows[0]) if rows else 0), _K2())


def _dm_K_to_K2(D: DomainMatrix) -> DomainMatrix:
    return _dm2([[_to_K2(e) for e in row] for row in D.to_list()])


def _c2(e) -> sp.Expr:
    """Canonical form of an element of Q(sqrt3, sqrt5): expand after rationalising."""
    return sp.expand(sp.radsimp(sp.expand(e)))


@lru_cache(maxsize=None)
def _k2_units():
    K2 = _K2()
    return K2.from_sympy(sqrt(3)), K2.from_sympy(sqrt(5)), K2.from_sympy(sqrt(15))


def _to_K2_fast(e) -> object:
    """Fast conversion of a real sympy number a + b sqrt3 + c sqrt5 + d sqrt15 (rational a..d)
    to Q(sqrt3, sqrt5); raises ValueError for anything else (e.g. complex numbers)."""
    K2 = _K2()
    e = _c2(sp.sympify(e))
    s3, s5, s15 = sqrt(3), sqrt(5), sqrt(15)
    b, c, d = e.coeff(s3), e.coeff(s5), e.coeff(s15)
    a = sp.expand(e - b * s3 - c * s5 - d * s15)
    parts = (a, b, c, d)
    if not all(x.is_Rational for x in parts):
        raise ValueError(f"{e} is not in Q(sqrt3, sqrt5)")
    u3, u5, u15 = _k2_units()
    conv = lambda r: K2.convert(QQ(int(r.p), int(r.q)))
    return conv(a) + conv(b) * u3 + conv(c) * u5 + conv(d) * u15


@dataclass(frozen=True)
class QuarticForm:
    """An I-invariant quartic form F(eta) = sum_{p,q} M[p,q] conj(eta^p) eta^q on the
    quadratic monomials eta^q = eta_i eta_j (i <= j), with M exact over Q(sqrt3, sqrt5)."""
    dim: int
    matrix: DomainMatrix

    @property
    def pairs(self):
        return pair_index(self.dim)

    def __add__(self, other: "QuarticForm") -> "QuarticForm":
        return QuarticForm(self.dim, self.matrix + other.matrix)

    def __sub__(self, other: "QuarticForm") -> "QuarticForm":
        return QuarticForm(self.dim, self.matrix - other.matrix)

    def scale(self, c) -> "QuarticForm":
        return QuarticForm(self.dim, self.matrix * _to_K2(c))

    def conj(self) -> "QuarticForm":
        """The form F(eta*) (eta <-> eta*), i.e. the transposed matrix."""
        return QuarticForm(self.dim, self.matrix.transpose())

    def is_zero(self) -> bool:
        return all(e == _K2().zero for row in self.matrix.to_list() for e in row)

    def flat(self) -> list:
        return [e for row in self.matrix.to_list() for e in row]

    def expr(self, eta, etab) -> sp.Expr:
        K2 = _K2()
        pairs = self.pairs
        terms = []
        for p, (k, l) in enumerate(pairs):
            for q, (i, j) in enumerate(pairs):
                c = self.matrix[p, q].element
                if c != K2.zero:
                    terms.append(K2.to_sympy(c) * etab[k] * etab[l] * eta[i] * eta[j])
        return sp.expand(sp.Add(*terms))

    def tensor(self) -> np.ndarray:
        """Q[i,j,k,l] with F = sum Q[i,j,k,l] eta_i eta_j etab_k etab_l, symmetric in (ij), (kl)."""
        d = self.dim
        pairs = self.pairs
        Q = np.zeros((d, d, d, d), dtype=complex)
        vals = self.matrix.to_list()
        for p, (k, l) in enumerate(pairs):
            for q, (i, j) in enumerate(pairs):
                c = complex(_K2().to_sympy(vals[p][q]))
                if c == 0:
                    continue
                pi = {(i, j), (j, i)}
                pk = {(k, l), (l, k)}
                w = c / (len(pi) * len(pk))
                for (a, b) in pi:
                    for (cc, dd) in pk:
                        Q[a, b, cc, dd] += w
        return Q

    def at(self, e: np.ndarray) -> complex:
        e = np.asarray(e, dtype=complex)
        return evaluate_form(self.tensor(), e)


def form_from_expr(expr, eta, etab) -> QuarticForm:
    """QuarticForm from a sympy quartic form (used for I1, I2)."""
    d = len(eta)
    pairs = pair_index(d)
    index = {pr: n for n, pr in enumerate(pairs)}
    K2 = _K2()
    M = [[K2.zero] * len(pairs) for _ in pairs]
    P = Poly(sp.expand(expr), *eta, *etab)
    for m, c in zip(P.monoms(), P.coeffs()):
        ii = tuple(i for i, e in enumerate(m[:d]) for _ in range(e))
        kk = tuple(k for k, e in enumerate(m[d:]) for _ in range(e))
        assert len(ii) == 2 and len(kk) == 2, m
        M[index[kk]][index[ii]] += _to_K2(c)
    return QuarticForm(d, _dm2(M))


def _pairing_matrix(Cf: DomainMatrix, Gram: DomainMatrix, Cg: DomainMatrix) -> DomainMatrix:
    """sum_{m,m'} Cf[m,p] Gram[m,m'] Cg[m',q]  (the form  int conj(f) g dOmega/4pi)."""
    return Cf.transpose() * Gram * Cg


@dataclass(frozen=True)
class QuarticInvariants:
    """Harmonic content of Delta^2 and the invariants N_L of a channel (legacy gl_inv2.py).

    ``components[L]`` is the coefficient matrix (degree-L monomials x quadratic eta-monomials)
    of [Delta^2]_L; ``N[L]`` the form int |[Delta^2]_L|^2 dOmega/4pi; ``quartic`` the form
    int |Delta|^4 dOmega/4pi; ``I1`` = (eta.eta*)^2, ``I2`` = |eta.eta|^2."""
    channel: str
    eta: Tuple[sp.Symbol, ...]
    etab: Tuple[sp.Symbol, ...]
    delta: sp.Expr
    components: Dict[int, DomainMatrix]
    N: Dict[int, QuarticForm]
    quartic: QuarticForm
    I1: QuarticForm
    I2: QuarticForm

    @property
    def gens(self):
        return tuple(self.eta) + tuple(self.etab)

    @property
    def dim(self) -> int:
        return len(self.eta)

    def component_expr(self, L: int) -> sp.Expr:
        """[Delta^2]_L as a polynomial in x, y, z with coefficients quadratic in eta."""
        K2 = _K2()
        pairs = pair_index(self.dim)
        C = self.components[L].to_list()
        mons = monomials(L)
        out = S(0)
        for m, row in zip(mons, C):
            coeff = sum(K2.to_sympy(c) * self.eta[i] * self.eta[j] for c, (i, j) in zip(row, pairs))
            out += coeff * m
        return sp.expand(out)

    def expr(self, name: str) -> sp.Expr:
        """Sympy expression of one of the forms: 'N0', 'N2', ..., 'quartic', 'I1', 'I2'."""
        if name.startswith("N"):
            return self.N[int(name[1:])].expr(self.eta, self.etab)
        return getattr(self, name).expr(self.eta, self.etab)


@lru_cache(maxsize=None)
def _gram_K2(l: int) -> DomainMatrix:
    return _dm_K_to_K2(sphere_gram(l))


@lru_cache(maxsize=None)
def _harmonic_projection_K2(l: int):
    from .harmonics import harmonic_projection_matrices
    Hp, Qp = harmonic_projection_matrices(l)
    conv = lambda M: _dm2([[_to_K2(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]) if M.rows and M.cols else None
    return conv(Hp), conv(Qp)


def _product_coefficients(ch: Channel) -> DomainMatrix:
    """Coefficient matrix (degree-2l monomials x pairs) of Delta^2 = sum_{i<=j} (2 - delta_ij)
    eta_i eta_j b^_i b^_j, with b^_i = b_i/sqrt(n_i).  The products of the unnormalised basis
    functions are formed first and scaled by 1/sqrt(n_i n_j), which lies in Q(sqrt5) for
    every channel (the individual sqrt(n_i) may not: sqrt7 for T2 and G)."""
    K2 = _K2()
    l = ch.l
    polys = [Poly(f, X, Y, Z, domain=K2) for f in ch.basis]
    exps = exponents(2 * l)
    index = {e: n for n, e in enumerate(exps)}
    pairs = pair_index(ch.dim)
    cols = []
    for (i, j) in pairs:
        scale = _to_K2(1 / sqrt(ch.norms2[i] * ch.norms2[j]))
        prod = polys[i] * polys[j]
        col = [K2.zero] * len(exps)
        for e, c in prod.as_dict(native=True).items():
            c = c * scale
            col[index[e]] = c if i == j else c + c
        cols.append(col)
    return _dm2([[cols[q][m] for q in range(len(pairs))] for m in range(len(exps))])


@lru_cache(maxsize=None)
def quartic_invariants(name: str) -> QuarticInvariants:
    """Harmonic decomposition of Delta^2 = sum_L [Delta^2]_L r^(2l-L) and the exact forms
    N_L = int |[Delta^2]_L|^2 dOmega/4pi, int |Delta|^4 = sum_L N_L (checked), I1, I2."""
    ch = channel(name)
    d, l = ch.dim, ch.l
    eta = sp.symbols(f"eta1:{d + 1}")
    etab = sp.symbols(f"etab1:{d + 1}")
    delta = sum(e * b for e, b in zip(eta, ch.onb))
    C = _product_coefficients(ch)                           # Delta^2 in degree-2l monomials
    comps = {}
    rest = C
    for L in range(2 * l, -1, -2):
        Hp, Qp = _harmonic_projection_K2(L)
        comps[L] = Hp * rest
        if L < 2:
            break
        rest = Qp * rest
        if all(e == _K2().zero for row in rest.to_list() for e in row):
            break
    N = {L: QuarticForm(d, _pairing_matrix(CL, _gram_K2(L), CL)) for L, CL in comps.items()}
    quartic = QuarticForm(d, _pairing_matrix(C, _gram_K2(2 * l), C))
    total = N[max(N)]
    for L in N:
        if L != max(N):
            total = total + N[L]
    assert (total - quartic).is_zero(), name
    I1 = form_from_expr(sp.expand(sum(e * eb for e, eb in zip(eta, etab)) ** 2), eta, etab)
    I2 = form_from_expr(sp.expand(sum(e * e for e in eta) * sum(eb * eb for eb in etab)), eta, etab)
    # |eta|^2 normalisation: the basis is orthogonal with the stated norms, so that
    # int |Delta|^2 dOmega/4pi = sum |eta_i|^2 for the normalised functions
    Bc = _dm2([[_to_K2(c) for c in row] for row in _coeff_matrix(ch.basis, l).tolist()])
    G2 = (Bc.transpose() * _gram_K2(l) * Bc).to_list()
    assert all(G2[i][j] == (_to_K2(ch.norms2[i]) if i == j else _K2().zero) for i in range(d) for j in range(d)), name
    return QuarticInvariants(name, eta, etab, delta, comps, N, quartic, I1, I2)


# --------------------------------------------------------------------------- relations
def express(target: QuarticForm, basis: Dict[str, QuarticForm]) -> Optional[Dict[str, sp.Expr]]:
    """Write target = sum_k c_k basis_k exactly (over Q(sqrt3, sqrt5)); {k: c_k} or None."""
    K2 = _K2()
    names = list(basis)
    cols = [basis[k].flat() for k in names]
    tvec = target.flat()
    A = DomainMatrix([[cols[j][i] for j in range(len(names))] for i in range(len(tvec))], (len(tvec), len(names)), K2)
    b = DomainMatrix([[t] for t in tvec], (len(tvec), 1), K2)
    try:
        x = ksolve(A, b)
    except ValueError:
        return None
    return {k: K2.to_sympy(x[i, 0].element) for i, k in enumerate(names)}


def rank_of_forms(forms: Sequence[QuarticForm]) -> int:
    K2 = _K2()
    cols = [f.flat() for f in forms]
    A = DomainMatrix([[cols[j][i] for j in range(len(forms))] for i in range(len(cols[0]))], (len(cols[0]), len(forms)), K2)
    return A.rank()


@lru_cache(maxsize=None)
def relations(name: str) -> Dict[str, Dict[str, sp.Expr]]:
    """N_L and int|Delta|^4 as exact linear combinations of I1, I2 (and N2 for G):
    Eqs. (18)-(22) of the paper (legacy gl_inv3.py)."""
    q = quartic_invariants(name)
    basis = {"I1": q.I1, "I2": q.I2}
    if name == "G":
        basis["N2"] = q.N[2]
    out = {}
    for L, NL in q.N.items():
        if name == "G" and L == 2:
            continue
        r = express(NL, basis)
        if r is not None:
            out[f"N{L}"] = r
    if name == "H":
        out["N2+N4"] = express(q.N[2] + q.N[4], basis)
    out["quartic"] = express(q.quartic, basis)
    return out


# --------------------------------------------------------------------------- legacy cross-check
def sphere_norm2_complex_expr(h, eta, etab) -> sp.Expr:
    """int |h|^2 dOmega/4pi for h with coefficients polynomial in eta (eta* -> etab), by direct
    symbolic expansion exactly as legacy gl_inv2.py (slow; used only as a cross-check)."""
    gens = tuple(eta) + tuple(etab)
    hb = h.subs(dict(zip(gens, tuple(etab) + tuple(eta))), simultaneous=True)
    P = Poly(sp.expand(h * hb), X, Y, Z)
    val = sp.expand(sum(c * mono_int(*m) for m, c in zip(P.monoms(), P.coeffs())))
    return sp.expand(val)


# =========================================================================== H channel
@dataclass(frozen=True)
class HChannel:
    """G/H split of [Delta^2]_4, intertwiner J, isometry scale and cross terms (legacy gl_H.py)."""
    inv: QuarticInvariants
    h4G: DomainMatrix          # coefficient matrices of the G and H parts of [Delta^2]_4
    h4H: DomainMatrix
    N4G: QuarticForm
    N4H: QuarticForm
    J: Matrix                  # 6 x 15, degree-4 -> degree-2 monomial coordinates
    Jh4H: DomainMatrix         # coefficients (degree-2 monomials x pairs) of J [Delta^2]_4H
    lam: sp.Expr               # |J h|^2 / |h|^2 on the H_4 subspace
    C: QuarticForm             # <[Delta^2]_2, J [Delta^2]_4H>
    CR: QuarticForm            # Re C
    CIi: QuarticForm           # i Im C
    six_independent: bool

    def expr(self, name: str) -> sp.Expr:
        return getattr(self, name).expr(self.inv.eta, self.inv.etab)

    def component_expr(self, which: str) -> sp.Expr:
        """'h4G', 'h4H' (degree 4) or 'Jh4H' (degree 2) as polynomials in x, y, z."""
        K2 = _K2()
        q = self.inv
        pairs = pair_index(q.dim)
        M = getattr(self, which)
        L = 2 if which == "Jh4H" else 4
        out = S(0)
        for m, row in zip(monomials(L), M.to_list()):
            out += m * sum(K2.to_sympy(c) * q.eta[i] * q.eta[j] for c, (i, j) in zip(row, pairs))
        return sp.expand(out)


def _legacy_seed_matrix() -> Matrix:
    """The generic 6 x 15 seed of the intertwiner, exactly as legacy gl_H.py draws it:
    ``random.seed(3)``; entries Rational(randint(-4,4), randint(1,3)) in row-major order.
    A private ``random.Random(3)`` yields the identical sequence without reseeding the
    global generator."""
    rng = random.Random(3)
    rows = []
    for i in range(6):
        row = []
        for j in range(15):
            p = rng.randint(-4, 4)
            qd = rng.randint(1, 3)
            row.append(Rational(p, qd))
        rows.append(row)
    return Matrix(rows)


@lru_cache(maxsize=None)
def intertwiner_J() -> DomainMatrix:
    """J = (1/60) sum_g rho2(g) L rho4(g^-1): the I-equivariant map from degree-4 to degree-2
    polynomials (monomial coordinates) obtained by Schur averaging of the legacy seed L.
    Equivariance J rho4(g) = rho2(g) J is checked for all g."""
    G = icosahedral_group()
    Lk = DomainMatrix([[to_K(e) for e in row] for row in _legacy_seed_matrix().tolist()], (6, 15), K)
    reps2, reps4 = rep_matrices(2), rep_matrices(4)
    Jk = None
    for i in range(60):
        term = reps2[i] * Lk * reps4[G.inverse[i]]
        Jk = term if Jk is None else Jk + term
    Jk = Jk * to_K(Rational(1, 60))
    for i in range(60):
        assert (Jk * reps4[i]).to_list() == (reps2[i] * Jk).to_list(), "J not equivariant"
    return Jk


@lru_cache(maxsize=None)
def h_channel() -> HChannel:
    q = quartic_invariants("H")
    d = q.dim
    PG4 = _dm_K_to_K2(projector(4, "G"))
    PH4 = _dm_K_to_K2(projector(4, "H"))
    h4 = q.components[4]
    h4G, h4H = PG4 * h4, PH4 * h4
    assert (h4G + h4H).to_list() == h4.to_list()
    N4G = QuarticForm(d, _pairing_matrix(h4G, _gram_K2(4), h4G))
    N4H = QuarticForm(d, _pairing_matrix(h4H, _gram_K2(4), h4H))
    assert (N4G + N4H - q.N[4]).is_zero()
    Jk = intertwiner_J()
    J2 = _dm_K_to_K2(Jk)
    Jh4H = J2 * h4H
    assert any(e != _K2().zero for row in Jh4H.to_list() for e in row)
    # the image is harmonic (lands in the l = 2 harmonics): Laplacian of every coefficient vanishes
    lap2 = _dm_K_to_K2(laplacian_matrix(2))
    assert all(e == _K2().zero for row in (lap2 * Jh4H).to_list() for e in row)
    # isometry scale lambda: |J h|^2 = lambda |h|^2 identically on the H_4 subspace (Schur)
    JN = QuarticForm(d, _pairing_matrix(Jh4H, _gram_K2(2), Jh4H))
    lam = None
    for p in range(len(N4H.flat())):
        den = N4H.flat()[p]
        if den != _K2().zero:
            lam = JN.flat()[p] / den
            break
    assert lam is not None
    assert (JN - N4H.scale(_K2().to_sympy(lam))).is_zero(), "J is not an isometry up to a scalar"
    lam = _c2(_K2().to_sympy(lam))
    # cross term C = <[Delta^2]_2, J [Delta^2]_4H>
    C = QuarticForm(d, _pairing_matrix(q.components[2], _gram_K2(2), Jh4H))
    CR = (C + C.conj()).scale(Rational(1, 2))
    CIi = (C - C.conj()).scale(Rational(1, 2))
    six = [q.I1, q.I2, q.N[2], N4G, CR, CIi]
    independent = rank_of_forms(six) == 6
    return HChannel(q, h4G, h4H, N4G, N4H, to_matrix(Jk), Jh4H, lam, C, CR, CIi, independent)


# =========================================================================== numerics
def evaluate_form(Q: np.ndarray, e: np.ndarray) -> complex:
    ec = np.conj(e)
    return np.einsum("ijkl,i,j,k,l", Q, e, e, ec, ec)


def _ratio_and_grad(Q: np.ndarray, v: np.ndarray):
    d = Q.shape[0]
    e = v[:d] + 1j * v[d:]
    ec = np.conj(e)
    F = np.einsum("ijkl,i,j,k,l", Q, e, e, ec, ec).real
    n = np.vdot(e, e).real
    Gk = 2 * np.einsum("ijkl,i,j,l->k", Q, e, e, ec)      # dF/d etab_k
    dF = np.concatenate([2 * Gk.real, 2 * Gk.imag])
    dn = 2 * v
    R = F / n**2
    dR = dF / n**2 - 2 * F * dn / n**3
    return R, dR


@dataclass(frozen=True)
class Minimum:
    channel: str
    value: float
    eta: np.ndarray                    # normalised
    I2: float                          # |eta.eta|^2 at the minimum
    all_values: np.ndarray             # sorted values of all restarts
    fraction_at_min: float


def weak_coupling_ratio(name: str, e: np.ndarray) -> float:
    """R = int|Delta|^4 / (int|Delta|^2)^2 at the order parameter e (any normalisation)."""
    Q = _tensor(name)
    e = np.asarray(e, dtype=complex)
    n = np.vdot(e, e).real
    return float(evaluate_form(Q, e).real / n**2)


def weak_coupling_ratio_exact(name: str, e: Sequence) -> sp.Expr:
    """R = int|Delta|^4 / (int|Delta|^2)^2 at an *exact* order parameter e (sympy numbers,
    e.g. ``(1, 0, 0)`` or ``(1, I, 0)``; any normalisation), evaluated in exact arithmetic:
    the quartic form at (e, conj e) divided by (e.conj e)^2.  The real state (1, 0, ...) and
    the null-cone state (1, i, 0, ...) give the closed forms 9/5, 6/5 (T1); 5061/2145,
    3374/2145 (T2); 15/7, 10/7 (H) of Eq. (22).  Real vectors with components in
    Q(sqrt3, sqrt5) are evaluated through the exact form matrix (fast); other exact vectors
    (e.g. complex) through sympy substitution."""
    q = quartic_invariants(name)
    vals = [sp.sympify(x) for x in e]
    if len(vals) != q.dim:
        raise ValueError(f"{name}: expected {q.dim} components, got {len(vals)}")
    try:
        ke = [_to_K2_fast(v) for v in vals]
    except ValueError:
        ke = None
    if ke is not None:                                   # real vector: bilinear form over K2
        K2 = _K2()
        pairs = pair_index(q.dim)
        mon = [ke[i] * ke[j] for (i, j) in pairs]
        M = q.quartic.matrix.to_list()
        num = K2.zero
        for pi, mp in enumerate(mon):
            for qi, mq in enumerate(mon):
                num += mp * M[pi][qi] * mq
        norm2 = sum((x * x for x in ke), K2.zero)
        return _c2(K2.to_sympy(num / (norm2 * norm2)))
    sub = {**dict(zip(q.eta, vals)), **{s: sp.conjugate(v) for s, v in zip(q.etab, vals)}}
    num = sp.expand(q.expr("quartic").subs(sub, simultaneous=True))
    norm2 = sp.expand(sum(v * sp.conjugate(v) for v in vals))
    return _c2(num / norm2**2)


@lru_cache(maxsize=None)
def _matrices_K2(name: str) -> Tuple[DomainMatrix, ...]:
    """The exact irrep matrices D(g) of a channel as DomainMatrices over Q(sqrt3, sqrt5)."""
    ch = channel(name)
    return tuple(_dm2([[_to_K2_fast(D[i, j]) for j in range(ch.dim)] for i in range(ch.dim)])
                 for D in ch.matrices)


def fixed_space_exact(name: str, subgroup: str, character: str) -> List[List[sp.Expr]]:
    """The fixed space {eta : D(g) eta = chi(g) eta, g in K} computed EXACTLY (null space over
    Q(sqrt3, sqrt5)) for a one-dimensional character with values +-1 (D2 A/B1/B2/B3, D3 and D5
    A1/A2, T A, I A); returns a basis as lists of exact numbers.  Complex characters (C_n chi_m,
    T 1E/2E) would need the cyclotomic field and raise ValueError."""
    chi = dict(characters_1d(subgroup))[character]
    if any(abs(v.imag) > 1e-12 or abs(abs(v) - 1) > 1e-12 for v in chi.values()):
        raise ValueError(f"{subgroup} {character}: only real (+-1) characters are handled exactly")
    K2 = _K2()
    d = channel(name).dim
    Ds = _matrices_K2(name)
    rows = []
    for g, v in chi.items():
        sgn = K2.one if v.real > 0 else -K2.one
        Dg = Ds[g].to_list()
        for i in range(d):
            rows.append([Dg[i][j] - (sgn if i == j else K2.zero) for j in range(d)])
    N = _dm2(rows).nullspace()
    return [[K2.to_sympy(x) for x in row] for row in N.to_list()]


def exact_ratio_of_fixed_state(name: str, subgroup: str, character: str) -> Optional[sp.Expr]:
    """R_wc of the symmetry-fixed state (K, chi) in exact arithmetic, when chi is real and the
    fixed space is one-dimensional; None otherwise (the C_n chi_m and T 1E/2E states)."""
    try:
        basis = fixed_space_exact(name, subgroup, character)
    except ValueError:
        return None
    if len(basis) != 1:
        return None
    return weak_coupling_ratio_exact(name, basis[0])


@lru_cache(maxsize=None)
def _tensor(name: str) -> np.ndarray:
    return quartic_invariants(name).quartic.tensor()


def minimise_ratio(name: str, restarts: int = 60, seed: int = 0, gtol: float = 1e-10,
                   null_cone_penalty: float = 0.0) -> Minimum:
    """Minimise R = int|Delta|^4/(int|Delta|^2)^2 over eta by BFGS from Gaussian random
    starts (numpy seed as given; legacy gl_min.py uses 60 restarts, seed 0).  With
    ``null_cone_penalty`` > 0 the term penalty*|eta.eta|^2/|eta|^4 is added (legacy
    gl_states.py, penalty 50) to find the best state on the null cone."""
    from scipy.optimize import minimize

    Q = _tensor(name)
    d = Q.shape[0]
    rng = np.random.RandomState(seed)

    def fun(v):
        R, dR = _ratio_and_grad(Q, v)
        if null_cone_penalty:
            e = v[:d] + 1j * v[d:]
            n = np.vdot(e, e).real
            s = np.dot(e, e)
            R = R + null_cone_penalty * abs(s) ** 2 / n**2
            # |s|^2 = s sbar ; d/d e_k = 2 e_k sbar ; d/d ebar_k = 2 ebar_k s
            dA = 2 * np.conj(e) * s           # d/d ebar_k
            ds = np.concatenate([2 * dA.real, 2 * dA.imag])
            dn = 2 * v
            dR = dR + null_cone_penalty * (ds / n**2 - 2 * abs(s) ** 2 * dn / n**3)
        return R, dR

    results = []
    for _ in range(restarts):
        v0 = rng.randn(2 * d)
        res = minimize(fun, v0, jac=True, method="BFGS", options={"gtol": gtol})
        results.append((res.fun, res.x))
    results.sort(key=lambda t: t[0])
    vals = np.array([r[0] for r in results])
    e = results[0][1][:d] + 1j * results[0][1][d:]
    e /= np.linalg.norm(e)
    Rbest = weak_coupling_ratio(name, e)
    return Minimum(name, float(Rbest), e, float(abs(np.dot(e, e)) ** 2), vals,
                   float(np.mean(vals < vals[0] + 1e-6)))


def stabiliser(name: str, e: np.ndarray) -> Tuple[Counter, Counter]:
    """Elements g of I with D(g) e = phase * e (and with D(g) e = phase * e* for the
    time-reversal-combined stabiliser), counted by class."""
    ch = channel(name)
    Dn = ch.numeric_matrices()
    G = icosahedral_group()
    e = np.asarray(e, dtype=complex)
    e = e / np.linalg.norm(e)
    st, tst = Counter(), Counter()
    for r, D in zip(G.rotations, Dn):
        if abs(np.vdot(e, D @ e)) > 1 - _TOL:
            st[r.cls] += 1
        if abs(np.vdot(np.conj(e), D @ e)) > 1 - _TOL:
            tst[r.cls] += 1
    return st, tst


def _sphere_grid(n_theta: int, n_phi: int, endpoint: bool = True):
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=endpoint)
    T, P = np.meshgrid(th, ph, indexing="ij")
    return T, P, np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)


def gap_on_grid(name: str, e: np.ndarray, n_theta: int = 300, n_phi: int = 600):
    ch = channel(name)
    fs = ch.numeric_basis()
    T, P, Xg, Yg, Zg = _sphere_grid(n_theta, n_phi)
    D = sum(c * f(Xg, Yg, Zg) for c, f in zip(e, fs))
    return T, P, Xg, Yg, Zg, np.abs(D)


def min_gap(name: str, e: np.ndarray, n_theta: int = 300, n_phi: int = 600) -> float:
    """min|Delta|/max|Delta| on a theta-phi grid (legacy isotropy.py: 300 x 600)."""
    *_, A = gap_on_grid(name, e, n_theta, n_phi)
    return float(A.min() / A.max())


def node_fraction(name: str, e: np.ndarray, ngrid: int = 400, thresh: float = 0.02):
    """(min|Delta|/max, area fraction with |Delta| < 2% of max) (legacy gl_states.py)."""
    T, P, *_, A = gap_on_grid(name, e, ngrid, 2 * ngrid)
    A = A / A.max()
    w = np.sin(T)
    return float(A.min()), float(np.sum(w * (A < thresh)) / np.sum(w))


# --------------------------------------------------------------------------- G ground state
def g_ground_state(restarts: int = 80, seed: int = 1) -> Dict[str, object]:
    """Unconstrained and null-cone-constrained minima of R for G (legacy gl_states.py)."""
    m = minimise_ratio("G", restarts=restarts, seed=seed, gtol=1e-12)
    st, tst = stabiliser("G", m.eta)
    mn, fr = node_fraction("G", m.eta)
    # null-cone local minimum: continue the same random stream as the legacy script
    rng = np.random.RandomState(seed)
    for _ in range(restarts):
        rng.randn(8)
    from scipy.optimize import minimize
    Q = _tensor("G")
    best2 = None
    for _ in range(restarts):
        v0 = rng.randn(8)

        def fun(v):
            e = v[:4] + 1j * v[4:]
            n = np.vdot(e, e).real
            R, dR = _ratio_and_grad(Q, v)
            s = np.dot(e, e)
            dA = 2 * np.conj(e) * s
            ds = np.concatenate([2 * dA.real, 2 * dA.imag])
            return R + 50 * abs(s) ** 2 / n**2, dR + 50 * (ds / n**2 - 2 * abs(s) ** 2 * (2 * v) / n**3)

        res = minimize(fun, v0, jac=True, method="BFGS", options={"gtol": 1e-12})
        if best2 is None or res.fun < best2.fun:
            best2 = res
    e2 = best2.x[:4] + 1j * best2.x[4:]
    e2 /= np.linalg.norm(e2)
    st2, tst2 = stabiliser("G", e2)
    return {"R": m.value, "eta": m.eta, "moduli": np.abs(m.eta), "I2": m.I2,
            "stabiliser": dict(st), "stabiliser_order": sum(st.values()),
            "tr_stabiliser": dict(tst), "min_gap": mn, "node_area_fraction": fr,
            "null_cone_R": weak_coupling_ratio("G", e2), "null_cone_eta": e2,
            "null_cone_I2": float(abs(np.dot(e2, e2)) ** 2),
            "null_cone_stabiliser": dict(st2), "null_cone_tr_stabiliser": dict(tst2),
            "fraction_of_restarts_at_min": m.fraction_at_min}


@lru_cache(maxsize=None)
def g_stratum() -> Dict[str, object]:
    """R on the C3 stratum eta = (1, 1, 1, kappa e^{i phi}) of G (legacy gl_final.py):
    the exact function R(kappa, phi), its numerical minimum (Nelder-Mead from (1.7, pi/2)),
    and the exact stationary points at phi = pi/2 in u = kappa^2."""
    from scipy.optimize import minimize

    q = quartic_invariants("G")
    kap, ph = sp.symbols("kappa phi_", real=True)
    quartic = q.expr("quartic")
    sub = {q.eta[0]: 1, q.eta[1]: 1, q.eta[2]: 1, q.eta[3]: kap * sp.exp(IMAG * ph),
           q.etab[0]: 1, q.etab[1]: 1, q.etab[2]: 1, q.etab[3]: kap * sp.exp(-IMAG * ph)}
    Qk = sp.expand(quartic.subs(sub))
    Nk = (3 + kap**2) ** 2
    Rk = Qk / Nk
    Rk_cos = sp.simplify(sp.expand(Rk.rewrite(sp.cos)))
    f = lambdify((kap, ph), Rk, "numpy")
    res = minimize(lambda v: float(np.real(f(v[0], v[1]))), [1.7, np.pi / 2], method="Nelder-Mead",
                   options={"xatol": 1e-12, "fatol": 1e-14})
    u = sp.symbols("u", positive=True)
    Qu_k = sp.expand(Qk.subs(ph, sp.pi / 2))
    Pk = Poly(Qu_k, kap)
    assert all(m[0] % 2 == 0 for m in Pk.monoms()), "not even in kappa"
    Qu = sp.expand(sum(_c2(c) * u ** (m[0] // 2) for m, c in zip(Pk.monoms(), Pk.coeffs())))
    Ru = Qu / (3 + u) ** 2
    dR = sp.together(sp.diff(Ru, u))
    num = sp.numer(dR)
    sols = [s for s in sp.solve(sp.expand(num), u) if s.is_real and s > 0]
    Rvals = [_c2(Ru.subs(u, s)) for s in sols]
    e = np.array([1, 1, 1, res.x[0] * np.exp(1j * res.x[1])])
    e /= np.linalg.norm(e)
    return {"R_expr": Rk_cos, "kappa": float(res.x[0]), "phi0_deg": float(np.degrees(res.x[1]) % 360),
            "R": float(res.fun), "I2_over_I1": float(abs(np.dot(e, e)) ** 2),
            "R_u": sp.factor(Ru), "stationary_u": sols, "stationary_R": Rvals,
            "R_check": weak_coupling_ratio("G", e)}


def g_nodes(kappa: Optional[float] = None, n_theta: int = 1200, thresh: float = 0.01,
            cluster_radius: float = 0.08) -> Dict[str, object]:
    """Point nodes of the C3-stratum state eta = (1, 1, 1, i kappa) found on a 1200 x 2400 grid
    (|Delta| < 1% of max, clustered with radius 0.08), and how many lie on five-fold axes.
    By default kappa = sqrt(108/35) = 1.7566, the exact stationary point of R on the
    phi = pi/2 line (R = 15505/10153 = 1.52713), exactly as legacy gl_final.py -- not the
    global minimum kappa = 1.752, phi0 = 92.6 deg of ``g_stratum`` (R = 1.52545); pass
    ``kappa`` to use another point.  Both give the same 18 point nodes, 12 on five-fold
    axes (Section 6.3 of the paper)."""
    from .groups import fivefold_axes

    if kappa is None:
        st = g_stratum()
        kappa = float(sp.sqrt(st["stationary_u"][0])) if st["stationary_u"] else st["kappa"]
    e = np.array([1, 1, 1, 1j * kappa])
    e /= np.linalg.norm(e)
    T, P, Xg, Yg, Zg, A = gap_on_grid("G", e, n_theta, 2 * n_theta)
    A = A / A.max()
    idx = np.argwhere(A < thresh)
    pts = np.array([[Xg[i, j], Yg[i, j], Zg[i, j]] for i, j in idx])
    clusters: List[List[np.ndarray]] = []
    for p in pts:
        for c in clusters:
            if np.linalg.norm(p - c[0]) < cluster_radius:
                c.append(p)
                break
        else:
            clusters.append([p])
    centres = [np.mean(c, axis=0) for c in clusters]
    centres = [c / np.linalg.norm(c) for c in centres]
    axes = []
    for a in fivefold_axes():
        v = np.array([as_float(x) for x in a])
        v /= np.linalg.norm(v)
        axes += [v, -v]
    on_axis = sum(1 for c in centres if min(np.linalg.norm(c - a) for a in axes) < 0.05)
    return {"kappa": kappa, "n_nodes": len(clusters), "centres": centres, "on_fivefold_axes": on_axis}


# --------------------------------------------------------------------------- isotropy
_D2_LABEL = {(1, 1, 1): "A", (-1, -1, 1): "B1", (-1, 1, -1): "B2", (1, -1, -1): "B3"}


@lru_cache(maxsize=None)
def subgroups() -> Dict[str, Tuple[int, ...]]:
    """C2 (about z), C3 (about (1,1,1)), C5 (about (phi,1,0)), D2, D3, D5, T, I as index sets."""
    G = icosahedral_group()
    kz = rotation_about((0, 0, 1), "C2").index
    kx = rotation_about((1, 0, 0), "C2").index
    k3 = rotation_about((1, 1, 1), "C3").index
    k5 = rotation_about((PHI, 1, 0), "C5").index
    return {"C2": generate_subgroup([kz]), "C3": generate_subgroup([k3]), "C5": generate_subgroup([k5]),
            "D2": generate_subgroup([kz, kx]), "D3": generate_subgroup([k3, _perp2(k3)]),
            "D5": generate_subgroup([k5, _perp2(k5)]), "T": generate_subgroup([kz, kx, k3]),
            "I": tuple(range(60))}


def _perp2(k: int) -> int:
    """A two-fold rotation whose axis is perpendicular to that of rotation k (first in index order)."""
    G = icosahedral_group()
    a = G.rotations[k].axis
    for i in G.classes["C2"]:
        b = G.rotations[i].axis
        if canon((a.T * b)[0]) == 0:
            return i
    raise ValueError


def _generators() -> Dict[str, int]:
    return {"C2": rotation_about((0, 0, 1), "C2").index, "C3": rotation_about((1, 1, 1), "C3").index,
            "C5": rotation_about((PHI, 1, 0), "C5").index, "C2x": rotation_about((1, 0, 0), "C2").index,
            "C2y": rotation_about((0, 1, 0), "C2").index}


def characters_1d(name: str) -> List[Tuple[str, Dict[int, complex]]]:
    """The one-dimensional characters of the subgroup (as dicts index -> value), labelled:
    C_n: chi_m (m mod n); D3, D5: A1, A2; D2: A, B1, B2, B3 (invariant under C2z, C2y, C2x);
    T: A, 1E, 2E; I: A."""
    G = icosahedral_group()
    elems = subgroups()[name]
    gen = _generators()
    chars = []
    if name in ("C2", "C3", "C5"):
        n = len(elems)
        g = gen[name]
        for m in range(n):
            chi, el = {}, 0
            for p in range(n):
                chi[el] = np.exp(2j * np.pi * m * p / n)
                el = G.mult[el][g]
            chars.append((f"chi{m}", chi))
    elif name in ("D3", "D5"):
        n = len(elems) // 2
        g = gen["C" + name[1]]
        h = _perp2(g)
        for s in (1, -1):
            chi = {}
            for p in range(n):
                el = 0
                for _ in range(p):
                    el = G.mult[el][g]
                chi[el] = 1.0
                chi[G.mult[el][h]] = float(s)
            chars.append(("A1" if s == 1 else "A2", chi))
    elif name == "D2":
        kz, kx, ky = gen["C2"], gen["C2x"], gen["C2y"]
        for sx, sy in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
            chi = {0: 1.0, kx: float(sx), ky: float(sy), kz: float(sx * sy)}
            chars.append((_D2_LABEL[(sx, sy, sx * sy)], chi))
    elif name == "T":
        kz, kx, k3 = gen["C2"], gen["C2x"], gen["C3"]
        D2 = generate_subgroup([kz, kx])
        for m in range(3):
            w = np.exp(2j * np.pi * m / 3)
            chi = {}
            for p in range(3):
                c = 0
                for _ in range(p):
                    c = G.mult[c][k3]
                for d_ in D2:
                    chi[G.mult[d_][c]] = w**p
            chars.append((["A", "1E", "2E"][m], chi))
    elif name == "I":
        chars.append(("A", {g: 1.0 for g in elems}))
    for nm, chi in chars:
        assert set(chi) == set(elems), (name, nm)
        for a in elems:
            for b in elems:
                assert abs(chi[a] * chi[b] - chi[G.mult[a][b]]) < 1e-9, (name, nm)
    return chars


@dataclass(frozen=True)
class FixedState:
    channel: str
    subgroup: str
    character: str
    dim_fixed: int
    eta: Optional[np.ndarray]
    R: Optional[float]
    I2: Optional[float]
    time_reversal: Optional[bool]        # I2 = I1 at |eta| = 1 (the paper's TR column)
    tr_up_to_rotation: Optional[bool]    # exists g in I with eta* = phase * D(g) eta (legacy flag)
    min_gap: Optional[float]
    stabiliser_order: Optional[int]


def isotropy_table(channels: Sequence[str] = CHANNELS) -> List[FixedState]:
    """For every subgroup K and one-dimensional character chi, the fixed space
    {eta : D(g) eta = chi(g) eta, g in K}; for one-dimensional fixed spaces the state, its
    weak-coupling ratio R, |eta.eta|^2, time-reversal flag and min|Delta|/max
    (legacy isotropy.py; Table 7)."""
    out = []
    for name in channels:
        ch = channel(name)
        Dn = ch.numeric_matrices()
        d = ch.dim
        for sub, elems in subgroups().items():
            for cname, chi in characters_1d(sub):
                A = np.vstack([Dn[g] - chi[g] * np.eye(d) for g in elems])
                u, s, vh = np.linalg.svd(A)
                null = vh[np.sum(s > _TOL):].conj().T
                k = null.shape[1]
                if k == 0:
                    continue
                if k == 1:
                    e = null[:, 0]
                    e = e / np.linalg.norm(e)
                    R = weak_coupling_ratio(name, e)
                    I2 = float(abs(np.dot(e, e)) ** 2)
                    trs = any(abs(np.vdot(np.conj(e), D @ e)) > 1 - _TOL for D in Dn)
                    mg = min_gap(name, e)
                    st, _ = stabiliser(name, e)
                    out.append(FixedState(name, sub, cname, 1, e, float(R), I2, bool(abs(I2 - 1) < 1e-6),
                                          bool(trs), mg, sum(st.values())))
                else:
                    out.append(FixedState(name, sub, cname, k, None, None, None, None, None, None, None))
    return out


def symmetry_fixed_states(channels: Sequence[str] = CHANNELS) -> List[FixedState]:
    """The entries of Table 7: one-dimensional fixed spaces whose state has K as its full
    stabiliser in I (up to a phase), i.e. K is the isotropy subgroup and not a proper
    subgroup of it (e.g. the polar T1 state fixed by C5 chi0 is listed under D5 A2)."""
    return [r for r in isotropy_table(channels)
            if r.dim_fixed == 1 and r.stabiliser_order == len(subgroups()[r.subgroup])]


# --------------------------------------------------------------------------- H candidates
def _frame(a: np.ndarray):
    t = np.cross(a, [0.3, 0.5, 0.7])
    t /= np.linalg.norm(t)
    u = np.cross(a, t)
    return t, u


def _Y22(a):
    t, u = _frame(a)
    return lambda Xg, Yg, Zg: ((Xg * t[0] + Yg * t[1] + Zg * t[2]) + 1j * (Xg * u[0] + Yg * u[1] + Zg * u[2])) ** 2


def _Y21(a):
    t, u = _frame(a)
    return lambda Xg, Yg, Zg: (Xg * a[0] + Yg * a[1] + Zg * a[2]) * ((Xg * t[0] + Yg * t[1] + Zg * t[2]) + 1j * (Xg * u[0] + Yg * u[1] + Zg * u[2]))


def h_coordinates(func, n: int = 300) -> np.ndarray:
    """Coordinates of a function on the sphere in the orthonormal H basis by the legacy
    quadrature (theta-phi grid n x 2n, weights sin(theta) (pi/n)^2)."""
    ch = channel("H")
    fs = ch.numeric_basis()
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    w = np.sin(T) * (np.pi / n) * (np.pi / n)
    Xg, Yg, Zg = np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)
    D = func(Xg, Yg, Zg)
    return np.array([np.sum(w * np.conj(f(Xg, Yg, Zg)) * D) / (4 * np.pi) for f in fs])


def candidate_axes() -> Dict[str, np.ndarray]:
    phi = (1 + 5**0.5) / 2
    return {"C5": np.array([phi, 1, 0]) / np.sqrt(phi**2 + 1), "C3": np.array([1, 1, 1]) / np.sqrt(3),
            "C2": np.array([0, 0, 1.0])}


@lru_cache(maxsize=None)
def _h_numeric_forms():
    H = h_channel()
    q = H.inv
    forms = {"I1": q.I1, "I2": q.I2, "N2": q.N[2], "N4": q.N[4], "N4G": H.N4G, "N4H": H.N4H,
             "CR": H.CR, "CIi": H.CIi}
    return {k: v.tensor() for k, v in forms.items()}, float(sp.N(H.lam))


def h_invariants_at(e: np.ndarray) -> Dict[str, float]:
    """The quartic invariants of H at a normalised eta; Re C, Im C with J an isometry."""
    Fs, lam = _h_numeric_forms()
    e = np.asarray(e, dtype=complex)
    e = e / np.linalg.norm(e)
    v = {k: complex(evaluate_form(Q, e)) for k, Q in Fs.items()}
    return {"I1": v["I1"].real, "I2": v["I2"].real, "N2": v["N2"].real, "N4": v["N4"].real,
            "N4G": v["N4G"].real, "N4H": v["N4H"].real,
            "ReC": v["CR"].real / np.sqrt(lam), "ImC": (v["CIi"] / 1j).real / np.sqrt(lam)}


def h_candidates() -> Dict[str, Dict[str, object]]:
    """The null-cone H states of Table 8: Y2,+-2 and Y2,+-1 about the C5, C3, C2 axes and the
    cyclic state x^2 + w y^2 + w^2 z^2 (both chiralities), with N2, N4G, N4H, Re C, Im C at
    |eta| = 1, the stabiliser order and min|Delta|/max (legacy gl_final.py / gl_states.py).
    The real uniaxial state 3(a.n)^2 - 1 about C5 (I2 = 1, computed by legacy gl_states.py) is
    appended for comparison; it is not a null-cone state and not part of Table 8."""
    om = np.exp(2j * np.pi / 3)
    cands = {}
    for k, a in candidate_axes().items():
        cands[f"Y22 about {k}"] = _Y22(a)
        cands[f"Y21 about {k}"] = _Y21(a)
    cands["cyclic (T)"] = lambda Xg, Yg, Zg: Xg**2 + om * Yg**2 + om**2 * Zg**2
    cands["cyclic (T), other chirality"] = lambda Xg, Yg, Zg: Xg**2 + om**2 * Yg**2 + om * Zg**2
    a5 = candidate_axes()["C5"]
    cands["real uniaxial 3(a.n)^2-1 about C5"] = lambda Xg, Yg, Zg: 3 * (Xg * a5[0] + Yg * a5[1] + Zg * a5[2]) ** 2 - 1
    rows = {}
    for name, fn in cands.items():
        e = h_coordinates(fn)
        e /= np.linalg.norm(e)
        inv = h_invariants_at(e)
        st, tst = stabiliser("H", e)
        rows[name] = {**inv, "eta": e, "stabiliser_order": sum(st.values()),
                      "min_gap": min_gap("H", e)}
    return rows


# =========================================================================== summaries
def sym2_table() -> Dict[str, Dict[str, object]]:
    return {name: {"sym2": sym2_by_characters(name), "hermitian_forms": hermitian_form_count(name),
                   "quartic_terms": tr_even_quartic_count(name)} for name in CHANNELS}


def format_decomposition(dec: Dict[str, int], sep: str = " + ") -> str:
    return sep.join(f"{m if m > 1 else ''}{k}" for k, m in dec.items() if m)
