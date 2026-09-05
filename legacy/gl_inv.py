"""
gl_inv.py -- explicit quartic Ginzburg-Landau invariants for the icosahedral pairing channels
T1, T2, G, H. Exact arithmetic. Builds explicit orthogonal irrep matrices D(g) from the
harmonic-polynomial realisations (orthonormal on the unit sphere), then decomposes the symmetric
square Sym^2(Gamma) into isotypic components and writes the invariants as squared norms of the
isotypic components of the pair tensor eta_i eta_j.
"""
import sys, itertools
import sympy as sp
from sympy import sqrt, Rational, Matrix, symbols, expand, Poly, S, pi, factorial2
src = open('ih_basis.py').read().split('# ---------------------------------------------------------------- action on harmonic polynomials')[0]
src = src.replace("LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6", "LMAX=0")
exec(src)
phi_ = symbols('varphi')
def to_phi(e):
    e = sp.nsimplify(sp.expand(S(e)), [sqrt(5)])
    if not e.has(sqrt(5)): return e
    p = e.subs(sqrt(5), 0); q = sp.nsimplify(sp.simplify(sp.expand(e - p)/sqrt(5)))
    return sp.nsimplify(p - q) + 2*q*phi_

# ---- sphere inner product of homogeneous polynomials (exact): int x^a y^b z^c dOmega
def mono_int(a, b, c):
    if a % 2 or b % 2 or c % 2: return S(0)
    return 4*pi*factorial2(a-1)*factorial2(b-1)*factorial2(c-1)/factorial2(a+b+c+1)
def sph_inner(f, g):
    P = Poly(expand(f*g), x, y, z)
    return sp.nsimplify(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi), [sqrt(5)])

def act(R, f):
    Rt = R.T
    return expand(f.subs({x: Rt[0,0]*x+Rt[0,1]*y+Rt[0,2]*z, y: Rt[1,0]*x+Rt[1,1]*y+Rt[1,2]*z, z: Rt[2,0]*x+Rt[2,1]*y+Rt[2,2]*z}, simultaneous=True))

def chi_of(R, irr):
    c = cos_theta(R)
    for (cc, _), val in zip(classes, chi[irr]):
        if c == cc: return val

# ---- channel bases (from ih_seeds output; orthogonal on the sphere by parity arguments)
bases = {
 'T1': [x, y, z],
 'H' : [x*y, y*z, z*x, (x**2-y**2)/2, (2*z**2-x**2-y**2)/(2*sqrt(3))],
 'T2': [x**3 + 3*(phi-1)*x*y**2 - 3*phi*x*z**2,
        y**3 + 3*(phi-1)*y*z**2 - 3*phi*y*x**2,
        z**3 + 3*(phi-1)*z*x**2 - 3*phi*z*y**2],
 'G' : [x**3 - phi**2*x*y**2 - x*z**2/phi**2,
        y**3 - phi**2*y*z**2 - y*x**2/phi**2,
        z**3 - phi**2*z*x**2 - z*y**2/phi**2,
        x*y*z],
}
results = {}
for irr, B in bases.items():
    d = len(B)
    Gram = Matrix(d, d, lambda i, j: sph_inner(B[i], B[j]))
    # orthogonality check (off-diagonal zero)
    for i in range(d):
        for j in range(d):
            if i != j: assert Gram[i, j] == 0, (irr, i, j, Gram[i, j])
    norms2 = [Gram[i, i] for i in range(d)]
    # orthonormalise: b_i / sqrt(norm2_i)
    Bn = [B[i]/sp.sqrt(norms2[i]) for i in range(d)]
    print(f"[{irr}] squared norms of basis functions (units of 4pi):", [str(to_phi(n)) for n in norms2])
    # representation matrices D(g)_{ji} = <b_j, g.b_i>
    Ds = []
    for R in I_mats:
        D = Matrix(d, d, lambda j, i: sp.nsimplify(sp.radsimp(sp.simplify(sph_inner(Bn[j], act(R, Bn[i])))), [sqrt(5)]))
        Ds.append(D)
    # checks: orthogonal, character
    for R, D in zip(I_mats[:12], Ds[:12]):
        assert (D.T*D - sp.eye(d)).applyfunc(sp.simplify) == sp.zeros(d), (irr, "not orthogonal")
    for R, D in zip(I_mats, Ds):
        assert sp.simplify(D.trace() - chi_of(R, irr)) == 0, (irr, "character mismatch")
    # homomorphism spot-check
    for a in range(0, 60, 17):
        for b in range(0, 60, 23):
            Rab = (I_mats[a]*I_mats[b]).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
            idx = [k for k, R in enumerate(I_mats) if (R - Rab).applyfunc(sp.simplify) == sp.zeros(3)][0]
            assert (Ds[a]*Ds[b] - Ds[idx]).applyfunc(sp.simplify) == sp.zeros(d), "not a homomorphism"
    print(f"[ok] {irr}: explicit orthogonal irrep matrices, character and homomorphism verified")
    # action on symmetric tensors S = eta eta^T (vectorised d x d), isotypic projectors
    dd = d*d
    Kron = [sp.kronecker_product(D, D) for D in Ds]
    dec = {}
    for lam in irreps:
        dl = chi[lam][0]
        P = sp.zeros(dd)
        for R, K in zip(I_mats, Kron):
            P += chi_of(R, lam)*K
        P = (P*Rational(dl, 60)).applyfunc(lambda e: sp.nsimplify(sp.simplify(e), [sqrt(5)]))
        # restrict to symmetric tensors: rank of P on Sym^2 = dl * multiplicity
        Ssym = []
        for i in range(d):
            for j in range(i, d):
                v = sp.zeros(dd, 1); v[i*d+j] += 1; v[j*d+i] += 1
                Ssym.append(v)
        img = Matrix.hstack(*[P*v for v in Ssym])
        rk = img.rank(simplify=True)
        assert rk % dl == 0
        m = rk // dl
        if m: dec[lam] = (m, P)
    print(f"      Sym^2({irr}) = " + " + ".join(f"{(str(m)+' ') if m>1 else ''}{lam}" for lam, (m, P) in dec.items()))
    results[irr] = (d, Bn, Ds, dec, norms2)

import pickle
pickle.dump({k: (v[0], v[1], v[2], {l: (m, P) for l, (m, P) in v[3].items()}, v[4]) for k, v in results.items()}, open('gl_data.pkl', 'wb'))
print("[written] gl_data.pkl")
