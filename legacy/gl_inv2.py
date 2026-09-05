"""Explicit quartic invariants via the gap function Delta(n) = sum eta_i bhat_i(n) and the harmonic
decomposition of Delta^2. Exact. Produces polynomials in eta, conj(eta)."""
import sympy as sp, pickle, itertools
from sympy import sqrt, Rational, Matrix, symbols, expand, Poly, S, pi, factorial2, conjugate, I as Im
exec(open('gl_inv.py').read().split("results = {}")[0])   # reuse group, bases, sph_inner, act, chi_of

def harm_components(F):
    """decompose homogeneous F (degree n) as sum_l h_l r^(n-l): returns dict l -> h_l (harmonic)"""
    n = Poly(F, x, y, z).total_degree()
    r2 = x**2+y**2+z**2
    comps = {}
    rest = expand(F)
    for l in range(n, -1, -2):
        # top harmonic component of rest (degree l): solve rest = h + r2*q
        mons = monomials(l)
        if l >= 2:
            mons_l2 = monomials(l-2)
            L = Matrix.hstack(*[coeff_vector(lap(m), mons_l2) for m in mons])
            Hb = L.nullspace()
            Q = [coeff_vector(expand(r2*m), mons) for m in mons_l2]
            B = Matrix.hstack(*(Hb+Q))
            v = coeff_vector(rest, mons)
            sol = B.solve(v)
            h = expand(sum(sol[i]*sum(c*m for c, m in zip(Hb[i], mons)) for i in range(len(Hb))))
            q = expand(sum(sol[len(Hb)+i]*m for i, m in enumerate(mons_l2)))
        else:
            h = rest; q = S(0)
        comps[l] = h
        rest = q
        if rest == 0: break
    return comps
def monomials(l): return [x**a*y**b*z**(l-a-b) for a in range(l+1) for b in range(l+1-a)]
def coeff_vector(poly, mons):
    P = Poly(expand(poly), x, y, z); d = dict(zip(P.monoms(), P.coeffs()))
    return Matrix([[d.get(Poly(m, x, y, z).monoms()[0], 0)] for m in mons])
def lap(p): return expand(sp.diff(p,x,2)+sp.diff(p,y,2)+sp.diff(p,z,2))
def sph_norm2_complex(h, etas):
    """int |h|^2 dOmega/4pi for h with complex coefficients (polynomial in etas): treat eta_i, etab_i independent"""
    hb = h.subs({e: eb for e, eb in etas}, simultaneous=True)
    P = Poly(expand(h*hb), x, y, z)
    val = expand(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi))
    # clean coefficients (elements of Q(sqrt5)) monomial by monomial in the etas
    Pe = Poly(val, *[v for pr in etas[:len(etas)//2] for v in pr])
    return expand(sum(sp.nsimplify(sp.radsimp(sp.expand(c)), [sqrt(5)])*sp.prod([g**e for g, e in zip(Pe.gens, m)]) for m, c in zip(Pe.monoms(), Pe.coeffs())))

data = pickle.load(open('gl_data.pkl','rb'))
out = {}
for irr in ['T1', 'T2', 'G', 'H']:
    d, Bn, Ds, dec, norms2 = data[irr]
    eta = symbols(f'eta1:{d+1}'); etab = symbols(f'etab1:{d+1}')
    pairs = list(zip(eta, etab)) + list(zip(etab, eta))
    Delta = sum(e*b for e, b in zip(eta, Bn))
    # |eta|^2 normalisation: int |Delta|^2 = sum |eta_i|^2 (orthonormal basis)
    n2 = expand(sph_norm2_complex(Delta, pairs)); assert n2 == sum(e*eb for e, eb in zip(eta, etab))
    F = expand(Delta**2)
    comps = harm_components(F)
    invs = {}
    for l, h in comps.items():
        invs[f'N{l}'] = expand(sph_norm2_complex(h, pairs))
    tot = expand(sum(invs.values()))
    quartic = expand(sph_norm2_complex(Delta, pairs)*0 + sp.expand(sph_norm2_complex(F, pairs)))  # int |Delta|^4
    assert expand(tot - quartic) == 0
    out[irr] = (eta, etab, comps, invs, quartic)
    print(f"\n[{irr}] harmonic content of Delta^2: l = {sorted(comps)}")
    for k, v in invs.items():
        print(f"   {k}: {len(Poly(v, *eta, *etab).terms())} terms")
pickle.dump({k: v for k, v in out.items()}, open('gl_inv2.pkl','wb'))
