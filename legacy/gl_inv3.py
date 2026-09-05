import sympy as sp, pickle, itertools, random
from sympy import sqrt, Rational, Matrix, symbols, expand, Poly, S, pi
exec(open('gl_inv2.py').read().split("data = pickle.load")[0])
data = pickle.load(open('gl_data.pkl','rb')); inv2 = pickle.load(open('gl_inv2.pkl','rb'))

def lin_relation(target, basis_polys, gens):
    """express target as linear combination of basis_polys (exact); returns coefficients or None"""
    cs = symbols(f'c0:{len(basis_polys)}')
    diff = expand(target - sum(c*b for c, b in zip(cs, basis_polys)))
    eqs = Poly(diff, *gens).coeffs()
    sol = sp.solve(eqs, cs, dict=True)
    return sol[0] if sol else None

# ---------- T1, T2, G: relations among N_l and the basic invariants
for irr in ['T1', 'T2', 'G']:
    eta, etab, comps, invs, quartic = inv2[irr]
    gens = list(eta)+list(etab)
    S1 = expand(sum(e*eb for e, eb in zip(eta, etab))**2)          # (eta.etab)^2
    S2 = expand(sum(e*e for e in eta)*sum(eb*eb for eb in etab))    # |eta.eta|^2
    print(f"\n[{irr}]")
    print("   N0 = ", lin_relation(invs['N0'], [S1, S2], gens))
    if irr in ('T1', 'T2'):
        print("   N2 = ", lin_relation(invs['N2'], [S1, S2], gens))
        print("   int|Delta|^4 = ", lin_relation(quartic, [S1, S2], gens), "  (basis: (eta.eta*)^2, |eta.eta|^2)")
    else:
        # third invariant: take N2; express N4, N6, quartic in basis (S1, S2, N2)
        for k in ['N4', 'N6']:
            print(f"   {k} = ", lin_relation(invs[k], [S1, S2, invs['N2']], gens), "  (basis: (eta.eta*)^2, |eta.eta|^2, N2)")
        print("   int|Delta|^4 = ", lin_relation(quartic, [S1, S2, invs['N2']], gens))

# ---------- H: anisotropic invariants
eta, etab, comps, invs, quartic = inv2['H']
gens = list(eta)+list(etab)
S1 = expand(sum(e*eb for e, eb in zip(eta, etab))**2); S2 = expand(sum(e*e for e in eta)*sum(eb*eb for eb in etab))
print("\n[H]")
print("   N0 = ", lin_relation(invs['N0'], [S1, S2], gens))
print("   N2 + N4 = ", lin_relation(expand(invs['N2']+invs['N4']), [S1, S2], gens))
# l=4 icosahedral projectors on monomial coordinates
mons4 = monomials(4)
def rep_matrix(R, l, mons):
    Rt = R.T
    sub = {x: Rt[0,0]*x+Rt[0,1]*y+Rt[0,2]*z, y: Rt[1,0]*x+Rt[1,1]*y+Rt[1,2]*z, z: Rt[2,0]*x+Rt[2,1]*y+Rt[2,2]*z}
    M = Matrix.hstack(*[coeff_vector(m.subs(sub, simultaneous=True), mons) for m in mons])
    return M.applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
reps4 = [rep_matrix(R, 4, mons4) for R in I_mats]
def proj(l_irr, reps, mons):
    dl = chi[l_irr][0]; P = sp.zeros(len(mons))
    for R, M in zip(I_mats, reps): P += chi_of(R, l_irr)*M
    return (P*Rational(dl, 60)).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
PG4, PH4 = proj('G', reps4, mons4), proj('H', reps4, mons4)
h4 = comps[4]
v4 = coeff_vector(h4, mons4)
h4G = expand(sum(c*m for c, m in zip(PG4*v4, mons4))); h4H = expand(sum(c*m for c, m in zip(PH4*v4, mons4)))
assert expand(h4G + h4H - h4) == 0
pairs = list(zip(eta, etab)) + list(zip(etab, eta))
def clean(expr):
    P = Poly(expand(expr), *gens)
    return expand(sum(sp.nsimplify(sp.radsimp(sp.expand(c)), [sqrt(5)])*sp.prod([g**e for g, e in zip(gens, m)]) for m, c in zip(P.monoms(), P.coeffs())))
N4G = clean(sph_norm2_complex(h4G, pairs)); N4H = clean(sph_norm2_complex(h4H, pairs))
dd_ = clean(N4G + N4H - invs['N4'])
if dd_ != 0:
    print("   residual N4G+N4H-N4 numeric max coeff:", max(abs(complex(c)) for c in Poly(dd_, *gens).coeffs()))
assert dd_ == 0 or max(abs(complex(c)) for c in Poly(dd_, *gens).coeffs()) < 1e-12
print("   N4 = N4G + N4H (icosahedral split of the l=4 part): verified")
# intertwiner J: (l=4, H) -> (l=2, H)
# orthonormal basis of the l=4 H-isotypic component: take P_H images of harmonic seeds, Gram-Schmidt on the sphere
seeds = [x**4, x**2*y*z, x**3*y, y**4, y**2*z*x, y**3*z, z**4, z**2*x*y, z**3*x, x*y**3]
def harm4(p):
    return harm_components(expand(p))[4]
cand = []
for s in seeds:
    v = coeff_vector(harm4(s), mons4)
    cand.append(expand(sum(c*m for c, m in zip(PH4*v, mons4))))
# Gram-Schmidt (exact) to get 5 orthonormal functions
basis4 = []
for c in cand:
    v = c
    for b in basis4:
        v = expand(v - sph_inner(b, c)*b)
    n2 = sph_inner(v, v)
    if n2 != 0:
        basis4.append(expand(v/sp.sqrt(n2)))
    if len(basis4) == 5: break
assert len(basis4) == 5
d, Bn2, Ds2, dec, norms2 = data['H']
D4 = [Matrix(5, 5, lambda j, i: sp.nsimplify(sp.radsimp(sp.simplify(sph_inner(basis4[j], act(R, basis4[i])))), [sqrt(5)])) for R in I_mats]
for R, D in zip(I_mats, D4):
    assert sp.simplify(D.trace() - chi_of(R, 'H')) == 0
random.seed(1)
X = Matrix(5, 5, lambda i, j: Rational(random.randint(-3, 3)))
J = sp.zeros(5)
for D2, Dd4 in zip(Ds2, D4):
    J += D2*X*Dd4.T
J = (J/60).applyfunc(lambda e: sp.nsimplify(sp.radsimp(sp.simplify(e)), [sqrt(5)]))
JtJ = (J.T*J).applyfunc(lambda e: sp.nsimplify(sp.radsimp(sp.simplify(e)), [sqrt(5)]))
lam = JtJ[0, 0]
assert (JtJ - lam*sp.eye(5)).applyfunc(sp.simplify) == sp.zeros(5), "intertwiner not proportional to orthogonal"
J = (J/sp.sqrt(lam)).applyfunc(lambda e: sp.nsimplify(sp.radsimp(sp.simplify(e)), [sqrt(5)]))
for D2, Dd4 in zip(Ds2[:10], D4[:10]):
    assert (D2*J - J*Dd4).applyfunc(sp.simplify) == sp.zeros(5)
print("   intertwiner J: (l=4,H) -> (l=2,H) built, orthogonal, equivariance verified")
# coordinates of h2 and h4H in the orthonormal bases
h2 = comps[2]
c2 = [sp.expand(sph_inner_poly(h2, b)) if False else None for b in Bn2]
def coords(h, basis):
    # h has complex coefficients (polynomial in etas): inner product linear in h
    out = []
    for b in basis:
        P = Poly(expand(h*b), x, y, z)
        out.append(expand(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi)))
    return Matrix(out)
c2 = coords(h2, Bn2); c4 = coords(h4H, basis4)
Jc4 = J*c4
def conj_eta(e): return e.subs({a: b for a, b in pairs}, simultaneous=True)
C = clean(sum(conj_eta(c2[i])*Jc4[i] for i in range(5)))
# split into real and imaginary parts: C = CR + i CI where conjugation swaps eta<->etab
Cbar = expand(C.subs({a: b for a, b in pairs}, simultaneous=True))
CR = clean((C + Cbar)/2); CI_i = clean((C - Cbar)/2)   # CI_i = i*CI
print("   cross invariants C_R, C_I built; terms:", len(Poly(CR, *gens).terms()), len(Poly(CI_i, *gens).terms()))
# independence of the six
six = [S1, S2, invs['N2'], N4G, CR, sp.I*CI_i]
cs = symbols('k0:6')
diff = expand(sum(c*b for c, b in zip(cs, six)))
sol = sp.solve(Poly(diff, *gens).coeffs(), cs, dict=True)
print("   linear independence of the six quartic invariants:", sol == [{c: 0 for c in cs}] or sol)
print("   int|Delta|^4 = ", lin_relation(quartic, [S1, S2, invs['N2'], N4G, CR, sp.I*CI_i], gens))
pickle.dump({'S1': S1, 'S2': S2, 'N2': invs['N2'], 'N4': invs['N4'], 'N4G': N4G, 'N4H': N4H, 'CR': CR, 'CI_i': CI_i, 'eta': eta, 'etab': etab,
             'basis2': Bn2, 'basis4H': basis4, 'J': J}, open('gl_H.pkl', 'wb'))
