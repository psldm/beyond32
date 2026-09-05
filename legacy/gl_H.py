import sympy as sp, pickle, random, time
from sympy import sqrt, Rational, Matrix, symbols, expand, Poly, S, pi
t0=time.time()
exec(open('gl_inv2.py').read().split("data = pickle.load")[0])
inv2 = pickle.load(open('gl_inv2.pkl','rb'))
eta, etab, comps, invs, quartic = inv2['H']
gens = list(eta)+list(etab); pairs = list(zip(eta, etab)) + list(zip(etab, eta))
def clean(expr):
    P = Poly(expand(expr), *gens)
    return expand(sum(sp.nsimplify(sp.radsimp(sp.expand(c)), [sqrt(5)])*sp.prod([g**e for g, e in zip(gens, m)]) for m, c in zip(P.monoms(), P.coeffs())))
S1 = expand(sum(e*eb for e, eb in zip(eta, etab))**2); S2 = expand(sum(e*e for e in eta)*sum(eb*eb for eb in etab))
mons4, mons2 = monomials(4), monomials(2)
def rep_matrix(R, l, mons):
    Rt = R.T
    sub = {x: Rt[0,0]*x+Rt[0,1]*y+Rt[0,2]*z, y: Rt[1,0]*x+Rt[1,1]*y+Rt[1,2]*z, z: Rt[2,0]*x+Rt[2,1]*y+Rt[2,2]*z}
    M = Matrix.hstack(*[coeff_vector(m.subs(sub, simultaneous=True), mons) for m in mons])
    return M.applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
reps4 = [rep_matrix(R, 4, mons4) for R in I_mats]
reps2 = [rep_matrix(R, 2, mons2) for R in I_mats]
reps4inv = [rep_matrix(R.T, 4, mons4) for R in I_mats]
def proj(l_irr, reps, mons):
    dl = chi[l_irr][0]; P = sp.zeros(len(mons))
    for R, M in zip(I_mats, reps): P += chi_of(R, l_irr)*M
    return (P*Rational(dl, 60)).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
PG4, PH4 = proj('G', reps4, mons4), proj('H', reps4, mons4)
h4 = comps[4]; v4 = coeff_vector(h4, mons4)
h4G = expand(sum(c*m for c, m in zip(PG4*v4, mons4))); h4H = expand(sum(c*m for c, m in zip(PH4*v4, mons4)))
N4G = clean(sph_norm2_complex(h4G, pairs)); N4H = clean(sph_norm2_complex(h4H, pairs))
res = clean(N4G + N4H - invs['N4']); assert res == 0, res
print(f"[ok] N4 = N4G + N4H  ({time.time()-t0:.0f}s)")
# seed map L: degree-4 poly -> harmonic degree-2 poly, L(h) = harm2(d^2 h/dx^2)
r2 = x**2+y**2+z**2
def L_apply(h):
    p = expand(sp.diff(h, x, 2)); c = lap(p)/6
    return expand(p - c*r2)
random.seed(3)
Lmat = Matrix(6, 15, lambda i, j: Rational(random.randint(-4, 4), random.randint(1, 3)))   # generic seed 6 x 15
J = sp.zeros(6, 15)
for R2, R4i in zip(reps2, reps4inv):
    J += R2*Lmat*R4i
J = (J/60).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
# equivariance check: J rho4(g) = rho2(g) J
for k in range(0, 60, 11):
    assert ((J*reps4[k] - reps2[k]*J).applyfunc(sp.simplify)) == sp.zeros(6, 15)
Jh4H = expand(sum(c*m for c, m in zip(J*coeff_vector(h4H, mons4), mons2)))
assert Jh4H != 0
assert lap(Jh4H.subs({e: 1 for e in eta}).subs({e: 1 for e in etab})) == 0  # lands in harmonic l=2
# normalise J: isometry on the H_4 subspace -> compare norms for a numeric eta
num = {e: Rational(random.randint(1,5), 7) + sp.I*Rational(random.randint(1,5), 3) for e in eta}
numb = {eb: sp.conjugate(num[e]) for e, eb in zip(eta, etab)}
def norm2_num(h):
    hn = expand(h.subs(num).subs(numb)); hb = expand(sp.conjugate(hn))
    P = Poly(expand(hn*hb), x, y, z)
    return sp.nsimplify(sp.simplify(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi)), [sqrt(5)])
lam = sp.nsimplify(sp.simplify(norm2_num(Jh4H)/norm2_num(h4H)), [sqrt(5)])
print("   |J h|^2/|h|^2 =", lam, "=", float(lam))
# check it is the same for a second random eta (isometry up to scalar)
num2 = {e: Rational(random.randint(1,5), 11) - sp.I*Rational(random.randint(1,5), 5) for e in eta}
numb2 = {eb: sp.conjugate(num2[e]) for e, eb in zip(eta, etab)}
def norm2_num2(h):
    hn = expand(h.subs(num2).subs(numb2)); hb = expand(sp.conjugate(hn))
    P = Poly(expand(hn*hb), x, y, z)
    return sp.nsimplify(sp.simplify(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi)), [sqrt(5)])
lam2 = sp.nsimplify(sp.simplify(norm2_num2(Jh4H)/norm2_num2(h4H)), [sqrt(5)])
assert sp.simplify(lam - lam2) == 0, (lam, lam2)
print("[ok] J is an isometry up to the scalar above (Schur)")
h2 = comps[2]
# cross invariant C = <h2, J h4H>_sphere / sqrt(lam)
h2b = h2.subs({a: b for a, b in pairs}, simultaneous=True)
P = Poly(expand(h2b*Jh4H), x, y, z)
C = clean(sum(c*mono_int(*m) for m, c in zip(P.monoms(), P.coeffs()))/(4*pi))
Cbar = clean(C.subs({a: b for a, b in pairs}, simultaneous=True))
CR = clean((C + Cbar)/2); CIi = clean((C - Cbar)/2)     # CIi = i * C_I
print("   C_R terms:", len(Poly(CR, *gens).terms()), " C_I terms:", len(Poly(CIi, *gens).terms()))
six = [S1, S2, invs['N2'], N4G, CR, CIi]
cs = symbols('k0:6')
diff = expand(sum(c*b for c, b in zip(cs, six)))
sol = sp.solve(Poly(diff, *gens).coeffs(), cs, dict=True)
print("   six invariants linearly independent:", sol == [{c: 0 for c in cs}])
pickle.dump({'S1': S1, 'S2': S2, 'N2': invs['N2'], 'N4': invs['N4'], 'N4G': N4G, 'N4H': N4H, 'CR': CR, 'CIi': CIi, 'lam': lam,
             'eta': eta, 'etab': etab, 'J': J, 'h2': h2, 'h4G': h4G, 'h4H': h4H}, open('gl_H.pkl', 'wb'))
print(f"[written] gl_H.pkl ({time.time()-t0:.0f}s)")
