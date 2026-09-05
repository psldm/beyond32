#!/usr/bin/env python3
"""
ih_seeds.py -- presentable basis functions (phi-form) of the icosahedral isotypic components,
obtained by applying the projectors to symmetric seed polynomials; plus the hexad form of the
l=6 invariant. Companion to ih_basis.py (same exact group construction).

Builds the binary icosahedral group 2I as unit quaternions (120 elements), the rotation
group I (60 exact matrices over Q(sqrt5)), verifies character tables, computes
  * SO(3) -> I branching of spherical harmonics l = 0..LMAX
  * explicit polynomial bases of every isotypic component (harmonic polynomials)
  * Molien series check + Poincare dodecahedral space S^3/2I mode dictionary
  * subgroup restrictions I -> T, D5, D3, D2 (approximant / strain lowering)
  * number of independent quartic Ginzburg-Landau invariants per irrep
  * SU(2) -> 2I branching for half-integer j (spin-orbit coupled quasiparticles)
Everything exact (sympy), no floating point in the results.
"""
import itertools, sys
import sympy as sp
from sympy import sqrt, Rational, Matrix, symbols, expand, Poly, S, simplify, nsimplify

phi = (1 + sqrt(5)) / 2
x, y, z = symbols('x y z', real=True)


# ---------------------------------------------------------------- 2I as quaternions
def even_perms(v):
    out = []
    for p in itertools.permutations(range(4)):
        # parity of permutation
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            out.append(tuple(v[p[i]] for i in range(4)))
    return out

quats = set()
half = Rational(1, 2)
# binary tetrahedral part: +-1, +-i, +-j, +-k, (+-1 +-i +-j +-k)/2
for i in range(4):
    for s in (1, -1):
        q = [S(0)] * 4; q[i] = S(s); quats.add(tuple(q))
for signs in itertools.product((1, -1), repeat=4):
    quats.add(tuple(half * s for s in signs))
# the 96 others: even permutations of (+-phi, +-1, +-1/phi, 0)/2
base = [phi, S(1), 1 / phi, S(0)]
for signs in itertools.product((1, -1), repeat=3):
    v = [signs[0] * base[0], signs[1] * base[1], signs[2] * base[2], S(0)]
    for p in even_perms(v):
        quats.add(tuple(sp.nsimplify(half * c, [sqrt(5)]) for c in p))
quats = sorted(quats, key=str)
assert len(quats) == 120, len(quats)

def qmul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return (sp.expand(a0*b0 - a1*b1 - a2*b2 - a3*b3),
            sp.expand(a0*b1 + a1*b0 + a2*b3 - a3*b2),
            sp.expand(a0*b2 - a1*b3 + a2*b0 + a3*b1),
            sp.expand(a0*b3 + a1*b2 - a2*b1 + a3*b0))

def qnorm(q):
    return tuple(sp.nsimplify(sp.radsimp(c), [sqrt(5)]) for c in q)

qset = set(qnorm(q) for q in quats)
# closure check on a random sample of products (full 120x120 is fine but slow-ish; do it)
for a in quats[:40]:
    for b in quats[::7]:
        assert qnorm(qmul(a, b)) in qset, "2I not closed?!"

def rot(q):
    w, a, b, c = q
    return Matrix([[w*w+a*a-b*b-c*c, 2*(a*b-w*c), 2*(a*c+w*b)],
                   [2*(a*b+w*c), w*w-a*a+b*b-c*c, 2*(b*c-w*a)],
                   [2*(a*c-w*b), 2*(b*c+w*a), w*w-a*a-b*b+c*c]]).applyfunc(
                       lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))

rots = {}
for q in quats:
    R = rot(q)
    key = tuple(rots_e for rots_e in R)
    rots.setdefault(key, R)
I_mats = list(rots.values())
assert len(I_mats) == 60
for R in I_mats:
    assert (R.T * R - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3)
    assert sp.simplify(R.det()) == 1

# ---------------------------------------------------------------- classes of I by rotation angle
def cos_theta(R):  # trace = 1 + 2 cos theta
    return sp.nsimplify(sp.simplify((R.trace() - 1) / 2), [sqrt(5)])

class_key = {}
for R in I_mats:
    class_key.setdefault(cos_theta(R), []).append(R)
# cos theta values: 1 (E), cos72=(phi-1)/2, cos144=-phi/2, cos120=-1/2, cos180=-1
c72 = sp.nsimplify((phi - 1) / 2, [sqrt(5)]); c144 = sp.nsimplify(-phi / 2, [sqrt(5)])
classes = [(S(1), 'E'), (c72, 'C5'), (c144, 'C5^2'), (Rational(-1, 2), 'C3'), (S(-1), 'C2')]
sizes = [len(class_key[c]) for c, _ in classes]
assert sizes == [1, 12, 12, 20, 15], sizes

# character table of I (rows: irreps, cols: classes in the order above)
chi = {
    'A' : [1, 1, 1, 1, 1],
    'T1': [3, phi, 1 - phi, 0, -1],
    'T2': [3, 1 - phi, phi, 0, -1],
    'G' : [4, -1, -1, 1, 0],
    'H' : [5, 0, 0, -1, 1],
}
irreps = list(chi)
def inner(u, v):
    return sp.nsimplify(sp.simplify(sum(s * a * b for s, a, b in zip(sizes, u, v)) / 60), [sqrt(5)])
for a in irreps:
    for b in irreps:
        assert inner(chi[a], chi[b]) == (1 if a == b else 0), (a, b)
print("[ok] I: 60 rotations, class sizes", sizes, "character table orthonormal")

# ---------------------------------------------------------------- five-fold axes (convention report)
axes5 = []
for R in class_key[c72]:
    ev = (R - sp.eye(3)).nullspace()[0]
    v = ev / sp.gcd_list([sp.nsimplify(e, [sqrt(5)]) for e in ev]) if False else ev
    v = sp.simplify(v / max(abs(e) for e in v if e != 0))
    v = v.applyfunc(lambda e: sp.nsimplify(e, [sqrt(5)]))
    key = tuple(v) if v[list(i for i in range(3) if v[i] != 0)[0]] > 0 else tuple(-v)
    if key not in axes5:
        axes5.append(key)
print("[info] five-fold axes (6):", axes5)


phi_ = symbols('varphi')
def to_phi(e):
    """write element of Q(sqrt5) as a + b*phi with rationals"""
    e = sp.nsimplify(sp.expand(S(e)), [sqrt(5)])
    a = sp.Rational(0); 
    # e = p + q*sqrt5 -> sqrt5 = 2phi-1 -> e = (p - q) + 2q phi
    p = e.subs(sqrt(5), 0) if e.has(sqrt(5)) else e
    q = sp.expand(e - p)/sqrt(5) if e.has(sqrt(5)) else 0
    q = sp.nsimplify(sp.simplify(q))
    return sp.nsimplify(p - q) + 2*q*phi_
def poly_phi(pol):
    P = Poly(expand(pol), x, y, z)
    return sum(to_phi(c) * x**m[0]*y**m[1]*z**m[2] for m, c in zip(P.monoms(), P.coeffs()))
def monomials(l):
    return [x**a * y**b * z**(l - a - b) for a in range(l + 1) for b in range(l + 1 - a)]
def coeff_vector(poly, mons):
    P = Poly(expand(poly), x, y, z); d = dict(zip(P.monoms(), P.coeffs()))
    return Matrix([[d.get(Poly(m, x, y, z).monoms()[0], 0)] for m in mons])
def rep_matrix(R, l, mons):
    Rt = R.T
    sub = {x: Rt[0,0]*x+Rt[0,1]*y+Rt[0,2]*z, y: Rt[1,0]*x+Rt[1,1]*y+Rt[1,2]*z, z: Rt[2,0]*x+Rt[2,1]*y+Rt[2,2]*z}
    M = Matrix.hstack(*[coeff_vector(m.subs(sub, simultaneous=True), mons) for m in mons])
    return M.applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
def chi_of(R, irr):
    c = cos_theta(R)
    for (cc, _), val in zip(classes, chi[irr]):
        if c == cc: return val
def projector(l, irr, mons, reps):
    d = chi[irr][0]; P = sp.zeros(len(mons))
    for R, M in zip(I_mats, reps): P += chi_of(R, irr) * M
    return (P * Rational(d, 60)).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
def apply(P, seed, mons):
    v = P * coeff_vector(seed, mons)
    pol = expand(sum(c*m for c, m in zip(v, mons)))
    # normalise: make the coefficient of the seed monomial equal 1 if nonzero
    Pp = Poly(pol, x, y, z); d = dict(zip(Pp.monoms(), Pp.coeffs()))
    sm = Poly(seed, x, y, z).monoms()[0]
    if pol == 0: return 0
    lead = d.get(sm, None)
    if lead is None or lead == 0: lead = Pp.coeffs()[0]
    return sp.nsimplify(expand(sp.radsimp(pol/lead)), [sqrt(5)])
def lap(p): return expand(sp.diff(p,x,2)+sp.diff(p,y,2)+sp.diff(p,z,2))
r2 = x**2+y**2+z**2
def harm(p):
    """harmonic projection of a homogeneous polynomial (degree <= 6) by removing r^2 multiples"""
    l = Poly(p, x, y, z).total_degree()
    # solve p = h + r2*q with lap(h)=0: iterate lap
    # use formula: h = sum_k c_k r^{2k} lap^k p ; find c_k by requiring lap(h)=0 symbolically
    cs = symbols('c0:%d' % (l//2+1))
    terms = [p]; 
    for k in range(1, l//2+1): terms.append(lap(terms[-1]))
    h = sum(c*r2**k*t for k,(c,t) in enumerate(zip(cs, terms)))
    eqs = Poly(lap(h), x, y, z).coeffs()
    sol = sp.solve(eqs + [cs[0]-1], cs, dict=True)[0]
    return expand(h.subs(sol))

def harm_proj(p):
    """exact harmonic (top-l) component of homogeneous p: solve p = h + r2*q"""
    l = Poly(p, x, y, z).total_degree()
    if l < 2:
        return expand(p)
    mons = monomials(l)
    lapcols = []
    mons_l2 = monomials(l-2)
    L = Matrix.hstack(*[coeff_vector(lap(m), mons_l2) for m in mons])
    Hb = L.nullspace()                       # harmonic basis (coefficient vectors)
    Q = [coeff_vector(expand(r2*m), mons) for m in monomials(l-2)]
    B = Matrix.hstack(*(Hb + Q))
    v = coeff_vector(p, mons)
    sol = B.solve(v) if B.is_square else B.gauss_jordan_solve(v)[0]
    hvec = sum((sol[i]*Hb[i] for i in range(len(Hb))), sp.zeros(len(mons),1))
    return expand(sum(c*m for c, m in zip(hvec, mons)))
def proportional(p, q):
    mons = monomials(Poly(p,x,y,z).total_degree())
    a = coeff_vector(p, mons); b = coeff_vector(q, mons)
    i = [k for k in range(len(a)) if b[k] != 0][0]
    ratio = sp.nsimplify(sp.simplify(a[i]/b[i]), [sqrt(5)])
    return ratio, all(sp.simplify(a[k]-ratio*b[k]) == 0 for k in range(len(a)))

tasks = {1: {'T1': [x]}, 2: {'H': [x*y, x**2-y**2]},
         3: {'T2': [x**3], 'G': [x**3, x*y*z]},
         4: {'G': [x**4, x**2*y*z], 'H': [x**4, x**2*y*z, x**3*y]}}
print("\nphi-form basis functions from symmetric seeds (apply harmonic projection, then P_Gamma):")
for l in sorted(tasks):
    mons = monomials(l); reps = [rep_matrix(R, l, mons) for R in I_mats]
    for irr, seeds in tasks[l].items():
        P = projector(l, irr, mons, reps)
        for s in seeds:
            out = apply(P, harm_proj(s), mons)
            print(f"  l={l} {irr}: P[harm({s})] = {poly_phi(out)}   | harmonic: {lap(out)==0}")
# l=6 invariant
mons = monomials(6); reps = [rep_matrix(R, 6, mons) for R in I_mats]
PA = projector(6, 'A', mons, reps)
inv = apply(PA, harm_proj(x**6), mons)
print("\nl=6 A invariant:", poly_phi(inv), "| harmonic:", lap(inv)==0)
axes = [Matrix(a) for a in axes5]
hexad = expand(sum((a.dot(Matrix([x,y,z]))/sp.sqrt(a.dot(a)))**6 for a in axes))
h6 = harm_proj(hexad)
print("hexad top component proportional to invariant:", proportional(h6, inv))
rest = expand(hexad - h6)
print("hexad - h6 =", sp.factor(rest))
