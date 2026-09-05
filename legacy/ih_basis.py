#!/usr/bin/env python3
"""
ih_basis.py -- exact icosahedral group theory for superconducting pairing channels.

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
LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6

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

# ---------------------------------------------------------------- action on harmonic polynomials
def monomials(l):
    return [x**a * y**b * z**(l - a - b) for a in range(l + 1) for b in range(l + 1 - a)]

def coeff_vector(poly, mons):
    P = Poly(expand(poly), x, y, z)
    d = dict(zip(P.monoms(), P.coeffs()))
    return Matrix([[d.get(Poly(m, x, y, z).monoms()[0], 0)] for m in mons])

def rep_matrix(R, l, mons):
    # (rho(R) f)(r) = f(R^{-1} r); R^{-1} = R^T
    Rt = R.T
    sub = {x: Rt[0, 0]*x + Rt[0, 1]*y + Rt[0, 2]*z,
           y: Rt[1, 0]*x + Rt[1, 1]*y + Rt[1, 2]*z,
           z: Rt[2, 0]*x + Rt[2, 1]*y + Rt[2, 2]*z}
    cols = [coeff_vector(m.subs(sub, simultaneous=True), mons) for m in mons]
    M = Matrix.hstack(*cols)
    return M.applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))

def harmonic_basis(l, mons):
    # kernel of Laplacian on homogeneous degree-l polynomials, as coefficient vectors
    lap_cols = []
    mons_lm2 = monomials(l - 2) if l >= 2 else []
    for m in mons:
        lm = sp.diff(m, x, 2) + sp.diff(m, y, 2) + sp.diff(m, z, 2)
        lap_cols.append(coeff_vector(lm, mons_lm2) if l >= 2 else Matrix([[0]]))
    L = Matrix.hstack(*lap_cols) if l >= 2 else sp.zeros(1, len(mons))
    return L.nullspace()

def chi_of(R, irr):
    c = cos_theta(R)
    for (cc, _), val in zip(classes, chi[irr]):
        if c == cc:
            return val
    raise ValueError

def vec_to_poly(v, mons):
    return expand(sum(c * m for c, m in zip(v, mons)))

def nice(p):
    p = sp.nsimplify(expand(p), [sqrt(5)])
    # scale to make leading coefficient 1 and clear denominators
    P = Poly(p, x, y, z)
    coeffs = [sp.nsimplify(c, [sqrt(5)]) for c in P.coeffs()]
    # try to make integer-ish: divide by gcd of rational parts if all rational
    if all(c.is_Rational for c in coeffs):
        g = sp.gcd_list(coeffs)
        p = expand(p / g)
    else:
        lc = coeffs[0]
        p = expand(sp.radsimp(p / lc))
        p = sp.nsimplify(p, [sqrt(5)])
    return p

branching = {}
bases = {}
mult_A = {}
print()
for l in range(0, LMAX + 1):
    mons = monomials(l)
    H = harmonic_basis(l, mons)            # list of coefficient vectors
    Hmat = Matrix.hstack(*H)               # (n_mons) x (2l+1)
    assert Hmat.shape[1] == 2 * l + 1
    reps = [rep_matrix(R, l, mons) for R in I_mats]
    # projectors onto harmonic subspace coordinates: work in monomial coords, then restrict
    line = []
    for irr in irreps:
        d = chi[irr][0]
        P = sp.zeros(len(mons))
        for R, M in zip(I_mats, reps):
            P += chi_of(R, irr) * M
        P = (P * Rational(d, 60)).applyfunc(lambda e: sp.nsimplify(sp.expand(e), [sqrt(5)]))
        img = (P * Hmat)                   # images of harmonic basis
        # basis of image
        img_cols = [img[:, j] for j in range(img.shape[1])]
        rref_basis = Matrix.hstack(*img_cols).T.rref()[0]
        rows = [rref_basis[i, :].T for i in range(rref_basis.rows) if any(e != 0 for e in rref_basis[i, :])]
        m = len(rows) // d
        assert len(rows) % d == 0, (l, irr, len(rows))
        branching[(l, irr)] = m
        if m:
            bases[(l, irr)] = [nice(vec_to_poly(r, mons)) for r in rows]
            line.append(f"{m if m>1 else ''}{irr}")
        if irr == 'A':
            mult_A[l] = m
    dimcheck = sum(branching[(l, irr)] * chi[irr][0] for irr in irreps)
    assert dimcheck == 2 * l + 1
    print(f"l={l}: " + " + ".join(line))

# character-formula cross-check of branching (independent of the polynomial computation)
import math
def chi_l(l, c):  # character of SO(3) irrep l on class with cos(theta)=c
    if c == 1:
        return 2 * l + 1
    th = sp.acos(c)
    return sp.nsimplify(sp.simplify(sp.sin((l + Rational(1, 2)) * th) / sp.sin(th / 2)), [sqrt(5)])
for l in range(0, LMAX + 1):
    for irr in irreps:
        m = inner(chi[irr], [chi_l(l, c) for c, _ in classes])
        assert m == branching[(l, irr)], (l, irr, m, branching[(l, irr)])
print("[ok] branching agrees with character formula")

# ---------------------------------------------------------------- Molien / Poincare dictionary
print()
mA = [inner(chi['A'], [chi_l(l, c) for c, _ in classes]) for l in range(0, 31)]
t = symbols('t')
series = sp.series((1 + t**15) / ((1 - t**6) * (1 - t**10)), t, 0, 31).removeO()
coeffs = [series.coeff(t, l) for l in range(0, 31)]
assert [int(a) for a in mA] == [int(c) for c in coeffs], (mA, coeffs)
print("[ok] m_A(l), l=0..30:", [int(a) for a in mA])
print("     = coefficients of (1+t^15)/((1-t^6)(1-t^10))")
poinc = [(k, (k + 1) * int(mA[k // 2])) for k in range(0, 61, 2) if int(mA[k // 2]) > 0]
print("     Poincare space S^3/2I modes (k, multiplicity):", poinc[:12])

# ---------------------------------------------------------------- subgroup restrictions
print()
def restrict(sub_name, sub_classes, sub_chi):
    """sub_classes: list of (cos theta, size) for the subgroup; sub_chi: dict irr -> chars.
       Character of I-irrep on subgroup class = chi[irr] at that cos theta (rotation classes fuse)."""
    order = sum(s for _, s in sub_classes)
    def val(irr, c):
        for (cc, _), v in zip(classes, chi[irr]):
            if cc == c:
                return v
        raise ValueError(c)
    res = {}
    for irr in irreps:
        r = [val(irr, c) for c, _ in sub_classes]
        dec = {}
        for sirr, sch in sub_chi.items():
            norm = sp.nsimplify(sp.simplify(sum(s * sp.conjugate(a) * a for (c, s), a in zip(sub_classes, sch)) / order), [sqrt(5)])
            m = sp.nsimplify(sp.simplify(sum(s * sp.conjugate(a) * b for (c, s), a, b in zip(sub_classes, sch, r)) / order / norm), [sqrt(5)])
            if m != 0:
                dec[sirr] = m
        res[irr] = dec
        print(f"  {sub_name}: {irr:2s} -> " + " + ".join(f"{(str(m)+' ') if m!=1 else ''}{k}" for k, m in dec.items()))
    return res

# T (order 12): classes E(1), C3 (8, cos=-1/2), C2 (3, cos=-1); real irreps A, E(2-dim real), T
print("Restrictions (approximant / strain lowering):")
restrict('T ', [(S(1), 1), (Rational(-1, 2), 8), (S(-1), 3)],
         {'A': [1, 1, 1], 'E': [2, -1, 2], 'T': [3, 0, -1]})
# D5 (order 10): E(1), C5 (2, cos72), C5^2 (2, cos144), C2' (5, cos=-1)
restrict('D5', [(S(1), 1), (c72, 2), (c144, 2), (S(-1), 5)],
         {'A1': [1, 1, 1, 1], 'A2': [1, 1, 1, -1],
          'E1': [2, 2 * c72, 2 * c144, 0], 'E2': [2, 2 * c144, 2 * c72, 0]})
# D3 (order 6): E(1), C3 (2, cos=-1/2), C2' (3, cos=-1)
restrict('D3', [(S(1), 1), (Rational(-1, 2), 2), (S(-1), 3)],
         {'A1': [1, 1, 1], 'A2': [1, 1, -1], 'E': [2, -1, 0]})
# D2 (order 4): all three C2 fuse with same cos=-1 -> characters cannot distinguish B1,B2,B3
# (they do split, but by axis; report only the count of 1-dim pieces)
print("  D2: every irrep splits into 1-dim pieces (dim many); B1/B2/B3 assignment needs axis labels (TODO in paper)")

# ---------------------------------------------------------------- Ginzburg-Landau quartic invariants
print()
print("Quartic GL invariants for a complex order parameter in irrep Gamma:")
print("  (independent terms eta_i eta_j eta*_k eta*_l = dim Hom_I(Sym^2 Gamma, Sym^2 Gamma))")
def chi_class_sq(irr):
    """character of Sym^2(irr) on each class: (chi(g)^2 + chi(g^2))/2"""
    out = []
    for (c, name), v in zip(classes, chi[irr]):
        # g^2 lands in class with cos(2 theta) = 2c^2 - 1
        c2 = sp.nsimplify(sp.expand(2 * c * c - 1), [sqrt(5)])
        v2 = None
        for (cc, _), vv in zip(classes, chi[irr]):
            if cc == c2:
                v2 = vv
        assert v2 is not None, (irr, c, c2)
        out.append(sp.nsimplify(sp.simplify((v * v + v2) / 2), [sqrt(5)]))
    return out
for irr in ['T1', 'T2', 'G', 'H']:
    s2 = chi_class_sq(irr)
    dec = {k: inner(chi[k], s2) for k in irreps}
    dec = {k: m for k, m in dec.items() if m != 0}
    n4 = sum(m * m for m in dec.values())
    print(f"  {irr}: Sym^2 = " + " + ".join(f"{(str(m)+' ') if m!=1 else ''}{k}" for k, m in dec.items()) + f"  ->  {n4} quartic invariants")

# ---------------------------------------------------------------- 2I character table and SU(2) -> 2I
print()
# classes of 2I by quaternion real part w = cos(alpha/2), alpha = SU(2) angle in [0, 2pi]
qclasses = {}
for q in quats:
    qclasses.setdefault(sp.nsimplify(q[0], [sqrt(5)]), []).append(q)
qkeys = sorted(qclasses, key=lambda w: -float(w))
qsizes = [len(qclasses[w]) for w in qkeys]
print("[info] 2I classes by Re q = cos(alpha/2):", [(str(w), n) for w, n in zip(qkeys, qsizes)])
assert sum(qsizes) == 120
def chi_j(j, w):
    """character of SU(2) irrep spin j on the class with cos(alpha/2)=w."""
    if w == 1:
        return 2 * j + 1
    if w == -1:
        return (2 * j + 1) * (-1) ** int(2 * j)
    al2 = sp.acos(w)  # alpha/2
    return sp.nsimplify(sp.simplify(sp.sin((2 * j + 1) * al2) / sp.sin(al2)), [sqrt(5)])
def qinner(u, v):
    return sp.nsimplify(sp.simplify(sum(s * sp.conjugate(a) * b for s, a, b in zip(qsizes, u, v)) / 120), [sqrt(5)])
def galois(v):
    return [sp.nsimplify(sp.expand(S(e).subs(sqrt(5), -sqrt(5))), [sqrt(5)]) for e in v]
irr2 = {}
for j in [0, Rational(1, 2), 1, Rational(3, 2), 2, Rational(5, 2)]:
    irr2[f"{int(2*j+1)}"] = [chi_j(j, w) for w in qkeys]
# Galois conjugates give 2', 3', 4'
irr2["2'"] = galois(irr2['2']); irr2["3'"] = galois(irr2['3'])
prod = [a * b for a, b in zip(irr2['2'], irr2["2'"])]
irr2["4'"] = [sp.nsimplify(sp.expand(e), [sqrt(5)]) for e in prod]
names2 = ['1', '2', "2'", '3', "3'", '4', "4'", '5', '6']
for a in names2:
    for b in names2:
        assert qinner(irr2[a], irr2[b]) == (1 if a == b else 0), (a, b, qinner(irr2[a], irr2[b]))
assert sum(int(irr2[n][0]) ** 2 for n in names2) == 120
print("[ok] 2I: 9 irreps of dims", [int(irr2[n][0]) for n in names2], "orthonormal; sum of squares = 120")
print("SU(2) -> 2I branching (half-integer j, spin-orbit coupled quasiparticles):")
for tj in range(1, 16, 2):
    j = Rational(tj, 2)
    v = [chi_j(j, w) for w in qkeys]
    dec = {n: qinner(irr2[n], v) for n in names2}
    dec = {n: m for n, m in dec.items() if m != 0}
    print(f"  j={j}: " + " + ".join(f"{(str(m)+' ') if m!=1 else ''}{n}" for n, m in dec.items()))
print("pair decompositions in 2I (two quasiparticles in the same Kramers-type irrep):")
for n in ['2', "2'", '4', '6']:
    v = irr2[n]
    # antisymmetric square: (chi(g)^2 - chi(g^2))/2 ; symmetric: (chi(g)^2 + chi(g^2))/2
    def sq_class(w, sign):
        # g^2 has Re = 2w^2-1
        w2 = sp.nsimplify(sp.expand(2 * w * w - 1), [sqrt(5)])
        v2 = v[qkeys.index(w2)]
        vv = v[qkeys.index(w)]
        return sp.nsimplify(sp.simplify((vv * vv + sign * v2) / 2), [sqrt(5)])
    for sign, lab in [(-1, 'antisym'), (1, 'sym')]:
        s = [sq_class(w, sign) for w in qkeys]
        dec = {k: qinner(irr2[k], s) for k in names2}
        dec = {k: m for k, m in dec.items() if m != 0}
        print(f"  {n} x {n} ({lab}): " + " + ".join(f"{(str(m)+' ') if m!=1 else ''}{k}" for k, m in dec.items()))

# ---------------------------------------------------------------- print bases
print()
print("Explicit harmonic polynomial bases of the isotypic components (standard orientation:")
print("two-fold axes along x,y,z; three-fold along (1,1,1); five-fold axes listed above):")
for l in range(0, LMAX + 1):
    for irr in irreps:
        if (l, irr) in bases:
            print(f"\nl={l}, {irr} (dim {chi[irr][0]}, multiplicity {branching[(l, irr)]}):")
            for p in bases[(l, irr)]:
                print("   ", p)

# write a LaTeX fragment with the bases
with open('bases_fragment.tex', 'w') as f:
    for l in range(0, LMAX + 1):
        for irr in irreps:
            if (l, irr) in bases:
                f.write(f"\\paragraph{{$\\ell={l}$, ${irr[0]}_{{{irr[1:]}}}$ (multiplicity {branching[(l, irr)]}).}}\n")
                f.write("\\begin{align*}\n")
                items = [sp.latex(p) for p in bases[(l, irr)]]
                f.write(" \\\\\n".join("&" + it for it in items))
                f.write("\n\\end{align*}\n")
print("\n[written] bases_fragment.tex")
