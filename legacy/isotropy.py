"""Isotropy subgroups of I x U(1) (and time reversal) acting on each pairing channel:
for each subgroup K of I (up to conjugacy) and each 1-dim character chi of K, the fixed space
{eta : D(g) eta = chi(g) eta for all g in K}. States with 1-dim fixed space are fixed by symmetry."""
import pickle, numpy as np, sympy as sp, itertools, io, contextlib
from sympy import symbols, lambdify
from collections import Counter
data = pickle.load(open('gl_data.pkl','rb')); inv2 = pickle.load(open('gl_inv2.pkl','rb')); HH = pickle.load(open('gl_H.pkl','rb'))
x, y, z = symbols('x y z', real=True)
src = open('ih_basis.py').read().split('# ---------------------------------------------------------------- action on harmonic polynomials')[0].replace("LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6","LMAX=0")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
Rn = [np.array(R.evalf(30).tolist(), dtype=float) for R in I_mats]
def idx(M):
    for k, R in enumerate(Rn):
        if np.allclose(R, M, atol=1e-9): return k
    raise ValueError
def order(k):
    M = Rn[k].copy(); n = 1
    while not np.allclose(M, np.eye(3), atol=1e-9): M = M@Rn[k]; n += 1
    return n
def axis(k):
    w, v = np.linalg.eig(Rn[k]); a = np.real(v[:, np.argmin(abs(w-1))]); return a/np.linalg.norm(a)
def gen_group(gens):
    G = {0}; frontier = [0]
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                b = idx(Rn[a]@Rn[g])
                if b not in G: G.add(b); new.append(b)
        frontier = new
    return sorted(G)
# representatives
c2 = [k for k in range(60) if order(k)==2]; c3 = [k for k in range(60) if order(k)==3]; c5 = [k for k in range(60) if order(k)==5]
# choose C2 about z, C3 about (1,1,1), C5 about (phi,1,0)
kz = [k for k in c2 if abs(abs(axis(k)[2])-1)<1e-9][0]
k3 = [k for k in c3 if np.allclose(abs(axis(k)), np.ones(3)/np.sqrt(3), atol=1e-9)][0]
phi = (1+5**0.5)/2; a5 = np.array([phi,1,0])/np.sqrt(phi**2+1)
k5 = [k for k in c5 if np.allclose(abs(axis(k)), a5, atol=1e-9) and abs(np.trace(Rn[k])-(1+2*np.cos(2*np.pi/5)))<1e-9][0]
# perpendicular 2-folds for dihedral groups
def perp2(kax):
    a = axis(kax)
    return [k for k in c2 if abs(np.dot(axis(k), a))<1e-9][0]
subgroups = {
 'C2': gen_group([kz]), 'C3': gen_group([k3]), 'C5': gen_group([k5]),
 'D2': gen_group([kz, [k for k in c2 if abs(abs(axis(k)[0])-1)<1e-9][0]]),
 'D3': gen_group([k3, perp2(k3)]), 'D5': gen_group([k5, perp2(k5)]),
 'T':  gen_group([kz, [k for k in c2 if abs(abs(axis(k)[0])-1)<1e-9][0], k3]),
 'I':  list(range(60)),
}
for n, G in subgroups.items(): print(f"subgroup {n}: order {len(G)}")
# 1-dim characters: computed as homomorphisms K -> U(1) via generators
def characters(name, G):
    chars = []
    if name in ('C2','C3','C5'):
        n = len(G); g = {'C2':kz,'C3':k3,'C5':k5}[name]
        for m in range(n):
            chi = {}; el = 0
            for p in range(n):
                chi[el] = np.exp(2j*np.pi*m*p/n); el = idx(Rn[el]@Rn[g])
            chars.append((f"chi_{m}", chi))
    elif name in ('D3','D5'):
        n = len(G)//2; g = {'D3':k3,'D5':k5}[name]; h = perp2(g)
        for s in (1, -1):
            chi = {}
            for p in range(n):
                el = 0
                for _ in range(p): el = idx(Rn[el]@Rn[g])
                chi[el] = 1.0; chi[idx(Rn[el]@Rn[h])] = s
            chars.append((f"{'A1' if s==1 else 'A2'}", chi))
    elif name == 'D2':
        gx = [k for k in c2 if abs(abs(axis(k)[0])-1)<1e-9][0]; gy = [k for k in c2 if abs(abs(axis(k)[1])-1)<1e-9][0]
        for sx, sy in itertools.product((1,-1),(1,-1)):
            chi = {0:1.0, gx:sx, gy:sy, kz:sx*sy}
            chars.append((f"({sx:+d},{sy:+d},{sx*sy:+d})", chi))
    elif name == 'T':
        gx = [k for k in c2 if abs(abs(axis(k)[0])-1)<1e-9][0]
        for m in range(3):
            w = np.exp(2j*np.pi*m/3)
            chi = {}
            # T = D2 x <C3>: elements el = d * c3^p, chi = w^p
            D2 = gen_group([kz, gx])
            for p in range(3):
                C = np.eye(3)
                for _ in range(p): C = C@Rn[k3]
                for d in D2: chi[idx(Rn[d]@C)] = w**p
            chars.append((['A','1E','2E'][m], chi))
    elif name == 'I':
        chars.append(('A', {g:1.0 for g in G}))
    for nm, chi in chars:
        assert set(chi) == set(G), (name, nm, len(chi), len(G))
        for a in G:
            for b in G[::7]:
                assert abs(chi[a]*chi[b] - chi[idx(Rn[a]@Rn[b])]) < 1e-9, (name, nm)
    return chars
results = {}
for irr in ['T1','T2','G','H']:
    d, Bn, Ds, dec, norms2 = data[irr]
    Dn = [np.array(D.evalf(30).tolist(), dtype=float) for D in Ds]
    eta, etab, comps, invs, quartic = inv2[irr]
    fq = lambdify(list(eta)+list(etab), quartic, 'numpy')
    fs = [lambdify((x,y,z), b, 'numpy') for b in Bn]
    print(f"\n=== channel {irr} (dim {d}) ===")
    print(f"   {'K':3s} {'chi':12s} {'dim fix':>7s}  {'R_wc':>8s} {'|eta.eta|^2':>11s}  TRS?  min|D|/max  eta (moduli)")
    for name, G in subgroups.items():
        for cname, chi in characters(name, G):
            # fixed space: solve (D(g) - chi(g)) eta = 0 for all g in G
            A = np.vstack([Dn[g] - chi[g]*np.eye(d) for g in G])
            u, s, vh = np.linalg.svd(A)
            null = vh[np.sum(s > 1e-8):].conj().T
            k = null.shape[1]
            if k == 0: continue
            info = ''
            if k == 1:
                e = null[:, 0]; e /= np.linalg.norm(e)
                Rwc = fq(*e, *np.conj(e)).real
                S2 = abs(np.dot(e, e))**2
                # time reversal: exists g, phase with conj(e) = phase D(g) e ?
                trs = any(abs(np.vdot(np.conj(e), D@e)) > 1-1e-8 for D in Dn)
                th = np.linspace(0, np.pi, 300); ph = np.linspace(0, 2*np.pi, 600)
                T, P = np.meshgrid(th, ph, indexing='ij')
                X, Y, Z = np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)
                Dl = sum(c*f(X,Y,Z) for c, f in zip(e, fs)); Am = np.abs(Dl)
                info = f"{Rwc:8.4f} {S2:11.4f}  {'yes' if trs else 'NO '}   {Am.min()/Am.max():.3f}      {np.round(np.abs(e),3)}"
            print(f"   {name:3s} {cname:12s} {k:7d}  {info}")
