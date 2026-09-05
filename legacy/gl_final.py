import pickle, numpy as np, sympy as sp, io, contextlib
from sympy import symbols, lambdify, Rational, sqrt, nsimplify
from scipy.optimize import minimize_scalar, minimize
data = pickle.load(open('gl_data.pkl','rb')); inv2 = pickle.load(open('gl_inv2.pkl','rb')); HH = pickle.load(open('gl_H.pkl','rb'))
x, y, z = symbols('x y z', real=True)
# ---- G channel ground state: eta = (1,1,1, i*kappa) (C3-invariant stratum, phase in quadrature)
eta, etab, comps, invs, quartic = inv2['G']
kap, ph = symbols('kappa phi_', real=True)
sub = {eta[0]:1, eta[1]:1, eta[2]:1, eta[3]: kap*sp.exp(sp.I*ph), etab[0]:1, etab[1]:1, etab[2]:1, etab[3]: kap*sp.exp(-sp.I*ph)}
Q = sp.simplify(quartic.subs(sub)); N = (3 + kap**2)**2
Rk = sp.simplify(Q/N)
print("[G] R(kappa, phase) =", sp.simplify(sp.expand(Rk.rewrite(sp.cos))))
f = lambdify((kap, ph), Rk, 'numpy')
res = minimize(lambda v: float(np.real(f(v[0], v[1]))), [1.7, np.pi/2], method='Nelder-Mead', options={'xatol':1e-12,'fatol':1e-14})
print("    numeric min: kappa =", res.x[0], " phase =", res.x[1], " R =", res.fun)
# exact: with phase = pi/2, minimise in kappa^2 = u
u = symbols('u', positive=True)
Ru = sp.simplify(Rk.subs(ph, sp.pi/2).subs(kap, sp.sqrt(u)))
Ru = sp.nsimplify(sp.simplify(Ru), [sqrt(5)])
print("    R(u) at phase pi/2:", sp.factor(Ru))
dR = sp.together(sp.diff(Ru, u)); num = sp.numer(dR)
sols = sp.solve(sp.expand(num), u)
sols = [s for s in sols if s.is_real and s > 0]
print("    stationary u:", [sp.nsimplify(s) for s in sols], [float(s) for s in sols])
for s in sols:
    print("    R =", sp.nsimplify(sp.simplify(Ru.subs(u, s))), float(Ru.subs(u, s)))
# nodes of the G ground state
d, Bn, Ds, dec, norms2 = data['G']
fs = [lambdify((x,y,z), b, 'numpy') for b in Bn]
kk = float(sp.sqrt(sols[0])) if sols else res.x[0]
e = np.array([1,1,1,1j*kk]); e /= np.linalg.norm(e)
th = np.linspace(0, np.pi, 1200); phg = np.linspace(0, 2*np.pi, 2400)
T, P = np.meshgrid(th, phg, indexing='ij'); X, Y, Z = np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)
D = sum(c*f(X,Y,Z) for c, f in zip(e, fs)); A = np.abs(D)/np.abs(D).max()
idx = np.argwhere(A < 0.01)
pts = np.array([[X[i,j],Y[i,j],Z[i,j]] for i,j in idx])
# cluster node points
clusters = []
for p in pts:
    for c in clusters:
        if np.linalg.norm(p - c[0]) < 0.08: c.append(p); break
    else: clusters.append([p])
print(f"    G ground state: {len(clusters)} node clusters (|Delta|<1%); centres:")
for c in clusters: print("      ", np.round(np.mean(c, axis=0), 3))
# ---- H: invariants for the axial states about the C5 axis with l_z = +-1 (Y21-type) and +-2 (Y22)
etaH, etabH = HH['eta'], HH['etab']
Fs = {k: lambdify(list(etaH)+list(etabH), HH[k], 'numpy') for k in ['S1','S2','N2','N4G','N4H','CR','CIi']}
lam = float(HH['lam'])
dH, BnH, DsH, decH, normsH = data['H']
fsH = [lambdify((x,y,z), b, 'numpy') for b in BnH]
def coords_of(func, n=300):
    th = np.linspace(0, np.pi, n); ph = np.linspace(0, 2*np.pi, 2*n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij'); w = np.sin(T)*(np.pi/n)*(np.pi/n)
    X, Y, Z = np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)
    Dv = func(X, Y, Z)
    return np.array([np.sum(w*np.conj(fH(X,Y,Z))*Dv)/(4*np.pi) for fH in fsH])
phi = (1+5**0.5)/2
axes = {'C5': np.array([phi,1,0])/np.sqrt(phi**2+1), 'C3': np.array([1,1,1])/np.sqrt(3), 'C2': np.array([0,0,1.])}
def frame(a):
    t = np.cross(a, [0.3,0.5,0.7]); t/=np.linalg.norm(t); u = np.cross(a, t); return t, u
def Y22(a):
    t, u = frame(a); return lambda X,Y,Z: ((X*t[0]+Y*t[1]+Z*t[2]) + 1j*(X*u[0]+Y*u[1]+Z*u[2]))**2
def Y21(a):
    t, u = frame(a); return lambda X,Y,Z: (X*a[0]+Y*a[1]+Z*a[2])*((X*t[0]+Y*t[1]+Z*t[2]) + 1j*(X*u[0]+Y*u[1]+Z*u[2]))
om = np.exp(2j*np.pi/3)
cands = {}
for k, a in axes.items():
    cands[f'Y22 about {k}'] = Y22(a); cands[f'Y21 about {k}'] = Y21(a)
cands['cyclic (T)'] = lambda X,Y,Z: X**2 + om*Y**2 + om**2*Z**2
print("\n[H] null-cone candidate states, quartic invariants at |eta|=1 (C_R, C_I with J normalised to an isometry)")
print(f"    {'state':16s} {'N2':>7s} {'N4G':>7s} {'N4H':>7s} {'C_R':>8s} {'C_I':>8s}")
rows = {}
for name, fn in cands.items():
    e = coords_of(fn); e /= np.linalg.norm(e); args = (*e, *np.conj(e))
    v = {k: Fs[k](*args) for k in Fs}
    CR = v['CR'].real/np.sqrt(lam); CI = (v['CIi']/1j).real/np.sqrt(lam)
    rows[name] = (v['N2'].real, v['N4G'].real, v['N4H'].real, CR, CI)
    print(f"    {name:16s} {v['N2'].real:7.4f} {v['N4G'].real:7.4f} {v['N4H'].real:7.4f} {CR:8.4f} {CI:8.4f}   S2={abs(v['S2']):.1e}")
pickle.dump(rows, open('H_rows.pkl','wb'))
