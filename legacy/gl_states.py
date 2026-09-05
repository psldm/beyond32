import pickle, numpy as np, sympy as sp
from sympy import symbols, lambdify, sqrt, Matrix, S
from scipy.optimize import minimize
import warnings; warnings.filterwarnings('ignore')
data = pickle.load(open('gl_data.pkl','rb')); inv2 = pickle.load(open('gl_inv2.pkl','rb')); H = pickle.load(open('gl_H.pkl','rb'))
x, y, z = symbols('x y z', real=True)
# numeric group matrices
src = open('ih_basis.py').read().split('# ---------------------------------------------------------------- action on harmonic polynomials')[0].replace("LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6","LMAX=0")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): exec(src)
def angle_class(R):
    c = float(cos_theta(R)); return {1.0:'E', round(float((sp.sqrt(5)-1)/4),6):'C5', round(float(-(sp.sqrt(5)+1)/4),6):'C5^2', -0.5:'C3', -1.0:'C2'}[round(c,6)]
def stabilizer(irr, e):
    d, Bn, Ds, dec, norms2 = data[irr]
    Dn = [np.array(D.evalf(30).tolist(), dtype=float) for D in Ds]
    stab = []; tstab = []
    for R, D in zip(I_mats, Dn):
        ov = abs(np.vdot(e, D@e)); ovT = abs(np.vdot(np.conj(e), D@e))
        if ov > 1-1e-8: stab.append(angle_class(R))
        if ovT > 1-1e-8: tstab.append(angle_class(R))
    from collections import Counter
    return Counter(stab), Counter(tstab)
def nodes(irr, e, ngrid=400):
    d, Bn, Ds, dec, norms2 = data[irr]
    fs = [lambdify((x, y, z), b, 'numpy') for b in Bn]
    th = np.linspace(0, np.pi, ngrid); ph = np.linspace(0, 2*np.pi, 2*ngrid)
    T, P = np.meshgrid(th, ph, indexing='ij')
    X, Y, Z = np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)
    D = sum(c*f(X, Y, Z) for c, f in zip(e, fs))
    A = np.abs(D); A /= A.max()
    frac_small = np.mean(A < 0.02)   # fraction of sphere area (weighted) with tiny gap
    w = np.sin(T); frac = np.sum(w*(A < 0.02))/np.sum(w)
    return A.min(), frac
# ---- G minimiser
eta, etab, comps, invs, quartic = inv2['G']; d = 4
f = lambdify(list(eta)+list(etab), quartic, 'numpy')
def R(v):
    e = v[:d]+1j*v[d:]; n = np.vdot(e,e).real; return f(*e, *np.conj(e)).real/n**2
np.random.seed(1); best=None
for t in range(80):
    res = minimize(R, np.random.randn(2*d), method='BFGS', options={'gtol':1e-12})
    if best is None or res.fun < best.fun: best = res
e = best.x[:d]+1j*best.x[d:]; e /= np.linalg.norm(e)
print(f"[G] weak-coupling minimum R = {best.fun:.9f}")
print("    eta (moduli) =", np.round(np.abs(e),4), " |eta.eta|^2 =", round(abs(np.dot(e,e))**2,6))
st, tst = stabilizer('G', e)
print("    stabilizer in I (up to U(1) phase):", dict(st), " order", sum(st.values()))
print("    time-reversal x g stabilizer:", dict(tst))
mn, fr = nodes('G', e); print(f"    min|Delta|/max = {mn:.4f}; sphere fraction with |Delta|<2% of max: {fr:.4f}")
# null-cone local minimum for G
def Rc(v):
    e = v[:d]+1j*v[d:]; n = np.vdot(e,e).real; return f(*e, *np.conj(e)).real/n**2 + 50*abs(np.dot(e,e))**2/n**2
best2=None
for t in range(80):
    res = minimize(Rc, np.random.randn(2*d), method='BFGS', options={'gtol':1e-12})
    if best2 is None or res.fun < best2.fun: best2 = res
e2 = best2.x[:d]+1j*best2.x[d:]; e2 /= np.linalg.norm(e2)
print(f"    best null-cone state: R = {R(best2.x):.9f}, |eta.eta|^2 = {abs(np.dot(e2,e2))**2:.2e}")
st, tst = stabilizer('G', e2); print("    its stabilizer:", dict(st), "| T x g:", dict(tst))
# ---- H candidate states on the null cone, with the icosahedral invariants
etaH, etabH = H['eta'], H['etab']
Fs = {k: lambdify(list(etaH)+list(etabH), H[k], 'numpy') for k in ['S1','S2','N2','N4G','N4H','CR','CIi']}
lam = float(H['lam'])
dH, BnH, DsH, decH, normsH = data['H']
fsH = [lambdify((x,y,z), b, 'numpy') for b in BnH]
# express a target gap function Delta(n) in the orthonormal H basis by sphere projection (numeric quadrature)
def coords_of(func, n=200):
    th = np.linspace(0, np.pi, n); ph = np.linspace(0, 2*np.pi, 2*n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij'); w = np.sin(T)*(np.pi/n)*(np.pi/n)
    X, Y, Z = np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)
    D = func(X, Y, Z)
    return np.array([np.sum(w*np.conj(fH(X,Y,Z))*D)/(4*np.pi) for fH in fsH])
phi = (1+5**0.5)/2
axes = {'C5': np.array([phi,1,0])/np.sqrt(phi**2+1), 'C3': np.array([1,1,1])/np.sqrt(3), 'C2': np.array([0,0,1.])}
def axial(axis):
    a = axis; t = np.cross(a, [0.3,0.5,0.7]); t/=np.linalg.norm(t); u = np.cross(a, t)
    return lambda X,Y,Z: ((X*t[0]+Y*t[1]+Z*t[2]) + 1j*(X*u[0]+Y*u[1]+Z*u[2]))**2
om = np.exp(2j*np.pi/3)
cands = {f'axial(Y22) about {k}': axial(a) for k, a in axes.items()}
cands['cyclic x^2+w y^2+w^2 z^2'] = lambda X,Y,Z: X**2 + om*Y**2 + om**2*Z**2
cands['real uniaxial 3z^2-1 (C5 axis)'] = (lambda a: (lambda X,Y,Z: 3*(X*a[0]+Y*a[1]+Z*a[2])**2 - 1))(axes['C5'])
print("\n[H] candidate states: values of the quartic invariants at |eta|=1")
print(f"    {'state':34s} {'S2':>7s} {'N2':>7s} {'N4G':>7s} {'N4H':>7s} {'C_R':>8s} {'C_I':>8s}  stabilizer(order)  min|D|")
for name, fn in cands.items():
    e = coords_of(fn); e /= np.linalg.norm(e)
    args = (*e, *np.conj(e))
    vals = {k: Fs[k](*args) for k in Fs}
    CI = (vals['CIi']/1j).real; CR = vals['CR'].real
    st, tst = stabilizer('H', e)
    mn, fr = nodes('H', e)
    print(f"    {name:34s} {vals['S2'].real:7.4f} {vals['N2'].real:7.4f} {vals['N4G'].real:7.4f} {vals['N4H'].real:7.4f} {CR/np.sqrt(lam):8.4f} {CI/np.sqrt(lam):8.4f}  {sum(st.values()):>3d}  {mn:.3f}")
