import pickle, numpy as np, sympy as sp
from sympy import symbols, lambdify
from scipy.optimize import minimize
inv2 = pickle.load(open('gl_inv2.pkl','rb'))
np.random.seed(0)
for irr in ['T1','T2','G','H']:
    eta, etab, comps, invs, quartic = inv2[irr]
    d = len(eta)
    f = lambdify(list(eta)+list(etab), quartic, 'numpy')
    def R(v):
        e = v[:d] + 1j*v[d:]
        n = np.vdot(e, e).real
        return (f(*e, *np.conj(e)).real)/n**2
    best = []
    for trial in range(60):
        v0 = np.random.randn(2*d)
        res = minimize(R, v0, method='BFGS', options={'gtol':1e-10})
        best.append((res.fun, res.x))
    best.sort(key=lambda t: t[0])
    vals = np.array([b[0] for b in best])
    print(f"\n[{irr}] min R = int|Delta|^4/(int|Delta|^2)^2 = {vals[0]:.6f}   (fraction of restarts within 1e-6: {np.mean(vals < vals[0]+1e-6):.2f}); max over restarts {vals[-1]:.4f}")
    e = best[0][1][:d] + 1j*best[0][1][d:]; e /= np.linalg.norm(e)
    S2 = abs(np.dot(e, e))**2
    print(f"      at minimum: |eta.eta|^2 = {S2:.3e}")
    # distinct minima? compare invariants of several near-best solutions
    for k in range(0, 60, 12):
        ek = best[k][1][:d] + 1j*best[k][1][d:]; ek /= np.linalg.norm(ek)
        print(f"      restart {k}: R={best[k][0]:.6f}  |eta.eta|^2={abs(np.dot(ek,ek))**2:.3e}")
pickle.dump(None, open('/dev/null','wb'))
