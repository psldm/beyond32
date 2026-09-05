import numpy as np, itertools, io, contextlib
src = open('ih_basis.py').read().split('# ---------------------------------------------------------------- action on harmonic polynomials')[0].replace("LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6","LMAX=0")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
Rn = [np.array(R.evalf(30).tolist(), dtype=float) for R in I_mats]
phi = (1+5**0.5)/2
def cyc(v): return [np.array(v), np.array([v[2],v[0],v[1]]), np.array([v[1],v[2],v[0]])]
def signs(v, idx):
    out=[]
    for s in itertools.product((1,-1), repeat=len(idx)):
        w=np.array(v, dtype=float)
        for k,i in enumerate(idx): w[i]*=s[k]
        out.append(w)
    return out
def orbit(p):
    pts=[]
    for R in Rn:
        q=R@p
        if not any(np.allclose(q,x,atol=1e-9) for x in pts): pts.append(q)
    return pts
# vertex sets (unit sphere directions)
ico = orbit(np.array([phi,1,0])/np.sqrt(phi**2+1))                       # 12, five-fold axes
dod = orbit(np.array([1,1,1])/np.sqrt(3))                                 # 20, three-fold axes
idd = orbit(np.array([0,0,1.]))                                           # 30, two-fold axes
# truncated icosahedron: vertices of the soccer ball: orbit of (0, 1, 3phi)/norm (standard coordinates)
tri = orbit(np.array([0,1,3*phi])/np.linalg.norm([0,1,3*phi]))            # 60
sets = {'icosahedron 12':ico,'dodecahedron 20':dod,'icosidodecahedron 30':idd,'truncated icosahedron 60':tri}
# I_h = I x {E, i}; characters of I_h irreps: Gamma_g(i*R)=chi(R), Gamma_u(i*R)=-chi(R)
cos_cls = {1.0:0, round(float((5**0.5-1)/4),6):1, round(float(-(5**0.5+1)/4),6):2, -0.5:3, -1.0:4}
def cls(R): return cos_cls[round((np.trace(R)-1)/2,6)]
chin = {g:[float(v) for v in chi[g]] for g in chi}
for name, pts in sets.items():
    assert len(pts) in (12,20,30,60), (name, len(pts))
    def fixed(M): return sum(1 for p in pts if any(np.allclose(M@p, q, atol=1e-9) for q in pts) and np.allclose(M@p, p, atol=1e-9))
    mult = {}
    for g in chi:
        for par, sgn in (('g',1),('u',-1)):
            s = 0.0
            for R in Rn:
                s += chin[g][cls(R)]*fixed(R) + sgn*chin[g][cls(R)]*fixed(-R)
            mult[g+'_'+par] = round(s/120, 6)
    mult = {k:int(v) for k,v in mult.items() if abs(v)>1e-9}
    dim = sum(v*int(chin[k.split('_')[0]][0]) for k,v in mult.items())
    print(f"{name:26s} {mult}  dim {dim}")
