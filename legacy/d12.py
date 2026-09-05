"""d12.py -- dodecagonal group D12: character table check, assignment of in-plane harmonics
exp(+-i m phi) to irreps by m mod 12, and enforced-node conditions on the C2', C2'' axes and the
twelve-fold axis (Appendix B)."""
import numpy as np
n = 12
els = [('R', k) for k in range(n)] + [('P', k) for k in range(n)]   # rotations; C2 axes at k*pi/12 (k even: C2', odd: C2'')
def act(e, phi):
    t, k = e
    return phi + 2*np.pi*k/n if t == 'R' else 2*(k*np.pi/12) - phi
def chi(name, e):
    t, k = e
    if name == 'A1': return 1
    if name == 'A2': return 1 if t == 'R' else -1
    if name == 'B1': return (-1)**k if t == 'R' else (1 if k % 2 == 0 else -1)
    if name == 'B2': return (-1)**k if t == 'R' else (-1 if k % 2 == 0 else 1)
    m = int(name[1]); return 2*np.cos(2*np.pi*m*k/n) if t == 'R' else 0
irreps = ['A1', 'A2', 'B1', 'B2', 'E1', 'E2', 'E3', 'E4', 'E5']
for a in irreps:
    for b in irreps:
        assert abs(sum(chi(a, e)*chi(b, e) for e in els)/len(els) - (1 if a == b else 0)) < 1e-9
print('[ok] D12 character table orthonormal')
phis = np.linspace(0, 2*np.pi, 97)[:-1]
def rep_char(m, e):
    fs = [lambda p: np.cos(m*p), lambda p: np.sin(m*p)] if m > 0 else [lambda p: np.ones_like(p)]
    F = np.array([f(phis) for f in fs]).T; Fg = np.array([f(act(e, phis)) for f in fs]).T
    return np.trace(np.linalg.lstsq(F, Fg, rcond=None)[0])
for m in range(0, 13):
    dec = {a: int(round(sum(rep_char(m, e)*chi(a, e) for e in els)/len(els))) for a in irreps}
    print(f"m={m:2d}: {{{', '.join(f'{a}' for a, v in dec.items() if v)}}}")
for a in irreps:
    p, pp = chi(a, ('P', 0)), chi(a, ('P', 1)); has0 = abs(sum(chi(a, ('R', k)) for k in range(n))/n) > 1e-9
    lab = lambda v: 'contains' if v == 0 else ('yes' if v > 0 else 'no')
    print(f"{a}: trivial under C2' {lab(p)}; C2'' {lab(pp)}; C12 trivial {'yes' if has0 else 'no'}")
