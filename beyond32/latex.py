"""LaTeX fragments of the tables and displayed equations of the paper.

``write_tables(results, outdir)`` writes one ``.tex`` file per table, named after the
paper's labels (tab_charI, tab_char2I, tab_branching, tab_shells, tab_restrict, tab_GL,
tab_states, tab_Hcand, tab_D12) plus the displayed data of Eqs. (4)-(11) (eq_bases),
(12)-(13) (tab_double_group), (15)-(16) (tab_molien) and (18)-(23) (eq_invariants).  Each
fragment is a ``tabular`` (or ``align*``) environment that can be ``\\input`` into a
``table`` float; the caption text of the paper is reproduced as a comment line.
"""
from __future__ import annotations

import math
import os
import re
from fractions import Fraction
from typing import Any, Dict, List

TABLE_FILES = ("tab_charI", "tab_char2I", "tab_branching", "tab_shells", "tab_restrict", "tab_GL",
               "tab_states", "tab_Hcand", "tab_D12", "tab_molien", "tab_double_group",
               "eq_bases", "eq_invariants")


# --------------------------------------------------------------------------- symbols
def tex_irrep(name: str) -> str:
    """'T1' -> 'T_1', 'T1u' -> 'T_{1u}', '4s' -> '4_s', '1E' -> '{}^1E', 'chi1' -> '\\chi_{1}'."""
    name = str(name)
    m = re.fullmatch(r"chi_?(-?\d+)", name)
    if m:
        return rf"\chi_{{{m.group(1)}}}"
    m = re.fullmatch(r"([12])E", name)
    if m:
        return rf"{{}}^{{{m.group(1)}}}E"
    if name == "4s":
        return "4_s"
    if name == "2'":
        return "2'"
    m = re.fullmatch(r"([A-Z])(\d*)([gu]?)", name)
    if m:
        base, num, par = m.groups()
        sub = num + par
        return base + (f"_{{{sub}}}" if len(sub) > 1 else (f"_{sub}" if sub else ""))
    return name


def tex_decomposition(dec: Dict[str, int], order: List[str] | None = None) -> str:
    keys = list(dec) if order is None else [k for k in order if k in dec]
    parts = []
    for k in keys:
        m = dec[k]
        if not m:
            continue
        if m > 1:
            parts.append(f"{m}" + (r"\cdot " if k[0].isdigit() else r"\,") + tex_irrep(k))
        else:
            parts.append(tex_irrep(k))
    return r" \oplus ".join(parts) if parts else "0"


def tex_frac(s: str) -> str:
    """'-2/35' -> '-\\tfrac{2}{35}', '3' -> '3'."""
    s = str(s)
    m = re.fullmatch(r"(-?)(\d+)/(\d+)", s)
    if m:
        return rf"{m.group(1)}\tfrac{{{m.group(2)}}}{{{m.group(3)}}}"
    return s


def tex_num(s: Any) -> str:
    """A number given as a string in phi-form ('1 - varphi', '3/2') or sqrt5-form -> LaTeX."""
    s = str(s)
    s = s.replace("varphi", r"\varphi").replace("sqrt(5)", r"\sqrt5").replace("*", "")
    m = re.fullmatch(r"(-?)(\d+)/(\d+)", s)
    if m:
        return rf"{m.group(1)}\tfrac{{{m.group(2)}}}{{{m.group(3)}}}"
    s = re.sub(r"(\d+)/(\d+)", r"\\tfrac{\1}{\2}", s)
    return s


def _phi_coeff_parts(p: str, q: str):
    """(negative, body, compound) for the coefficient p + q*varphi (rational strings), written
    with a common denominator and the positive term first, as in the paper:
    '3', '3\varphi', '3\varphi - 3', '3 - 21\varphi', '\tfrac{1 - 7\varphi}{15}'."""
    P, Q = Fraction(p), Fraction(q)
    d = P.denominator * Q.denominator // math.gcd(P.denominator, Q.denominator)
    np_, nq = P.numerator * (d // P.denominator), Q.numerator * (d // Q.denominator)

    def phi(n):
        return ("" if abs(n) == 1 else str(abs(n))) + r"\varphi"

    compound = False
    if nq == 0:
        neg, num = np_ < 0, str(abs(np_))
    elif np_ == 0:
        neg, num = nq < 0, phi(nq)
    else:
        compound = True
        if np_ < 0 and nq < 0:
            neg, num = True, f"{abs(np_)} + {phi(nq)}"
        elif np_ > 0 and nq > 0:
            neg, num = False, f"{np_} + {phi(nq)}"
        elif np_ > 0 > nq:
            neg, num = False, f"{np_} - {phi(nq)}"
        else:
            neg, num = False, f"{phi(nq)} - {abs(np_)}"
    body = num if d == 1 else rf"\tfrac{{{num}}}{{{d}}}"
    return neg, body, compound and d == 1


def tex_phi_coeff(p: str, q: str) -> str:
    """The coefficient p + q*varphi as LaTeX (see _phi_coeff_parts)."""
    neg, body, _ = _phi_coeff_parts(p, q)
    return ("-" if neg else "") + body


def tex_poly_terms(terms) -> str:
    """[[a,b,c], p, q] term lists (results.json 'basis_functions_terms') -> LaTeX polynomial."""
    out = ""
    for (a, b, c), p, q in terms:
        mono = "".join(f"{v}^{{{e}}}" if e > 1 else v for v, e in zip("xyz", (a, b, c)) if e)
        neg, body, paren = _phi_coeff_parts(p, q)
        if body == "1" and mono:
            body = ""
        elif paren and mono:
            body = f"({body})"
        term = (body + ("\\," if body and mono else "") + mono) if mono else body
        if not out:
            out = ("-" if neg else "") + term
        else:
            out += (" - " if neg else " + ") + term
    return out


def _num4(x: float) -> str:
    if abs(x) < 5e-5:
        return "0"
    return f"{x:.4f}"


def _frac_or_num(frac: str | None, x: float, nd: int = 4) -> str:
    if frac:
        p, q = frac.split("/") if "/" in frac else (frac, "1")
        return rf"\tfrac{{{p}}}{{{q}}}" if q != "1" else p
    return f"{x:.{nd}f}"


def _tex_sci(x: float, nd: int = 1) -> str:
    """2.2e-03 -> '2.2 \\times 10^{-3}'."""
    mant, exp = f"{x:.{nd}e}".split("e")
    return rf"{mant} \times 10^{{{int(exp)}}}"


# --------------------------------------------------------------------------- tables
def tab_charI(res) -> str:
    g = res["groups"]["I"]
    cols = ["E", "12\\,C_5", "12\\,C_5^2", "20\\,C_3", "15\\,C_2"]
    lines = ["% Table 1: character table of I = A5 (verified: orthonormality, class sizes); dims 1,3,3,4,5; sum d^2 = 60",
             r"\begin{tabular}{l" + "r" * 5 + "}", r"\hline",
             " & ".join([""] + [f"${c}$" for c in cols]) + r" \\", r"\hline"]
    for irr, row in g["character_table"].items():
        lines.append(" & ".join([f"${tex_irrep(irr)}$"] + [f"${tex_num(c)}$" for c in row]) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tab_char2I(res) -> str:
    g = res["groups"]["2I"]
    angles = {0: "0", 72: r"2\pi/5", 120: r"2\pi/3", 144: r"4\pi/5", 180: r"\pi", 216: r"6\pi/5",
              240: r"4\pi/3", 288: r"8\pi/5", 360: r"2\pi"}
    lines = ["% Table 2: character table of the binary icosahedral group 2I (classes by SU(2) angle; -1 is alpha = 2 pi)",
             r"\begin{tabular}{l" + "r" * 9 + "}", r"\hline",
             " & ".join([r"$\alpha$"] + [f"${angles[a]}$" for a in g["angles_deg"]]) + r" \\",
             " & ".join(["size"] + [str(s) for s in g["class_sizes"]]) + r" \\", r"\hline"]
    for irr, row in g["character_table"].items():
        if irr == "2":
            lines.append(r"\hline")
        lines.append(" & ".join([f"${tex_irrep(irr)}$"] + [f"${tex_num(c)}$" for c in row]) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


_BRANCHING_REMARKS = {0: "conventional $s$", 1: "$p$-wave: three components",
                      2: "all of $d$-wave is one five-component channel", 3: "first four-component channel",
                      4: "", 5: "", 6: r"first non-trivial $A$: ``extended $s$''"}


def tab_branching(res) -> str:
    br = res["harmonics"]["branching_by_characters"]
    lines = ["% Table 3: branching SO(3) -> I of the spherical harmonics (character formula and explicit projection agree)",
             r"\begin{tabular}{rll}", r"\hline", r"$\ell$ & decomposition & remark \\", r"\hline"]
    for l, dec in br.items():
        order = ["A", "T1", "T2", "G", "H"]
        lines.append(f"{l} & ${tex_decomposition(dec, order)}$ & {_BRANCHING_REMARKS.get(int(l), '')} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


_SHELL_REALISED = {"12": "icosahedron (Bergman 1st, 3rd; Tsai 2nd; triacontahedron 5-fold vertices)",
                   "20": "dodecahedron (Bergman 2nd; Tsai 1st; triacontahedron 3-fold vertices)",
                   "30": "icosidodecahedron (Tsai 3rd); icosahedron and dodecahedron edges",
                   "60": "soccer ball (Bergman 3rd); triacontahedron edges (Tsai); icosidodecahedron edges"}


def tab_shells(res) -> str:
    rows = res["shells"]["table"]
    lines = ["% Table 4: permutation representations of I_h on its orbits (decomposition under I_h; the I decomposition in the last column)",
             r"\begin{tabular}{rllll}", r"\hline",
             r"orbit & stabiliser & realised by & decomposition under $I_h$ & under $I$ \\", r"\hline"]
    order_ih = [n + p for p in "gu" for n in ["A", "T1", "T2", "G", "H"]]
    for r in rows:
        stab = _tex_point_group(r["stabiliser_Ih"])
        lines.append(f"{r['size']} & ${stab}$ & {_SHELL_REALISED[r['orbit']]} & "
                     f"${tex_decomposition(r['decomposition_Ih'], order_ih)}$ & "
                     f"${tex_decomposition(r['decomposition_I'], ['A', 'T1', 'T2', 'G', 'H'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _tex_point_group(s: str) -> str:
    m = re.fullmatch(r"([A-Z])(\d*)([a-z]*)", s)
    if not m:
        return s
    base, num, suf = m.groups()
    sub = num + suf
    return base + (f"_{{{sub}}}" if sub else "")


def tab_restrict(res) -> str:
    tab = res["restrictions"]["table"]
    lines = ["% Table 5: restriction of the irreps of I to T (cubic approximants), D5, D3 and D2 (B1, B2, B3 invariant under C2 about z, y, x)",
             r"\begin{tabular}{lllll}", r"\hline",
             r"$\Gamma$ & $\downarrow T$ & $\downarrow D_5$ & $\downarrow D_3$ & $\downarrow D_2$ \\", r"\hline"]
    for irr in ["A", "T1", "T2", "G", "H"]:
        cells = [f"${tex_decomposition(tab[irr][sg])}$" for sg in ["T", "D5", "D3", "D2"]]
        lines.append(" & ".join([f"${tex_irrep(irr)}$"] + cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tab_GL(res) -> str:
    s2 = res["gl"]["sym2"]
    lines = ["% Table 6: quartic Ginzburg-Landau invariants per channel",
             r"\begin{tabular}{llcc}", r"\hline",
             r"$\Gamma$ & $\mathrm{Sym}^2\Gamma$ & Hermitian forms & quartic terms in $F$ \\", r"\hline"]
    for n in ["T1", "T2", "G", "H"]:
        lines.append(f"${tex_irrep(n)}$ & ${tex_decomposition(s2[n]['sym2'], ['A', 'T1', 'T2', 'G', 'H'])}$ & "
                     f"{s2[n]['hermitian_forms']} & {s2[n]['quartic_terms']} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _tex_isotropy(sub: str, chi: str) -> str:
    return f"{_tex_point_group(sub)}\\,{tex_irrep(chi)}"


def tab_states(res) -> str:
    """Table 7: the symmetry-fixed states (one-dimensional fixed spaces whose stabiliser is K),
    grouped per channel by (R, time reversal); plus the weak-coupling G ground state."""
    rows = [r for r in res["gl"]["isotropy"] if r["table7"]]
    gs = res["gl"]["G_stratum"]
    lines = ["% Table 7: symmetry-fixed states per channel (from the exact representation matrices); R_wc = int|Delta|^4/(int|Delta|^2)^2; TR: I2 = I1",
             "% last column: min|Delta|/max|Delta| on the 300 x 600 grid -- values of 0.001-0.002 are the grid residual of point nodes (the paper's 'points'), not small gaps",
             r"\begin{tabular}{llrlr}", r"\hline",
             r"channel & isotropy $(K,\chi)$ & $R_{\rm wc}$ & TR & $\min|\Delta|/\max|\Delta|$ (grid) \\", r"\hline"]
    for ch in ["T1", "T2", "G", "H"]:
        groups: Dict[tuple, list] = {}
        for r in rows:
            if r["channel"] != ch:
                continue
            key = (round(r["R"], 4), r["time_reversal"])
            groups.setdefault(key, []).append(r)
        for (R, tr), rs in sorted(groups.items(), key=lambda kv: -kv[0][0]):
            labels = []
            for r in rs:
                lab = _tex_isotropy(r["subgroup"], r["character"])
                if lab not in labels:
                    labels.append(lab)
            # the paper quotes exact fractions for T1, T2, H and four decimals for G
            frac = rs[0]["R_fraction"] if ch != "G" else None
            lines.append(f"${tex_irrep(ch)}$ & ${';\\ '.join(labels)}$ & ${_frac_or_num(frac, R)}$ & "
                         f"{'yes' if tr else 'no'} & {min(r['min_gap'] for r in rs):.3f} \\\\")
        if ch == "G":
            lines.append(f"$G$ & $C_3\\,\\chi_0$ stratum, Eq.~(23): $\\kappa = {gs['kappa']:.3f}$, "
                         f"$\\phi_0 = {gs['phi0_deg']:.1f}^\\circ$ & ${gs['R']:.4f}$ & no & "
                         f"{res['gl']['G_ground_state']['min_gap']:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tab_Hcand(res) -> str:
    rows = res["gl"]["H_candidates"]
    names = {"Y22 about C5": r"$Y_{2,\pm2}$ about $C_5$", "Y22 about C3": r"$Y_{2,\pm2}$ about $C_3$",
             "Y22 about C2": r"$Y_{2,\pm2}$ about $C_2$", "Y21 about C5": r"$Y_{2,\pm1}$ about $C_5$",
             "Y21 about C3": r"$Y_{2,\pm1}$ about $C_3$", "Y21 about C2": r"$Y_{2,\pm1}$ about $C_2$",
             "cyclic (T)": r"cyclic ($T$)", "cyclic (T), other chirality": r"cyclic ($T$), other chirality"}
    lines = ["% Table 8: anisotropic quartic invariants of the null-cone H states at |eta| = 1 (I2 = 0, N2 + N4 = 10/7); Re C with J an isometry",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             r"state & $N_2$ & $N_{4G}$ & $N_{4H}$ & $\mathrm{Re}\,C$ & $\mathrm{Im}\,C$ \\", r"\hline"]
    for k, r in rows.items():
        if abs(r["I2"]) > 1e-6:          # Table 8 lists the null-cone states only
            continue
        lines.append(f"{names.get(k, k)} & ${_num4(r['N2'])}$ & ${_num4(r['N4G'])}$ & ${_num4(r['N4H'])}$ & "
                     f"${_num4(r['ReC'])}$ & ${_num4(r['ImC'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tab_D12(res) -> str:
    d = res["d12"]
    lines = ["% Section 10 ('Appendix 10' in the paper's cross-references): D12 -- assignment of the in-plane harmonics e^{+-i m phi} to irreps by m mod 12 (Eq. 24), enforced nodes, Sym^2 E_m",
             "% residue 0 stands for m = 12, 24, ...: the constant m = 0 alone is A_1, A_2 (sin m phi) first appears at m = 12",
             r"\begin{tabular}{ll}", r"\hline", r"$m \bmod 12$ & irreps \\", r"\hline"]
    for r in d["residues"]:
        note = r"\ (m \ge 12;\ m = 0:\ A_1)" if r["residue"] == 0 else ""
        lines.append(f"{r['residue']} & ${_tex_d12_dec(r['irreps'])}{note}$ \\\\")
    lines += [r"\hline", r"\end{tabular}", "",
              r"\begin{tabular}{lccc}", r"\hline",
              r"irrep & node on $C_2'$ & node on $C_2''$ & node on the 12-fold axis \\", r"\hline"]
    for r in d["nodes"]:
        c2p, c2pp = r["C2'"], r["C2''"]
        lines.append(f"${tex_irrep(r['irrep'])}$ & {c2p} & {c2pp} & {r['axis']} \\\\")
    lines += [r"\hline", r"\end{tabular}", "",
              r"\begin{tabular}{llc}", r"\hline", r"$E_m$ & $\mathrm{Sym}^2 E_m$ & quartic invariants \\", r"\hline"]
    for r in d["sym2"]:
        lines.append(f"${tex_irrep(r['irrep'])}$ & ${_tex_d12_dec(r['Sym^2'])}$ & {r['quartic invariants']} \\\\")
    lines += [r"\hline", r"\end{tabular}", "",
              f"% weak coupling on the circle: R = {d['weak_coupling']['real']} (real), {d['weak_coupling']['chiral']} (chiral)"]
    return "\n".join(lines) + "\n"


def _tex_d12_dec(s: str) -> str:
    parts = [p.strip() for p in s.split("+")]
    out = []
    for p in parts:
        m = re.fullmatch(r"(\d*)\s*([A-Z]\d?)", p)
        if m:
            mult, irr = m.groups()
            out.append((f"{mult}\\," if mult else "") + tex_irrep(irr))
        else:
            out.append(p)
    return r" \oplus ".join(out)


def tab_molien(res) -> str:
    m = res["molien"]
    lines = ["% Eqs. (15)-(16): m_A(l) for l = 0..30 (= coefficients of (1+t^15)/((1-t^6)(1-t^10))) and the S^3/2I spectrum (k, mult)",
             r"\begin{tabular}{l" + "r" * 31 + "}", r"\hline",
             " & ".join([r"$\ell$"] + [str(l) for l in range(31)]) + r" \\",
             " & ".join([r"$m_A(\ell)$"] + [str(x) for x in m["m_A"]]) + r" \\", r"\hline", r"\end{tabular}", "",
             r"\begin{align*}", r"\sum_\ell m_A(\ell)\,t^\ell &= \frac{1+t^{15}}{(1-t^6)(1-t^{10})}, \\",
             r"(k,\mathrm{mult}) &= " + ",\\ ".join(f"({k},{v})" for k, v in m["spectrum"]) + r",\ \ldots",
             r"\end{align*}"]
    return "\n".join(lines) + "\n"


def tab_double_group(res) -> str:
    dg = res["double_group"]
    lines = ["% Eq. (12): SU(2) -> 2I branching for half-integer j; Eq. (13): pair decompositions (antisymmetric)_a + (symmetric)_s",
             r"\begin{tabular}{ll}", r"\hline", r"$j$ & $V_j \downarrow 2I$ \\", r"\hline"]
    for r in dg["eq12"]:
        lines.append(f"${r['j']}$ & ${tex_decomposition(r['irreps'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}", "", r"\begin{tabular}{lll}", r"\hline",
              r"$\Gamma \otimes \Gamma$ & antisymmetric & symmetric \\", r"\hline"]
    for r in dg["eq13"]:
        lines.append(f"${tex_irrep(r['irrep'])}\\otimes {tex_irrep(r['irrep'])}$ & "
                     f"${tex_decomposition(r['antisym'])}$ & ${tex_decomposition(r['sym'])}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


_BASIS_LABELS = {"l=1,T1": (1, "T_{1u}"), "l=2,H": (2, "H_g"), "l=3,T2": (3, "T_{2u}"), "l=3,G": (3, "G_u"),
                 "l=4,G": (4, "G_g"), "l=4,H": (4, "H_g")}


def eq_bases(res) -> str:
    bf = res["harmonics"]["basis_functions_terms"]
    lines = ["% Eqs. (4)-(9): basis functions of the isotypic components (phi = golden ratio), generated by the projectors; Eq. (11): P6",
             r"\begin{align*}"]
    for key, (l, lab) in _BASIS_LABELS.items():
        funcs = bf[key]
        items = [f"{_tex_fname(n)} &= {tex_poly_terms(t)}" for n, t in funcs.items()]
        lines.append(f"\\ell={l},\\ {lab}:\\quad " + r" \\ ".join(items) + r" \\")
    lines.append(f"P_6 &= {tex_poly_terms(res['harmonics']['P6_terms'])} \\\\")
    hx = res["harmonics"]["hexad"]
    lines.append(rf"\sum_i (\hat a_i\cdot\hat n)^6 &= {tex_frac(hx['c'])}\,P_6 + {tex_frac(hx['e'])}\,r^6 \quad(\text{{Eq. 10; on the unit sphere }} r^6 = 1)")
    lines.append(r"\end{align*}")
    return "\n".join(lines) + "\n"


def _tex_fname(n: str) -> str:
    n = n.replace("^2", "^2")
    m = re.fullmatch(r"([a-z])_([0xyz])", n)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return n


def eq_invariants(res) -> str:
    rel = res["gl"]["relations"]
    mins = res["gl"]["weak_coupling_minima"]
    gs = res["gl"]["G_stratum"]

    def comb(d):
        out = ""
        for k, v in d.items():
            if str(v) in ("0",):
                continue
            c = tex_frac(str(v))
            neg = c.startswith("-")
            body = c[1:] if neg else c
            term = ("" if body == "1" else body + r"\,") + _tex_inv(k)
            out += (("-" if neg else "") if not out else (" - " if neg else " + ")) + term
        return out or "0"

    lines = ["% Eqs. (18)-(21): N_L in terms of I1, I2 (and N2 for G); Eq. (22): int |Delta|^4; Eq. (23): the G ground state",
             r"\begin{align*}"]
    for ch in ["T1", "T2", "G", "H"]:
        r = rel[ch]
        items = []
        for k in sorted((k for k in r if k != "quartic"), key=lambda k: (len(k), k)):
            items.append(f"{_tex_inv(k)} = {comb(r[k])}")
        lines.append(f"{tex_irrep(ch)}:&\\quad " + ",\\quad ".join(items) + r" \\")
    lines.append(r"\int|\Delta|^4\,\frac{d\Omega}{4\pi} &= " + ";\\quad ".join(
        f"{comb(rel[ch]['quartic'])}\\ ({tex_irrep(ch)})" for ch in ["T1", "T2", "G", "H"]) + r" \\")
    lines.append(rf"\eta_G &\propto (1,1,1,\kappa e^{{i\phi_0}}),\quad \kappa \simeq {gs['kappa']:.3f},\quad "
                 rf"\phi_0 \simeq {gs['phi0_deg']:.1f}^\circ,\quad R = {gs['R']:.5f}\ \text{{against}}\ "
                 rf"{res['gl']['G_ground_state']['null_cone_R']:.5f}\ \text{{(best null-cone state)}}, "
                 rf"\ I_2/I_1 \simeq {_tex_sci(gs['I2_over_I1'])} \\")
    lines.append(rf"R_{{\min}} &= {mins['T1']['R_min']:.4f}\ (T_1),\ {mins['T2']['R_min']:.4f}\ (T_2),\ "
                 rf"{mins['G']['R_min']:.5f}\ (G),\ {mins['H']['R_min']:.4f}\ (H)")
    lines.append(r"\end{align*}")
    return "\n".join(lines) + "\n"


def _tex_inv(k: str) -> str:
    return {"I1": "I_1", "I2": "I_2", "N0": "N_0", "N2": "N_2", "N4": "N_4", "N6": "N_6",
            "N2+N4": "N_2 + N_4", "quartic": r"\int|\Delta|^4"}.get(k, k)


EMITTERS = {"tab_charI": tab_charI, "tab_char2I": tab_char2I, "tab_branching": tab_branching,
            "tab_shells": tab_shells, "tab_restrict": tab_restrict, "tab_GL": tab_GL, "tab_states": tab_states,
            "tab_Hcand": tab_Hcand, "tab_D12": tab_D12, "tab_molien": tab_molien,
            "tab_double_group": tab_double_group, "eq_bases": eq_bases, "eq_invariants": eq_invariants}


def write_tables(res: Dict[str, Any], outdir: str = "tables") -> List[str]:
    """Write every fragment into ``outdir``; returns the list of paths.  All fragments are
    rendered before the first file is opened, so a results.json that lacks a key raises
    ``KeyError`` without leaving partially written files behind."""
    rendered = {name: fn(res) for name, fn in EMITTERS.items()}
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, text in rendered.items():
        path = os.path.join(outdir, name + ".tex")
        with open(path, "w") as f:
            f.write(text)
        paths.append(path)
    return paths
